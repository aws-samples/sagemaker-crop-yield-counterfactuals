from datetime import datetime
import argparse
import json
import logging
import os
import time
import uuid

import boto3
import botocore
import geopandas as gp

from utils.fips_to_satellite_tiles_metadata_helper import (
    create_fips_isoweek_year_satellite_tiles_mapping,
    list_satellite_images_in_s3,
)

log_format = "%(asctime)s %(levelname)s %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO)
logger = logging.getLogger()


axisClient = boto3.client(service_name="sagemaker-geospatial", region_name=os.environ["REGION"])

# Used to get boundary polygon coordinates for a given county's based on FIPS
COUNTIES_GEOJSON = os.environ["COUNTIES_GEOJSON_FILE_PATH"]

AXIS_REQUEST_MANIFEST_PATH = "/opt/ml/processing/input/axis_requests_manifests/"

AXIS_ROLE_ARN = os.environ["AXIS_ROLE_ARN"]

s3_resource = boto3.resource("s3")


def fips_to_polygon_coordinates(fips):
    """Get polygon coordinates from fips id."""

    geo_df = gp.read_file(COUNTIES_GEOJSON)
    geo_df["FIPS"] = geo_df["STATE"] + geo_df["COUNTY"]
    geometries = geo_df[geo_df["FIPS"].isin(fips)]["geometry"]
    geometries_union = gp.GeoDataFrame(crs=geometries.crs, geometry=[geometries.unary_union])
    geom_type = geometries_union.geom_type[0]
    geometries_union = geometries_union["geometry"].iloc[0]

    if geom_type == "MultiPolygon":
        return [list(geom.boundary.coords) for geom in geometries_union.geoms]
    elif geom_type == "Polygon":
        return [list(geometries_union.boundary.coords)]
    else:
        raise ValueError(f"Unknown Geom type {geom_type}")


def list_requests_manifest_files():
    logger.info("Looking for Axis requests manifests file under" f" {AXIS_REQUEST_MANIFEST_PATH}")

    axis_requests_manifests = os.listdir(AXIS_REQUEST_MANIFEST_PATH)
    logger.info(f"Founds {len(axis_requests_manifests)} manifest files")
    logger.info(axis_requests_manifests)
    logger.info("Loading request manifests file")

    axis_requests_configs = []

    for manifest_file in axis_requests_manifests:
        mnifest_file_path = f"{AXIS_REQUEST_MANIFEST_PATH}/{manifest_file}"
        with open(mnifest_file_path, "r") as manifest:
            axis_requests_configs.append(json.load(manifest))
    return axis_requests_configs


def run_geomosaic_earth_observation_job(eoj_arn, output_bucket_name, key_prefix):
    """Run a geomosaic EOJ to merge multiple rasters into one raster for each band."""


    eoj_config = {"JobConfig": {"GeoMosaicConfig": {"AlgorithmName": "NEAR"}}}

    eojParams = {
        "Name": "geomosaic",
        "InputConfig": {"PreviousEarthObservationJobArn": eoj_arn},
        **eoj_config,
        "ExecutionRoleArn": AXIS_ROLE_ARN,
    }

    eoj_response = axisClient.start_earth_observation_job(**eojParams)

    job_arn = eoj_response["Arn"]

    while True:
        # Sleep for 1 minute before calling to get the job status
        time.sleep(60)

        eoj_status_response = axisClient.get_earth_observation_job(Arn=job_arn)
        job_status = eoj_status_response["Status"]
        logger.info(f"Geomosaic EOJ status is {job_status}")

        if job_status == "COMPLETED":
            logger.info("Geomosaic EOJ finished successfuly.")
            break

        if job_status == "FAILED":
            logger.info("Geomosaic EOJ Failed.")
            logger.info(eoj_status_response)
            break
            

    return job_arn, job_status, eoj_response["Arn"]


def run_bandmath_earth_observation_job(eoj_arn, spectral_indices, output_bucket_name, key_prefix):
    """Run EOJ to compute the spectral indices."""

    eoj_config = {
        "JobConfig": {
            "BandMathConfig": {"CustomIndices": {"Operations": []}},
        }
    }

    for indices in spectral_indices:
        eoj_config["JobConfig"]["BandMathConfig"]["CustomIndices"]["Operations"].append(
            {"Name": indices[0], "Equation": indices[1][1:-1]}
        )

    eojParams = {
        "Name": "bandmath",
        "InputConfig": {"PreviousEarthObservationJobArn": eoj_arn},
        **eoj_config,
        "ExecutionRoleArn": AXIS_ROLE_ARN,
    }

    eoj_response = axisClient.start_earth_observation_job(**eojParams)

    job_arn = eoj_response["Arn"]

    while True:
        # Sleep for 1 minute before calling to get the job status
        time.sleep(60)

        eoj_status_response = axisClient.get_earth_observation_job(Arn=job_arn)
        job_status = eoj_status_response["Status"]
        logger.info(f"Bandmath EOJstatus is {job_status}")

        if job_status == "COMPLETED":
            logger.info("Bandmath EOJ finished successfuly.")
            break

        if job_status == "FAILED":
            logger.info("Bandmath EOJ Failed.")
            logger.info(eoj_status_response)
            break

    # export results of an EarthObservationJob to an S3 location.

    eojParamsExport = {
        "Arn": eoj_response["Arn"],
        "ExecutionRoleArn": AXIS_ROLE_ARN,
        "OutputConfig": {"S3Data": {"S3Uri": f"s3://{output_bucket_name}/{key_prefix}"}},
    }

    eoj_response_export = axisClient.export_earth_observation_job(**eojParamsExport)

    job_arn_export = eoj_response_export["Arn"]

    while True:
        # Sleep for 1 minute before calling to get the job status
        time.sleep(60)

        eoj_status_response = axisClient.get_earth_observation_job(Arn=job_arn_export)
        job_status = eoj_status_response["ExportStatus"]
        logger.info(f"Export EOJstatus is {job_status}")

        if job_status == "SUCCEEDED":
            logger.info("Export EOJ finished successfuly.")
            break

        if job_status == "FAILED":
            logger.info("Export EOJ Failed.")
            logger.info(eoj_status_response)
            break

    return job_arn, job_status


def run_cloud_removal_earth_observation_job(
    start_time,
    end_time,
    county_fips,
    data_collections_names,
    data_collections_arns,
    output_bucket_name,
    key_prefix,
):
    """Run cloud_removal EOJ to remove the clouds."""

    request_polygon_coordinates = fips_to_polygon_coordinates(county_fips)

    eoj_input_config = {
        "RasterDataCollectionQuery": {
            "RasterDataCollectionArn": data_collections_arns[1],
            "AreaOfInterest": {
                "AreaOfInterestGeometry": {
                    "PolygonGeometry": {"Coordinates": request_polygon_coordinates}
                }
            },
            "TimeRangeFilter": {"StartTime": start_time, "EndTime": end_time},
            "PropertyFilters": {
                "Properties": [{"Property": {"EoCloudCover": {"LowerBound": 0, "UpperBound": 10}}}],
                "LogicalOperator": "AND",
            },
        }
    }

    eoj_config = {
        "JobConfig": {
            "CloudRemovalConfig": {
                "AlgorithmName": "INTERPOLATION",
                "InterpolationValue": "-9999",
                "TargetBands": ["red", "green", "blue", "nir", "swir16"],
            },
        }
    }

    eojParams = {
        "Name": "cloudremoval",
        "InputConfig": eoj_input_config,
        **eoj_config,
        "ExecutionRoleArn": AXIS_ROLE_ARN,
    }

    eoj_response = axisClient.start_earth_observation_job(**eojParams)

    job_arn = eoj_response["Arn"]

    while True:
        # Sleep for 1 minute before calling to get the job status
        time.sleep(60)

        eoj_status_response = axisClient.get_earth_observation_job(Arn=job_arn)
        job_status = eoj_status_response["Status"]
        logger.info(f"Cloud Removal EOJ status is {job_status}")

        if job_status == "COMPLETED":
            logger.info("Cloud Removal EOJ finished successfuly.")
            break

        if job_status == "FAILED":
            logger.info("Cloud Removal EOJ Failed.")
            logger.info(eoj_status_response)
            break

    return job_arn, job_status, eoj_response["Arn"]


def run_resample_earth_observation_job(eoj_arn, output_bucket_name, key_prefix):
    """Run rasample EOJ to resample the rasters to a 30m resolution
    Note: Required to match the crop masks resolution."""

    eoj_config = {
        "JobConfig": {
            "ResamplingConfig": {
                "OutputResolution": {"UserDefined": {"Value": 30, "Unit": "METERS"}},
                "AlgorithmName": "NEAR",
            },
        }
    }

    eojParams = {
        "Name": "resample",
        "InputConfig": {"PreviousEarthObservationJobArn": eoj_arn},
        **eoj_config,
        "ExecutionRoleArn": AXIS_ROLE_ARN,
    }

    eoj_response = axisClient.start_earth_observation_job(**eojParams)

    job_arn = eoj_response["Arn"]

    while True:
        # Sleep for 1 minute before calling to get the job status
        time.sleep(60)

        eoj_status_response = axisClient.get_earth_observation_job(Arn=job_arn)
        job_status = eoj_status_response["Status"]
        logger.info(f"Resample EOJ status is {job_status}")

        if job_status == "COMPLETED":
            logger.info("Resample EOJ finished successfuly.")
            break

        if job_status == "FAILED":
            logger.info("Resample EOJ Failed.")
            logger.info(eoj_status_response)
            break

    # export results of an EarthObservationJob to an S3 location.

    eojParamsExport = {
        "Arn": eoj_response["Arn"],
        "ExecutionRoleArn": AXIS_ROLE_ARN,
        "OutputConfig": {"S3Data": {"S3Uri": f"s3://{output_bucket_name}/{key_prefix}"}},
    }

    eoj_response_export = axisClient.export_earth_observation_job(**eojParamsExport)

    job_arn_export = eoj_response_export["Arn"]

    while True:
        # Sleep for 1 minute before calling to get the job status
        time.sleep(60)

        eoj_status_response = axisClient.get_earth_observation_job(Arn=job_arn_export)
        job_status = eoj_status_response["ExportStatus"]
        logger.info(f"Export EOJstatus is {job_status}")

        if job_status == "SUCCEEDED":
            logger.info("Export EOJ finished successfuly.")
            break

        if job_status == "FAILED":
            logger.info("Export EOJ Failed.")
            logger.info(eoj_status_response)
            break

    return job_arn, job_status, eoj_response["Arn"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output-bucket", type=str, required=True)
    parser.add_argument("--metadata-key", type=str, required=True)

    args, _ = parser.parse_known_args()
    unique_folder_name = uuid.uuid4().hex

    output_bucket_name = args.output_bucket
    output_metadata_key = args.metadata_key
    
    bucket_resource = s3_resource.Bucket(output_bucket_name)

    axis_requests_configs = list_requests_manifest_files()

    logger.info(f"Found {len(axis_requests_configs)} requests manifest files")

    logger.info("Listing raster data collections")
    # Perform raster operations using axisClient
    data_collections = axisClient.list_raster_data_collections()
    data_collections = data_collections["RasterDataCollectionSummaries"]
    data_collections_names = [c["Name"] for c in data_collections]
    data_collections_arns = [c["Arn"] for c in data_collections]
    logger.info("data_collections_arns, data_collections_names")
    print(data_collections_arns, data_collections_names)

    # Run Axis EOJ jobs for all available request manifest config files
    for axis_request_config in axis_requests_configs:
        logger.info(axis_request_config)
        start_time = axis_request_config["startime"]
        end_time = axis_request_config["endtime"]
        county_fips = axis_request_config["fips"].split(",")
        week = axis_request_config["week"]
        year = axis_request_config["year"]
        spectral_indices = axis_request_config["spectralindices"]
        key_prefix = f"geospatial-results/{datetime.utcnow():%Y-%m-%d-%H%M}-{week}/"
        print("county_fips type", type(county_fips))

        logger.info(
            f"Running chained EOJs for starttime {start_time}"
            f"endtime {end_time} and county fips {county_fips}"
        )

        # ====================================================================
        #  Cloud Removal EOJ
        # ====================================================================

        job_arn, job_status, eoj_arn = run_cloud_removal_earth_observation_job(
            start_time,
            end_time,
            county_fips,
            data_collections_names,
            data_collections_arns,
            output_bucket_name,
            key_prefix,
        )

        if job_status == "FAILED":
            logger.info("Skipping generating metadata mapping files.")
            continue

        # ====================================================================
        #  Geomosaic EOJ
        # ====================================================================

        job_arn, job_status, eoj_arn = run_geomosaic_earth_observation_job(
            eoj_arn, output_bucket_name, key_prefix
        )

        if job_status == "FAILED":
            logger.info("Skipping generating metadata mapping files.")
            continue

        # ====================================================================
        #  Resample EOJ
        # ====================================================================

        job_arn, job_status, eoj_arn = run_resample_earth_observation_job(
            eoj_arn, output_bucket_name, key_prefix
        )

        if job_status == "FAILED":
            logger.info("Skipping generating metadata mapping files.")
            continue

        # ====================================================================
        #  Bandmath EOJ
        # ====================================================================

        job_arn, job_status = run_bandmath_earth_observation_job(
            eoj_arn, spectral_indices, output_bucket_name, key_prefix
        )
        satellite_images = list_satellite_images_in_s3(bucket_resource, key_prefix)
        # keep only file that ends with .tif to avoid temporary files.

        satellite_images = [img for img in satellite_images if img.endswith(".tif")]

        logger.info(f"EOJ results: {satellite_images}")
        logger.info("Create tif file to metadata mapping file")

        for fips in county_fips:
            # Generate mapping from current geospatial results
            mapping_df = create_fips_isoweek_year_satellite_tiles_mapping(
                satellite_images, fips, start_time, end_time, week, year
            )

            unique_file_name = uuid.uuid4().hex
            mapping_df.to_csv(
                f"s3://{output_bucket_name}/{output_metadata_key}"
                f"{week}/{unique_file_name}.csv",
                index=False,
            )
