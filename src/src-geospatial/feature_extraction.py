import argparse
import glob
import itertools
import json
import logging
import os
import random
import shutil
import sys
import tempfile
from os import path
from shutil import move

import boto3
import dask.dataframe as dd
import geopandas as gp
import pandas as pd
import rasterio
import rasterio.mask
import rasterio.merge
from rasterstats import zonal_stats

from utils.satellite_image_preparation_helper import (
    crop_image_with_fips_shape,
    get_first_day_in_isoweek,
    reproject_tif_with_template_raster,
    write_rasterio_image_from_numy_array,
)


log_format = "%(asctime)s %(levelname)s %(message)s"
logging.basicConfig(format=log_format)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Get rid of "Found credentials in environment variables."
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)

# Inject environment variable through SageMaker Processing job' env parameter
COUNTIES_GEOJSON_FILE_PATH = os.environ.get("COUNTIES_GEOJSON_FILE_PATH")
input_crop_mask_bucket_name = os.environ.get("INPUT_CROP_MASK_BUCKET_NAME")
input_crop_mask_prefix = os.environ.get("INPUT_CROP_MASK_PREFIX")
output_bucket_name = os.environ.get("OUTPUT_BUCKET_NAME")
spectral_indices = os.environ.get("SPECTRAL_INDICES")

s3_client = boto3.client("s3")


def sort_bands(bands_tiles, sort_order):
    """Reorder input bands"""

    res = {tile.split("/")[-1].split(".")[0].split("_")[-1]: tile for tile in bands_tiles}
    res_list = list(enumerate(res))

    return [res[y[1]] for x in sort_order for y in res_list if y[1] == x]


def stack_bands_into_single_tile(mapping, temp_dirpath, sort_order, isoweek):

    bands_to_merge = [
        {
            "key": key,
            "tiles_to_merge": [
                f"{item['mosaic_s3_path']}" for item in group
            ],
        }
        for key, group in itertools.groupby(mapping, key=lambda x: x["starttime"])
    ]

    tiles_with_merged_bands = []
    for item in bands_to_merge:
        bands_tiles = item["tiles_to_merge"]

        bands_tiles = " ".join(sort_bands(bands_tiles, sort_order))
        bands_tiles = bands_tiles.replace("s3://", "/vsis3/")
        merged_bands_output_path = os.path.join(temp_dirpath, f"merged_{isoweek}.tif")
        tiles_with_merged_bands.append(merged_bands_output_path)
        cmd = f"rio stack {bands_tiles} -o {merged_bands_output_path}" " --overwrite"
        os.system(cmd)

    return tiles_with_merged_bands


def get_tiles_for_fips_isoweek_year(fips, starttime, endtime, mapping_df, band_names):
    res = mapping_df[
        (mapping_df["FIPS"] == fips)
        & (mapping_df["starttime"] == starttime)
        & (mapping_df["endtime"] == endtime)
    ]
    res = res[res["band_name"].isin(band_names)]
    return res.to_dict(orient="records")


def handler(event, geo_counties_fips, mapping_df, temp_dirpath):
    starttime = event["starttime"]
    endtime = event["endtime"]
    fips = str(event["fips"])
    type_of_crop = event["type_of_crop"]
    band_names = spectral_indices.split(",")

    year, isoweek = event["year"], event["week"]

    try:
        logger.info(f"Using temp_dirpath = {temp_dirpath}")
        # ====================================================================
        #  Get mapping between fips, isoweek, year and geotiffs paths in s3
        # ====================================================================

        logger.info("Get mapping between (fips, week, year) and s3 paths from mapping file")
        
        fips_tile_paths = get_tiles_for_fips_isoweek_year(
            fips, starttime, endtime, mapping_df, band_names
        )


        tiles_in_s3 = [
            f"{item['mosaic_s3_path']}"
            for item in fips_tile_paths
        ]

        logger.info(f"Band file in s3 to stack: {tiles_in_s3}")

        # ====================================================================
        #  Merge bands into a single multi-channel mosaic
        # ====================================================================

        if not path.exists(f"{temp_dirpath}/merged_{isoweek}.tif"):

            logger.info("Merge all bands into a single multi-channel mosaic")
            stack_bands_into_single_tile(fips_tile_paths, temp_dirpath, band_names, isoweek)

        # ====================================================================
        #  Crop the mosaic using county's shape
        # ====================================================================
        logger.info("Crop the combined mosaic using the county's shape")

        out_image, out_transform, out_meta, raster_meta = crop_image_with_fips_shape(
            f"{temp_dirpath}/merged_{isoweek}.tif",
            fips,
            geo_counties_fips,
            geojson_projection="EPSG:4326",
        )

        # Prepare the output raster image metadata
        output_raster_metadata = raster_meta.copy()
        output_raster_metadata["transform"] = out_transform
        output_raster_metadata["height"] = out_image.shape[1]
        output_raster_metadata["width"] = out_image.shape[2]

        # Write the cropped raster image.
        mosaic_prefix = f"s2-l2a_sat_image_{fips}"
        write_rasterio_image_from_numy_array(
            f"{temp_dirpath}/{mosaic_prefix}_cropped.tif",
            out_image,
            output_raster_metadata,
        )

        # ====================================================================
        #  Crop the mask file using county's shape and upload to s3
        # ====================================================================
        crop_mask_s3_path = (
            f"s3://{input_crop_mask_bucket_name}"
            f"/{input_crop_mask_prefix}"
            f"/{year}/{fips}/cdl_{type_of_crop}_mask_{fips}.tif"
        )

        logger.info("Crop the crop_mask image with the fips geometry")
        (
            crop_mask_out_image,
            mask_out_transform,
            mask_out_meta,
            mask_raster_meta,
        ) = crop_image_with_fips_shape(
            crop_mask_s3_path, fips, geo_counties_fips, geojson_projection="EPSG:4326"
        )

        mask_output_raster_metadata = mask_raster_meta.copy()
        mask_output_raster_metadata["transform"] = mask_out_transform
        mask_output_raster_metadata["height"] = crop_mask_out_image.shape[1]
        mask_output_raster_metadata["width"] = crop_mask_out_image.shape[2]

        write_rasterio_image_from_numy_array(
            f"{temp_dirpath}/cdl_{type_of_crop}_mask_{fips}_cropped.tif",
            crop_mask_out_image,
            mask_output_raster_metadata,
        )

        # ====================================================================
        #  Reproject satellite image to use the same crop_mask's projection
        # ====================================================================

        logger.info("Reproject the satellite image to the crop_mask's projection")

        src_file = f"{temp_dirpath}/{mosaic_prefix}_cropped.tif"
        dst_file = f"{temp_dirpath}/{mosaic_prefix}_cropped_resized.tif"

        # reproject sat image to Mask projection using WKT file
        # reproject_tiff_with_wkt(src_file, local_wkt_file, dst_file)
        # template raster used to deduce crs, affline, etc.

        mask_cropped_path = f"{temp_dirpath}/cdl_{type_of_crop}_mask_{fips}_cropped.tif"
        template_raster = f"{temp_dirpath}/cdl_{type_of_crop}_mask_{fips}_cropped.tif"

        reproject_tif_with_template_raster(src_file, dst_file, template_raster)

        print(f"resized_file {dst_file} size: {rasterio.open(mask_cropped_path).read(1).shape}")
        print(
            f"small file {mask_cropped_path} size: {rasterio.open(mask_cropped_path).read(1).shape}"
        )

        # ====================================================================
        #  Applying the crop mask and reproject the mosaic to the cells polygons's crs
        # ====================================================================

        logger.info("Upload the crop mosaic to s3")
        s3_client.upload_file(
            f"{temp_dirpath}/{mosaic_prefix}_cropped_resized.tif",
            output_bucket_name,
            f"crop-mosaic/{isoweek}/mosaic_{type_of_crop}_{year}_{fips}.tif",
        )

        rio_calc = "(* (read 1) (read 2))"

        cmd = (
            f'rio calc "{rio_calc}" "{temp_dirpath}/{mosaic_prefix}_cropped_resized.tif"'
            f' "{temp_dirpath}/cdl_{type_of_crop}_mask_{fips}_cropped.tif"'
            f' -o "{temp_dirpath}/{mosaic_prefix}_masked.tif" --overwrite'
        )
        os.system(cmd)

        logger.info("Upload the crop mosaic [masked] to s3")
        s3_client.upload_file(
            f"{temp_dirpath}/{mosaic_prefix}_masked.tif",
            output_bucket_name,
            f"crop-mosaic-masked/{isoweek}/mosaic_{type_of_crop}_{year}_{fips}.tif",
        )

        cmd = (
            f"rio warp --dst-crs EPSG:4326 '{temp_dirpath}/{mosaic_prefix}_masked.tif'"
            f" -o '{temp_dirpath}/{mosaic_prefix}_masked_resized.tif' --overwrite"
        )
        os.system(cmd)

        # ====================================================================
        #  Create zonal statistics by using the cells polygons
        # ====================================================================

        logger.info("Compute zonal statistics for the cells polygons")

        all_stats = []

        polygons_shp_file = f"/opt/ml/processing/input/polygons/cell_polygons_{fips}.shp"
        rasters_file = f"{temp_dirpath}/{mosaic_prefix}_masked_resized.tif"

        zonal_polygons = gp.read_file(polygons_shp_file)

        for band_idx, band_name in enumerate(band_names):

            stats = zonal_stats(polygons_shp_file, rasters_file, band=band_idx + 1)

            print(
                f"band name {band_name}, zonal stats {random.sample(stats, 2)}",
                end="\n\n",
            )

            zonal_stats_df = pd.DataFrame.from_records(stats)
            zonal_stats_df = zonal_stats_df.add_suffix("_{}".format(band_name))
            all_stats.append(zonal_stats_df)

        all_stats_gf = pd.concat(all_stats, axis=1)
        all_stats_gf = pd.concat([all_stats_gf, zonal_polygons], axis=1)

        # upload to s3 the concatenated zonal statistics for each isoweek/ fips combination
        all_stats_gf.to_csv(
            f"s3://{output_bucket_name}/data/zonal-statistics-allbands/{type_of_crop}/{year}/isoweek-{isoweek}/"
            f"zonal_stats_{fips}.csv",
            index=False,
        )

        result = os.listdir(f"{temp_dirpath}")

        logger.info("Finished successfully")
        return {
            "result": "success",
            "fips": f"{fips}",
            "isoweek": f"{isoweek}",
            "year": f"{year}",
            "output": result,
        }
    except Exception as e:
        error_msg = f"=== Error processing FIPS: {fips}  Week: {isoweek} ==="
        logger.error(error_msg)
        raise e


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--crop-type", type=str, required=True)
    args, _ = parser.parse_known_args()

    crop_type = args.crop_type

    sat_images_metadata_mapping = "/opt/ml/processing/input/sat_images_metadata_mapping"

    temp_dirpath = tempfile.mkdtemp()

    metadata_mapping_files = os.listdir(sat_images_metadata_mapping)


    if metadata_mapping_files:
        sat_images_metadata_mapping_df = dd.read_csv(
            f"{sat_images_metadata_mapping}/*/*", dtype={"FIPS": object}
        ).compute()

    else:
        logger.error(
            "No metadata mapping files found in this executer under"
            f" {sat_images_metadata_mapping}"
        )
        logger.info("Make sure you are copying the metadata files to the correct location.")
        logger.info(
            "This can be a normal behaviour if the number of executors"
            " is bigger than the number of partitions"
        )
        logger.info("Exiting..")
        sys.exit(0)

    metadata_mapping_dict = (
        sat_images_metadata_mapping_df[["starttime", "endtime", "FIPS", "year", "week"]]
        .drop_duplicates()
        .to_dict(orient="records")
    )

    logger.info("Load counties-fips geojson file with geopandas")
    geo_counties_fips = gp.read_file(COUNTIES_GEOJSON_FILE_PATH)
    geo_counties_fips["FIPS"] = geo_counties_fips["STATE"] + geo_counties_fips["COUNTY"]
    print(geo_counties_fips.head())

    for mapping in metadata_mapping_dict:

        mapping["fips"] = mapping["FIPS"]
        mapping["type_of_crop"] = crop_type

        handler(mapping, geo_counties_fips, sat_images_metadata_mapping_df, temp_dirpath)
