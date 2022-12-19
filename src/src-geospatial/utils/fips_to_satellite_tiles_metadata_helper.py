import boto3
import pandas as pd


s3_client = boto3.client("s3")
s3_resource = boto3.resource("s3")


def list_files_under_s3_key_prefix(bucket, s3_key_prefix, endswith=None):
    if endswith:
        return [
            f"s3://{s3_object.bucket_name}/{s3_object.key}"
            for s3_object in bucket.objects.filter(Prefix=s3_key_prefix)
            if s3_object.key.endswith(endswith)
        ]
    else:
        return [
            f"s3://{s3_object.bucket_name}/{s3_object.key}"
            for s3_object in bucket.objects.filter(Prefix=s3_key_prefix)
        ]


def list_satellite_images_in_s3(bucket, key_prefix):
    return list_files_under_s3_key_prefix(bucket, key_prefix)


def get_satellite_images_record_with_metadata(
    fips, startime, endtime, s3_sallite_image_path, week, year
):
    satellite_image_file_name = s3_sallite_image_path.split("/")[-1]
    filename_parts = satellite_image_file_name.split("_")
    return {
        "geographic_mosaic_filename": satellite_image_file_name,
        "starttime": startime,
        "endtime": endtime,
        "band_name": filename_parts[-1].split(".")[0],
        "mosaic_s3_path": s3_sallite_image_path,
        "FIPS": fips,
        "week": week,
        "year": year,
    }


def create_fips_isoweek_year_satellite_tiles_mapping(
    satellite_images_s3_keys, fips, startime, endtime, week, year
):
    """Fips to isoweek, year, satellite tiles mapping metadata."""
    satellite_images_with_metadata = pd.DataFrame(
        [
            get_satellite_images_record_with_metadata(
                fips, startime, endtime, s3_sat_image_path, week, year
            )
            for s3_sat_image_path in satellite_images_s3_keys
        ]
    )

    return satellite_images_with_metadata
