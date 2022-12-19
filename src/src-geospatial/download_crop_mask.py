import argparse
import itertools
import json
import logging
import os

import boto3
import dateutil
import pandas as pd
import requests
from bs4 import BeautifulSoup  # beautifulsoup4 lxml

logging.basicConfig(level=logging.INFO)

# Injected from SageMaker Processor
OUTPUT_CROP_MASK_BUCKET_NAME = os.environ.get("OUTPUT_CROP_MASK_BUCKET_NAME")
OUTPUT_CROP_MASK_PREFIX = os.environ.get("OUTPUT_CROP_MASK_PREFIX")
FIPS_STATS_CSV = "/opt/ml/processing/input/fips_stats/fips_county_stats.csv"

s3_client = boto3.client("s3")


def _s3_key_exists(bucket, key):
    """return boolean on whehter an s3 key's exists"""
    response = s3_client.list_objects_v2(
        Bucket=bucket,
        Prefix=key,
    )
    for obj in response.get("Contents", []):
        if obj["Key"] == key:
            return True
    return False


# ===============================================================================
# Get FIPS code by name
# ===============================================================================
def get_county_fips_by_state(state=None, us_fips_csv=FIPS_STATS_CSV):
    df = pd.read_csv(us_fips_csv, usecols=["FIPS", "STNAME", "CTYNAME"], dtype=str)
    if state:
        df = df[df["STNAME"] == state]
    return df


def getFIPSByName(county_name, state="Iowa"):
    fips_df = pd.read_csv(FIPS_STATS_CSV)
    iowa_fips = fips_df[(fips_df["STNAME"] == state) & (fips_df["CTYNAME"] == f"{county_name}")]
    FIPS = iowa_fips["FIPS"].astype(str).tolist()[0]
    return FIPS


# ===============================================================================
# cdl_helpers functions
# ===============================================================================
def extractCDLByValues(tif_filename, crop_types):
    """Return CDL Image by corp type values"""
    # Build URL
    url = "https://nassgeodata.gmu.edu/axis2/services/CDLService/ExtractCDLByValues?"
    values_str = ",".join([str(i) for i in crop_types])

    # Add flename and cdl values as parameters
    params = {"file": str(tif_filename), "values": values_str}

    # Perform the request
    try:
        req = requests.get(url, params=params, verify=True, timeout=30)
    except Exception as e:
        print(f"error {str(e)}")
        return
    resp = BeautifulSoup(req.text, features="lxml")
    new_tif_filename = resp.html.body.returnurl.text

    return new_tif_filename


def GetCDLFileRequest(url, params):
    # Perform the request.
    try:
        req = requests.get(url, params=params, verify=True, timeout=30)
    except Exception as e:
        print(f"error {str(e)}")
        return
    resp = BeautifulSoup(req.text, features="lxml")
    tif_filename = resp.html.body.returnurl.text

    return tif_filename


def getCDLByFIPS(fips, year=None):
    """Return a TIF Image for a given Fips (state+county Fips)"""

    # Build URL
    url = "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile?"

    # Add year, and bbox as parameters
    params = {"year": str(year), "fips": fips}

    tif_filename = GetCDLFileRequest(url, params)

    return tif_filename


def save_crop_mask(crop_index, fips, year=2020):
    crop_mapping = {1: "corn", 5: "soybeans"}
    crop_name = crop_mapping[crop_index]

    tif_file = extractCDLByValues(getCDLByFIPS(fips, year=year), [crop_index])

    # Request the GEOTiff image
    response = requests.get(tif_file)

    # save locally
    target_filename = f"/tmp/{str(year)}/{fips}/cdl_{crop_name}_mask_{fips}.tif"

    os.makedirs(os.path.dirname(target_filename), exist_ok=True)
    with open(target_filename, "wb") as f:
        f.write(response.content)
    return target_filename


def upload_to_s3(filename, bucket, key):
    s3 = boto3.client("s3")
    with open(filename, "rb") as f:
        s3.upload_fileobj(f, bucket, key)
    print(f"{filename} uploaded to s3://{bucket}/{key}")


def handler(event, context):
    print(event)
    if "fips_list" not in event:
        raise RuntimeError('parameter "fips_list" not in event')
    if "year" not in event:
        raise RuntimeError('parameter "year" not in event')
    if "crop_name" not in event:
        raise RuntimeError('parameter "crop_name" not in event')

    year = int(event["year"])
    crop_name = event["crop_name"]
    fips_list = event["fips_list"]

    crop_name_to_crop_index = {"corn": 1, "soybeans": 5}

    crop_index = crop_name_to_crop_index[crop_name]

    target_bucket = OUTPUT_CROP_MASK_BUCKET_NAME
    prefix = OUTPUT_CROP_MASK_PREFIX  # prefix for masks

    for fips in fips_list:
        crop_mask_s3_key = f"{prefix}/{year}/{fips}/cdl_{crop_name}_mask_{fips}.tif"

        if _s3_key_exists(target_bucket, crop_mask_s3_key):
            print(f"File {crop_mask_s3_key} exists --> skipping")
            continue
        try:
            print(f"==== Mirroring crop mask to S3 for crop_type:{crop_name} fips: {fips}")
            local_filename = save_crop_mask(crop_index, fips, year)
            target_filename = local_filename.replace("/tmp/", "")
            upload_to_s3(local_filename, target_bucket, f"{prefix}/{target_filename}")
        except Exception as e:
            print(f"Failed with {e}")

    return {"statusCode": 200, "body": json.dumps("Crop mask saved to S3")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--starttime", type=str, required=True)
    parser.add_argument("--endtime", type=str, required=True)
    parser.add_argument("--fips-list", type=str, help="comma sperated list of fips")
    parser.add_argument("--crop-type", type=str)
    args, _ = parser.parse_known_args()

    years = list(
        set(
            range(
                dateutil.parser.parse(args.starttime).year,
                dateutil.parser.parse(args.endtime).year + 1,
            )
        )
    )

    fips_list = args.fips_list.split(",")
    context = {}

    params = list(itertools.product(years, [args.crop_type]))
    for param in params:
        # Make more dynamic
        year = param[0]
        crop_name = param[1]
        event = {"fips_list": fips_list, "year": year, "crop_name": crop_name}

        handler(event, context)
