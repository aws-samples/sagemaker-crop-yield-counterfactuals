import os
from datetime import date, datetime, timedelta

import numpy as np
import rasterio


def get_first_day_in_isoweek(year, week):
    ret = datetime.strptime("%04d-%02d-1" % (year, week), "%Y-%W-%w")
    if date(year, 1, 4).isoweekday() > 4:
        ret -= timedelta(days=7)
    return ret

def create_county_shape_file(geo_counties_fips, fips, temp_folder="/tmp"):
    """Create's county shapefie from geopandas geometry"""
    county = geo_counties_fips[geo_counties_fips["FIPS"] == fips]
    county.to_file(driver="ESRI Shapefile", filename=f"{temp_folder}/{fips}.shp")
    return f"{temp_folder}/{fips}.shp"


def reproject_tif_with_template_raster(src_file, target_file, template_raster):
    """Reporject a tif file using a template raster."""
    print(src_file, target_file, template_raster)

    command = f"rio warp {src_file} {target_file} --like {template_raster} --overwrite"
    if os.system(command) == 0:
        return f"{target_file}"
    else:
        raise RuntimeError(
            "Error in rio warp, within reproject_tif_with_template_raster"
        )


def write_rasterio_image_from_numy_array(output_filename, numpy_array, ras_metadata):
    # https://gis.stackexchange.com/a/324693
    with rasterio.open(output_filename, "w", **ras_metadata) as dst:
        dst.write(numpy_array)


# source: https://gis.stackexchange.com/questions/371065/apply-same-coordinate-system-to-raster-image-and-geojson-with-rasterio
def crop_image_with_fips_shape(
    image_path, fips, geo_counties_fips, geojson_projection="EPSG:4326"
):
    """Crops an input image with the geojson county's shape"""
    with rasterio.open(image_path) as src:
        raster_meta = src.profile

        print(src.crs)
        allfeatures = geo_counties_fips[geo_counties_fips["FIPS"] == fips][
            "geometry"
        ].values
        allfeatures_reprojected = rasterio.warp.transform_geom(
            geojson_projection, src.crs, allfeatures
        )

        out_image, out_transform = rasterio.mask.mask(
            src, allfeatures_reprojected, crop=True
        )
        out_meta = src.meta

        return out_image, out_transform, out_meta, raster_meta
