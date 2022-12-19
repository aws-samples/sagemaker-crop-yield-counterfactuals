import os
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import s3fs
import seaborn as sns
from matplotlib.transforms import Bbox
from shapely import wkb

warnings.filterwarnings("ignore")  # silence warnings


def return_tiled_ids_roi(gpd_cells, tile_grids, region):

    # disolve by cells IDs
    gpd_cells = gpd_cells[gpd_cells["region"] == region]
    id_10_agg = gpd_cells[["id_10", "geometry", "cnty_nm"]].dissolve(by="id_10", aggfunc="sum")

    # drop the z coordinates from the tile grid geometry
    def drop_z(geom):
        return wkb.loads(wkb.dumps(geom, output_dimension=2))

    tile_grids.geometry = tile_grids.geometry.transform(drop_z)

    # overlay the tile grids and the cells aggregated geometries in order to
    # obtain the tiles intersect
    id_10_agg = id_10_agg.reset_index()
    tiles_intersect = id_10_agg.overlay(tile_grids, how="intersection")

    # plot the overlays (avoid text overlapping)
    ax = id_10_agg.plot(cmap="gray", figsize=(15, 7))
    tiles_intersect.plot(ax=ax, column="Name", cmap="tab20c")

    tiles_intersect = tiles_intersect.drop_duplicates(subset=["Name"])
    tiles_intersect["coords"] = tiles_intersect["geometry"].apply(lambda x: x.centroid.coords[:])
    tiles_intersect["coords"] = [coords[0] for coords in tiles_intersect["coords"]]

    tiles_intersect["sort_key"] = [coord[1] for coord in tiles_intersect["coords"]]
    tiles_intersect.sort_values("sort_key", ascending=False, inplace=True)
    del tiles_intersect["sort_key"]

    text_rectangles = []

    x_step = 0.05

    for idx, row in tiles_intersect.iterrows():
        text = plt.annotate(row["Name"], xy=row["coords"], fontsize=10)

        rect = text.get_window_extent()

        for other_rect in text_rectangles:
            while Bbox.intersection(rect, other_rect):
                x, y = text.get_position()
                x = x + x_step
                text.set_position((x, y))
                rect = text.get_window_extent()
        text_rectangles.append(rect)

    return list(tiles_intersect.Name.unique())


def prepare_roi(gpd_cells, tile_grid, fips_stats, fips_polygons, region, bucket):

    # disolve by cells IDs and summarize the quantative geometries by 'sum'
    gpd_cells = gpd_cells[gpd_cells["region"] == region]
    id_10_agg = gpd_cells[["id_10", "geometry", "cnty_nm"]].dissolve(by="id_10", aggfunc="sum")

    # drop the z coordinates from the tile grid geometry
    def drop_z(geom):
        return wkb.loads(wkb.dumps(geom, output_dimension=2))

    tile_grid.geometry = tile_grid.geometry.transform(drop_z)

    # overlay the tile grids and the cells aggregated geometries in order to
    # obtain the RoI
    tile_grid_temp = tile_grid.copy()
    tile_grid_temp["geometry"] = tile_grid_temp["geometry"].scale(0.7, 0.7)
    id_10_agg = id_10_agg.reset_index()
    zonal_stats_cells = id_10_agg.overlay(tile_grid_temp, how="intersection")

    # select cell IDs which intersect with the RoI
    valid_zonal_stats_cells = pd.merge(
        id_10_agg, zonal_stats_cells["id_10"], on="id_10", how="inner"
    )

    valid_zonal_stats_cells = valid_zonal_stats_cells.drop_duplicates(subset=["id_10"])

    # map RoI to FIPS
    valid_zonal_stats_cells["cnty_nm"] = valid_zonal_stats_cells["cnty_nm"] + " County"
    valid_zonal_stats_cells = valid_zonal_stats_cells.rename(columns={"cnty_nm": "CTYNAME"})

    zonal_fips_stats = pd.merge(valid_zonal_stats_cells, fips_stats, on="CTYNAME", how="inner")

    fips_list = list(zonal_fips_stats.FIPS.unique())

    # select only the FIPS which are fully observed wihtin the tile grid RoI
    filetered_fips_polygons = fips_polygons[fips_polygons.FIPS.isin(list(map(str, fips_list)))]

    tile_grid_union = gpd.GeoDataFrame(crs=tile_grid.crs, geometry=[tile_grid.unary_union])

    fips_overlayed = tile_grid_union.overlay(
        filetered_fips_polygons[["geometry", "FIPS"]], how="intersection"
    )

    fully_observed_fips = fips_overlayed.set_index("FIPS").contains(
        filetered_fips_polygons[["geometry", "FIPS"]].set_index("FIPS"), align=True
    )

    fully_observed_fips = fully_observed_fips.reset_index()
    fully_observed_fips = fully_observed_fips[fully_observed_fips[0]]

    fips_list_union = list(fully_observed_fips.reset_index().FIPS.unique())

    # save shape files (to be used later when zonal statistics are computed)
    zonal_fips_stats.to_file("tmp/zonal_fips_stats.geojson", driver="GeoJSON")

    # create a polygons folder
    try:
        os.makedirs("tmp/polygons")
    except FileExistsError:
        # directory already exists
        pass

    for fips in fips_list:
        zonal_fips_stats[zonal_fips_stats.FIPS == fips].to_file(
            f"tmp/polygons/cell_polygons_{fips}.shp"
        )

    # upload shape files to S3
    s3_file = s3fs.S3FileSystem()
    local_path = "tmp/polygons/"
    s3_path = f"s3://{bucket}/cell-polygons"
    s3_file.put(local_path, s3_path, recursive=True)

    # plot the overlays
    ax = id_10_agg.plot(cmap="gray", figsize=(15, 7))
    tile_grid.plot(ax=ax, cmap="coolwarm")
    zonal_fips_stats.plot(ax=ax, facecolor="none", edgecolor="orange")
    fips_polygons[fips_polygons.FIPS.isin(list(map(str, fips_list_union)))].plot(
        ax=ax, facecolor="none", edgecolor="lime"
    )

    return ",".join(map(str, fips_list_union))
