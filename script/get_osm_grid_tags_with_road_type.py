"""Extract OSM tags per grid cell over Glasgow using pyrosm.

Cell size is set by GRID_SIZE_M (must match generate_grids.py).

Steps:
  1) Read Glasgow boundary GeoJSON.
  2) Load OSM from PBF within that boundary.
  3) Build square cells from grid center CSV (glasgow_grid_{GRID_SIZE_M}m.csv).
  4) Spatially join OSM layers to cells.
  5) Melt tags to long form (grid_id, tag_key, tag_value).
  6) Summarise tags per cell and write CSV.
"""
# %% Imports
import os
import warnings

import geopandas as gpd
import pandas as pd
from pyrosm import OSM
from tqdm import tqdm

warnings.filterwarnings("ignore")
tqdm.pandas()

GRID_SIZE_M = 20

# %% Paths
# Boundary GeoJSON (e.g. from osmnx or manual export)
BOUNDARY_PATH = rf"C:\Users\2715439W\Projects\fariness_glasgow\data\glasgow_boundary_osmnx.geojson"
# Scotland extract PBF
PBF_PATH = rf"C:\Users\2715439W\Projects\fariness_glasgow\data\scotland-260417.osm.pbf"
# Grid centers from generate_grids.py
GRID_CSV = rf"C:\Users\2715439W\Projects\fariness_glasgow\result\glasgow_grid_{GRID_SIZE_M}m.csv"


# %% Helpers
def to_27700(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reproject to EPSG:27700 for metric geometry."""
    if gdf.crs is None:
        gdf = gdf.set_crs(4326, allow_override=True)
    return gdf.to_crs(27700)


def back_to_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reproject to WGS84 (EPSG:4326)."""
    return gdf.to_crs(4326)


def melt_tags(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """
    Turn OSM attribute columns into long format: id_col, tag_key, tag_value.

    Expects a spatial-join result with grid_id and no geometry needed here.
    """
    drop_cols = {
        id_col,
        "id",
        "osm_id",
        "timestamp",
        "version",
        "changeset",
        "uid",
        "layer",
        "index_right",
        "source",
        "z_order",
    }
    tag_cols = [
        c
        for c in df.columns
        if c not in drop_cols and c != "geometry" and not df[c].isna().all()
    ]
    if not tag_cols:
        return pd.DataFrame(columns=[id_col, "tag_key", "tag_value"])
    long_df = df[[id_col] + tag_cols].melt(
        id_vars=id_col, var_name="tag_key", value_name="tag_value"
    ).dropna(subset=["tag_value"])
    long_df["tag_value"] = long_df["tag_value"].astype(str)
    return long_df


# %% Load boundary
glasgow = gpd.read_file(BOUNDARY_PATH)
glasgow = glasgow.to_crs(4326)
bounding_polygon = glasgow.geometry.iloc[0]
print("Glasgow boundary loaded.")

# %% Load OSM
osm = OSM(PBF_PATH, bounding_box=bounding_polygon)

roads_all = osm.get_network(network_type="all")
roads_drivable = osm.get_network(network_type="driving")

buildings = osm.get_buildings()

landuse = osm.get_landuse()
natural = osm.get_natural()

pois = osm.get_pois()

amenities = pois[pois["amenity"].notnull()]
shops = pois[pois["shop"].notnull()]
tourism = pois[pois["tourism"].notnull()]

layers = {
    "roads": roads_all,
    "buildings": buildings,
    "landuse": landuse,
    "amenities": amenities,
    "natural": natural,
    "shops": shops,
    "tourism": tourism,
}

for k, v in layers.items():
    print(f"{k:9s} ->", "None" if v is None else f"{len(v)} features")

# %% Build square cells from grid CSV
df_grid = pd.read_csv(GRID_CSV)
gdf_points = gpd.GeoDataFrame(
    df_grid,
    geometry=gpd.points_from_xy(df_grid["query_lon"], df_grid["query_lat"]),
    crs=4326,
)

gdf_points_27700 = to_27700(gdf_points)
half = GRID_SIZE_M / 2
# cap_style=3: square buffers
gdf_points_27700["geometry"] = gdf_points_27700.geometry.buffer(half, cap_style=3)

grid = back_to_wgs84(gdf_points_27700)
if "grid_id" in df_grid.columns:
    grid["grid_id"] = df_grid["grid_id"].values
else:
    grid["grid_id"] = range(len(grid))
print(
    f"Built {len(grid)} {GRID_SIZE_M}m x {GRID_SIZE_M}m cells from {GRID_CSV}."
)

# %% Spatial join OSM layers to grid
grid_27700 = to_27700(grid)
joined_list = []

for name, gdf in layers.items():
    if gdf is None or len(gdf) == 0:
        continue
    gdf = gdf.to_crs(27700)
    keep_cols = [c for c in gdf.columns if c != "geometry"]
    j = gpd.sjoin(
        gdf[keep_cols + ["geometry"]],
        grid_27700[["grid_id", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    j = j.drop(columns=["index_right"])
    j["layer"] = name
    joined_list.append(j)

if not joined_list:
    raise RuntimeError(f"No intersection between OSM layers and grids from {GRID_CSV}!")

joined_all_27700 = pd.concat(joined_list, ignore_index=True)
joined_all = back_to_wgs84(joined_all_27700)
print("Spatial join done:", len(joined_all), "records")


# %% Road type: no-road, non-drivable, drivable
roads_all_27700 = to_27700(roads_all)

join_all = gpd.sjoin(
    grid_27700[["grid_id", "geometry"]],
    roads_all_27700[["geometry"]],
    how="left",
    predicate="intersects",
)

roads_drivable_27700 = to_27700(roads_drivable)
join_drive = gpd.sjoin(
    grid_27700[["grid_id", "geometry"]],
    roads_drivable_27700[["geometry"]],
    how="left",
    predicate="intersects",
)

grids_with_any_road = join_all.loc[join_all["index_right"].notnull(), "grid_id"].unique()
grids_with_drive_road = join_drive.loc[join_drive["index_right"].notnull(), "grid_id"].unique()

grid["road_type"] = "no-road"
grid.loc[grid["grid_id"].isin(grids_with_any_road), "road_type"] = "non-drivable"
grid.loc[grid["grid_id"].isin(grids_with_drive_road), "road_type"] = "drivable"

print(
    "Road type counts: "
    f"no-road={sum(grid['road_type']=='no-road')}, "
    f"non-drivable={sum(grid['road_type']=='non-drivable')}, "
    f"drivable={sum(grid['road_type']=='drivable')}"
)


# %% Tag summary and main highway per cell
joined_no_geom = joined_all.drop(columns=["geometry"], errors="ignore")
tags_long = melt_tags(joined_no_geom, id_col="grid_id")

SEMANTIC_KEYS = ["building", "landuse", "amenity", "natural", "shop", "tourism", "highway"]
tags_long = tags_long[tags_long["tag_key"].isin(SEMANTIC_KEYS)]
print("Tag rows (long format):", len(tags_long))

summary = (
    tags_long.groupby("grid_id")
    .agg(
        tag_key_list=("tag_key", list),
        tag_value_list=("tag_value", list),
        n_tags=("tag_key", "size"),
        unique_keys=("tag_key", "nunique"),
    )
    .reset_index()
)

highway_df = tags_long[tags_long["tag_key"] == "highway"]


def pick_main_highway(values):
    priority = [
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
        "residential",
        "service",
        "unclassified",
        "pedestrian",
        "track",
        "path",
        "footway",
    ]
    for p in priority:
        if p in values:
            return p
    return values[0] if values else None


highway_summary = (
    highway_df.groupby("grid_id")["tag_value"]
    .apply(lambda x: pick_main_highway(list(x)))
    .reset_index()
    .rename(columns={"tag_value": "grid_highway"})
)

summary = summary.merge(highway_summary, on="grid_id", how="left")

grid = grid.merge(summary, on="grid_id", how="left")
for c in ["n_tags", "unique_keys"]:
    grid[c] = grid[c].fillna(0).astype(int)

print("Cells with at least one tag:", grid["n_tags"].gt(0).sum())

# %% Write CSV
cols_to_keep = [
    "grid_id",
    "query_lat",
    "query_lon",
    "grid_highway",
    "road_type",
    "n_tags",
    "unique_keys",
    "tag_key_list",
    "tag_value_list",
]
grid_simplified = grid[cols_to_keep].copy()

output_path = os.path.join(
    r"C:\Users\2715439W\Projects\fariness_glasgow\result",
    f"grid_with_osm_tags_roads_{GRID_SIZE_M}m.csv",
)

grid_simplified.to_csv(output_path, index=False)
print(f"Wrote: {output_path}")
# %%
