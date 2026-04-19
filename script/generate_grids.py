"""Generate grid center points inside the Glasgow boundary (GeoJSON)."""
# %%

import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import os

# ---------------------------------------------------------------------------
# Grid spacing in metres (20 or 50)
# ---------------------------------------------------------------------------
GRID_SIZE_M = 50
if GRID_SIZE_M not in {20, 50}:
    raise ValueError("GRID_SIZE_M must be 20 or 50.")

# ---------------------------------------------------------------------------
# Load Glasgow boundary
# ---------------------------------------------------------------------------
boundary_path = r"C:\Users\2715439W\Projects\fariness_glasgow\data\glasgow_boundary_osmnx.geojson"
glasgow_gdf = gpd.read_file(boundary_path)
glasgow_poly = glasgow_gdf.unary_union

print("Glasgow polygon loaded")
print(f"bounds: {glasgow_gdf.total_bounds}")  # minx, miny, maxx, maxy

# ---------------------------------------------------------------------------
# Step size in degrees for GRID_SIZE_M spacing
# ---------------------------------------------------------------------------
minx, miny, maxx, maxy = glasgow_gdf.total_bounds
ref_lat = (miny + maxy) / 2

north_south_m = geodesic((maxy, ref_lat), (miny, ref_lat)).meters
east_west_m = geodesic((ref_lat, minx), (ref_lat, maxx)).meters

lat_step = (maxy - miny) / (north_south_m / GRID_SIZE_M)
lon_step = (maxx - minx) / (east_west_m / GRID_SIZE_M)

print(
    f"step size for {GRID_SIZE_M}m grid: "
    f"{lat_step:.7f} deg lat, {lon_step:.7f} deg lon"
)

# ---------------------------------------------------------------------------
# Build a lat/lon lattice and keep points inside the boundary
# ---------------------------------------------------------------------------
lat_vals = np.arange(miny, maxy, lat_step)
lon_vals = np.arange(minx, maxx, lon_step)

grid_points = []
for i, lat in enumerate(lat_vals):
    for j, lon in enumerate(lon_vals):
        p = Point(lon, lat)
        if glasgow_poly.contains(p):
            grid_points.append((lat, lon))
    if i % 100 == 0:
        print(f"row {i}/{len(lat_vals)} done, total grids: {len(grid_points)}")

print(f"\nGlasgow: {len(grid_points)} grid cells at {GRID_SIZE_M}m spacing")

# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------
out_dir = r"C:\Users\2715439W\Projects\fariness_glasgow\result"
os.makedirs(out_dir, exist_ok=True)
out_csv = os.path.join(out_dir, f"glasgow_grid_{GRID_SIZE_M}m.csv")

df = pd.DataFrame(grid_points, columns=["query_lat", "query_lon"])
df["grid_id"] = range(len(df))

df.to_csv(out_csv, index=False)
print(f"saved to {out_csv}")

# %%
