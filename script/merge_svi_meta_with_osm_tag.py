# %%
import pandas as pd

GRID_SIZE_M = 20

# %%
osm_tags = pd.read_csv(
    rf"C:\Users\2715439W\Projects\fariness_glasgow\result\grid_with_osm_tags_roads_{GRID_SIZE_M}m.csv"
)
svi_meta = pd.read_csv(
    rf"C:\Users\2715439W\Projects\fariness_glasgow\result\glasgow_streetview_metadata_grid_{GRID_SIZE_M}m_cleaned.csv"
)

# %%
print(f"OSM tags shape: {osm_tags.shape}")
print(f"SVI metadata shape: {svi_meta.shape}\n")

print(osm_tags.columns)
print("\n")
print(svi_meta.columns)
print("\n")
print(osm_tags.head(5))
print("\n")
print(svi_meta.head(5))
print("\n")
# %%
svi_meta["date"] = pd.to_datetime(
    svi_meta["year"].astype(int).astype(str) + "-" + svi_meta["month"].astype(int).astype(str)
)
# Merge SVI rows onto OSM grid rows by grid_id (one row per panoid after cleaning)
svi_meta.drop(columns=["query_lat", "query_lon"], inplace=True)
merged = osm_tags.merge(svi_meta, on="grid_id", how="left")

# Put grid_id first
cols = ["grid_id"] + [c for c in merged.columns if c != "grid_id"]
merged = merged[cols]

print(merged.head())
# %%
merged.to_csv(
    rf"C:\Users\2715439W\Projects\fariness_glasgow\result\merged_svi_osm_{GRID_SIZE_M}m.csv",
    index=False,
)

# %%
