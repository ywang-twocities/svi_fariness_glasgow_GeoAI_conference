"""
Clean Street View metadata CSV.

- Drop rows with missing year or month.
- For each panoid, keep one row: minimum distance_m (nearest grid point).
- Write cleaned CSV.
"""
# %%
import pandas as pd

GRID_SIZE_M = 50
meta_data = pd.read_csv(
    rf"C:\Users\2715439W\Projects\fariness_glasgow\result\glasgow_streetview_metadata_grid_{GRID_SIZE_M}m.csv"
)

# %%
meta_data[:10]

# %%
# Example: one panoid with multiple grid hits
meta_data[meta_data["panoid"] == "gEwac6dZ153bgC4Z_6YgtA"]

# %%
print("total rows:\n")
print(meta_data.shape[0])
print("rows with non-null year:\n")
print(meta_data[meta_data["year"].notna()].shape[0])
print("\nunique panoids:\n")
print(meta_data["panoid"].nunique())
print("\n")
print(
    f"panoids appearing in more than one grid (approx.): "
    f"{meta_data[meta_data['year'].notna()].shape[0] - meta_data['panoid'].nunique()}"
)
# %%

# Drop rows where year or month is missing
filtered = meta_data.dropna(subset=["year", "month"])

# One row per panoid: nearest grid (min distance_m)
filtered_unique = filtered.loc[filtered.groupby("panoid")["distance_m"].idxmin()]

# Same example after filtering
filtered_unique[filtered_unique["panoid"] == "gEwac6dZ153bgC4Z_6YgtA"]

print("total rows after filtering:\n")
print(filtered_unique.shape[0])
print("non-null year after filtering:\n")
print(filtered_unique[filtered_unique["year"].notna()].shape[0])
print("\nunique panoids after filtering:\n")
print(filtered_unique["panoid"].nunique())
print("\n")
print(
    f"panoids appearing in more than one grid (approx.): "
    f"{filtered_unique[filtered_unique['year'].notna()].shape[0] - filtered_unique['panoid'].nunique()}"
)
# %%
filtered_unique.to_csv(
    rf"C:\Users\2715439W\Projects\fariness_glasgow\result\glasgow_streetview_metadata_grid_{GRID_SIZE_M}m_cleaned.csv",
    index=False,
)
# %%
