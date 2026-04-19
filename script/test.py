# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# Load grid_summary (set path to your file)
input_csv = "/mnt/home/2715439w/sharedscratch/fairness/glasgow/results/grid_summary.csv"
df = pd.read_csv(input_csv)

print(f"Loaded {len(df)} grid rows.")

# Cells with SVI and at least one dated capture
df_valid = df[(df["has_svi"] == True) & (df["n_dates"] > 0)].copy()


def decode_year(idx):
    if pd.isna(idx):
        return np.nan
    idx = int(idx)
    year = idx // 12
    remainder = idx % 12
    if remainder == 0:
        return year - 1
    return year


def decode_month(idx):
    if pd.isna(idx):
        return np.nan
    idx = int(idx)
    remainder = idx % 12
    if remainder == 0:
        return 12
    return remainder


df_valid["first_year"] = df_valid["first_month_index"].apply(decode_year)
df_valid["latest_month"] = df_valid["latest_month_index"].apply(decode_month)

print("Decoded first_year and latest_month from month_index.")

sns.set_style("whitegrid")
colors = {"drivable": "#3498db", "non-drivable": "#e74c3c", "no-road": "#95a5a6"}
target_roads = ["drivable", "non-drivable"]
df_plot = df_valid[df_valid["road_type"].isin(target_roads)].copy()

# Part 1: recency by road_type
print("\n--- Part 1: recency ---")

plt.figure(figsize=(8, 6))
sns.boxplot(
    x="road_type",
    y="recency_months",
    data=df_plot,
    hue="road_type",
    order=target_roads,
    palette=colors,
    showfliers=False,
)
plt.title("Fig 1: The Recency Gap (Data Freshness)")
plt.ylabel("Months Since Last Update (Lower is Better)")
plt.savefig("fig1_recency_gap.png")
plt.show()

u_stat, p_val = mannwhitneyu(
    df_plot[df_plot["road_type"] == "drivable"]["recency_months"],
    df_plot[df_plot["road_type"] == "non-drivable"]["recency_months"],
)
print(f"Recency Mann-Whitney p-value: {p_val:.5e}")


# Part 2: stability (max gap) and first capture year
print("\n--- Part 2: stability and history ---")

plt.figure(figsize=(8, 6))
sns.ecdfplot(
    data=df_plot,
    x="max_gap_months",
    hue="road_type",
    hue_order=target_roads,
    palette=colors,
    linewidth=2,
)
plt.axvline(36, color="gray", linestyle="--", alpha=0.5, label="3-year gap")
plt.title("Fig 2a: Cumulative Distribution of Max Gap (Fragmentation)")
plt.xlabel("Max Gap Between Updates (Months)")
plt.ylabel("Proportion of Grids")
plt.legend()
plt.savefig("fig2a_stability_gap.png")
plt.show()

plt.figure(figsize=(8, 6))
sns.kdeplot(
    data=df_plot,
    x="first_year",
    hue="road_type",
    hue_order=target_roads,
    palette=colors,
    fill=True,
    common_norm=False,
    alpha=0.3,
)
plt.title("Fig 2b: Timeline of Digital Colonization")
plt.xlabel("Year of First SVI Capture")
plt.xlim(2008, 2024)
plt.savefig("fig2b_history_lag.png")
plt.show()


# Part 3: seasonality of latest capture month
print("\n--- Part 3: seasonality ---")


def get_season(month):
    if pd.isna(month):
        return None
    if month in [12, 1, 2]:
        return "Winter"
    if month in [3, 4, 5]:
        return "Spring"
    if month in [6, 7, 8]:
        return "Summer"
    return "Autumn"


df_plot["latest_season"] = df_plot["latest_month"].apply(get_season)

season_counts = df_plot.groupby(["road_type", "latest_season"]).size().reset_index(name="counts")
total_counts = df_plot.groupby("road_type").size().reset_index(name="total")
season_prop = season_counts.merge(total_counts, on="road_type")
season_prop["proportion"] = season_prop["counts"] / season_prop["total"]

plt.figure(figsize=(10, 6))
sns.barplot(
    x="latest_season",
    y="proportion",
    hue="road_type",
    data=season_prop,
    order=["Spring", "Summer", "Autumn", "Winter"],
    palette=colors,
)
plt.title("Fig 3: The Seasonality Trap (Latest Image Season)")
plt.ylabel("Proportion of Images")
plt.savefig("fig3_seasonality_trap.png")
plt.show()

print("\nAnalysis complete. Figures saved next to this script.")
# %%
import osmnx as ox
import geopandas as gpd

# Fetch boundary from Nominatim (example helper; convex_hull is optional)
gdf = ox.geocode_to_gdf("Glasgow City, Scotland, UK")

geom = gdf.geometry.iloc[0]

out = gpd.GeoDataFrame({"name": ["Glasgow City"]}, geometry=[geom], crs="EPSG:4326")
out.to_file(
    r"C:\Users\2715439W\Projects\fariness_glasgow\data\glasgow_boundary.geojson",
    driver="GeoJSON",
)
# %%
