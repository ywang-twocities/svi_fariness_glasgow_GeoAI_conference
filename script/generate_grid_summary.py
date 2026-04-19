# %%
import pandas as pd
import numpy as np

# %%
GRID_SIZE_M = 50
merged = pd.read_csv(
    rf"C:\Users\2715439W\Projects\fariness_glasgow\result\merged_svi_osm_{GRID_SIZE_M}m.csv"
)

# %%
# Parse month-level dates; invalid values become NaT (rows are kept)
merged["date"] = pd.to_datetime(merged["date"], errors="coerce")

# %%
# Month index: year * 12 + month (float). Differences are in whole months.
merged["month_index"] = (
    merged["date"].dt.year.astype("float64") * 12
    + merged["date"].dt.month.astype("float64")
)

# %%
# Latest month index in the dataset (for recency)
valid_month_idx = merged["month_index"].dropna()
global_latest_month_index = valid_month_idx.max()

# %%
def gap_stats(date_series: pd.Series):
    """
    For one grid: max gap in months between consecutive capture months,
    and count of non-null dates.

    Returns:
        (max_gap_months, n_valid_dates)
    """
    dates = date_series.dropna().sort_values()
    n = len(dates)

    if n < 2:
        return np.nan, n

    month_idx = dates.dt.year.to_numpy() * 12 + dates.dt.month.to_numpy()
    diffs = np.diff(month_idx)
    return float(diffs.max()), n


# %%
gap_df = (
    merged.groupby("grid_id")["date"]
    .apply(gap_stats)
    .apply(pd.Series)
    .rename(columns={0: "max_gap_months", 1: "n_valid_dates"})
    .reset_index()
)

# %%
def collect_month_indices(series):
    """All non-null month_index values as a list."""
    return [x for x in series.dropna().tolist()]


grid_summary = (
    merged.groupby("grid_id")
    .agg(
        query_lat=("query_lat", "first"),
        query_lon=("query_lon", "first"),
        grid_highway=("grid_highway", "first"),
        road_type=("road_type", "first"),
        n_tags=("n_tags", "first"),
        unique_keys=("unique_keys", "first"),
        tag_key_list=("tag_key_list", "first"),
        tag_value_list=("tag_value_list", "first"),
        n_panos=("panoid", lambda x: x.notna().sum()),
        n_dates=("date", "nunique"),
        first_month_index=("month_index", "min"),
        latest_month_index=("month_index", "max"),
        date_all=("month_index", collect_month_indices),
    )
)

# %%
# Months from first to last capture; months since last capture vs global latest
grid_summary["span_months"] = (
    grid_summary["latest_month_index"] - grid_summary["first_month_index"]
)
grid_summary["recency_months"] = (
    global_latest_month_index - grid_summary["latest_month_index"]
)

# %%
grid_summary["has_svi"] = grid_summary["n_panos"] > 0

# %%
grid_summary = grid_summary.merge(
    gap_df,
    left_on="grid_id",
    right_on="grid_id",
    how="left",
)

# %%
grid_summary = grid_summary.reset_index()
grid_summary.to_csv(
    rf"C:\Users\2715439W\Projects\fariness_glasgow\result\grid_summary_{GRID_SIZE_M}m.csv",
    index=False,
)
# One row per grid_id; grids without Street View are kept.

# %%
