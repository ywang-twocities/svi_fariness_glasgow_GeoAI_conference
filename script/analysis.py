# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, kruskal

GRID_SIZE_M = 50

# Plot style 
sns.set_theme(style="whitegrid", context="talk", font_scale=1.1)

grid_summary = pd.read_csv(
    rf"C:\Users\2715439W\Projects\fariness_glasgow\result\grid_summary_{GRID_SIZE_M}m.csv"
)

if grid_summary["has_svi"].dtype == object:
    grid_summary["has_svi"] = (
        grid_summary["has_svi"].astype(str).str.upper().map({"TRUE": True, "FALSE": False})
    )

# Drop rare highway types (small n) for cleaner plots
highway_counts = grid_summary["grid_highway"].value_counts()
valid_highways = highway_counts[highway_counts >= 30].index
df_clean = grid_summary[grid_summary["grid_highway"].isin(valid_highways)].copy()

print("Data loaded.\n")
# %%
# Section 1: SVI coverage by road_type
print("Generating plot 1: coverage rate...")

coverage_stats = (
    df_clean.groupby("road_type")
    .agg(total=("grid_id", "count"), covered=("has_svi", "sum"))
    .reset_index()
)
coverage_stats["coverage_rate"] = (coverage_stats["covered"] / coverage_stats["total"]) * 100

plt.figure(figsize=(8, 6))
ax1 = sns.barplot(
    data=coverage_stats,
    x="road_type",
    y="coverage_rate",
    palette="Blues_d",
    order=["drivable", "non-drivable", "no-road"],
)

for p in ax1.patches:
    ax1.annotate(
        f"{p.get_height():.1f}%",
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 10),
        textcoords="offset points",
        fontsize=14,
        fontweight="bold",
    )

plt.title("SVI Coverage Rate by Road Type", pad=20)
plt.ylabel("Coverage Rate (%)")
plt.xlabel("")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(f"poster_fig1_coverage_{GRID_SIZE_M}m.png", dpi=300)
plt.show()
# %%
# Section 2: Recency by highway type (drivable cells with SVI)
print("Generating plot 2: hierarchy boxplot...")

df_hierarchy = df_clean[
    (df_clean["has_svi"] == True) & (df_clean["road_type"] == "drivable")
].copy()

order_list = (
    df_hierarchy.groupby("grid_highway")["recency_months"]
    .median()
    .sort_values(ascending=True)
    .index
)

plt.figure(figsize=(16, 8))
sns.boxplot(
    data=df_hierarchy,
    x="grid_highway",
    y="recency_months",
    order=order_list,
    palette="viridis",
    showfliers=False,
    linewidth=1.5,
)

overall_median = df_hierarchy["recency_months"].median()
plt.axhline(
    overall_median,
    color="red",
    linestyle="--",
    alpha=0.6,
    label=f"Overall median ({overall_median:.1f} months)",
)

plt.xticks(rotation=45, ha="right", fontsize=14)
plt.title("The Infrastructure Hierarchy: Faster Roads = Fresher Data", fontsize=18, pad=20)
plt.ylabel("Recency (Months Since Last Update)", fontsize=16)
plt.xlabel("OSM Highway Type", fontsize=16)
plt.legend()
plt.tight_layout()
plt.savefig(f"poster_fig2_hierarchy_{GRID_SIZE_M}m.png", dpi=300)
plt.show()
# %%
# Section 3: Lifecycle stacked bar (drivable vs non-drivable)
print("Generating plot 3: lifecycle stacked bar...")

def time_class(row):
    if row["n_dates"] == 0:
        return "Invisible (No Data)"
    if row["max_gap_months"] >= 72:
        return "Forgotten (>6 yrs Gap)"
    if row["n_dates"] == 1:
        return "One-off (Only once)"
    return "Well-Maintained"


df_clean["lifecycle"] = df_clean.apply(time_class, axis=1)

lifecycle_counts = df_clean.groupby(["road_type", "lifecycle"]).size().unstack(fill_value=0)

lifecycle_pct = lifecycle_counts.div(lifecycle_counts.sum(axis=1), axis=0) * 100

lifecycle_pct = lifecycle_pct.reindex(["drivable", "non-drivable"])

lifecycle_order = [
    "Well-Maintained",
    "One-off (Only once)",
    "Forgotten (>6 yrs Gap)",
    "Invisible (No Data)",
]
lifecycle_pct = lifecycle_pct.reindex(columns=lifecycle_order).fillna(0)

lifecycle_colors = {
    "Well-Maintained": "#2E7D32",
    "One-off (Only once)": "#BDBDBD",
    "Forgotten (>6 yrs Gap)": "#E69F00",
    "Invisible (No Data)": "#4D4D4D",
}
color_list = [lifecycle_colors[c] for c in lifecycle_order]

fig, ax = plt.subplots(figsize=(8.5, 5.5))

lifecycle_pct.plot(
    kind="bar",
    stacked=True,
    ax=ax,
    color=color_list,
    width=0.65,
    edgecolor="white",
    linewidth=0.8,
)

ax.set_title("SVI Lifecycle Status by Infrastructure Type", fontsize=16, pad=12)
ax.set_ylabel("Percentage of grid cells (%)", fontsize=13)
ax.set_xlabel("")
ax.set_ylim(0, 100)

ax.set_xticklabels(["Drivable", "Non-drivable"], rotation=0, fontsize=12)
ax.tick_params(axis="y", labelsize=12)

ax.legend(
    title="Lifecycle status",
    title_fontsize=12,
    fontsize=11,
    frameon=False,
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
)

# Label segments >= 8% only
for container in ax.containers:
    labels = []
    for v in container.datavalues:
        labels.append(f"{v:.0f}%" if v >= 8 else "")
    ax.bar_label(
        container,
        labels=labels,
        label_type="center",
        fontsize=11,
        color="white",
        fontweight="bold",
    )

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"poster_fig3_lifecycle_nature_{GRID_SIZE_M}m.png", dpi=300, bbox_inches="tight")
plt.show()
# %%
# Section 4: Hypothesis tests (print p-values)
print("\n===== Statistical tests =====")

drivable_recency = df_clean[df_clean["road_type"] == "drivable"]["recency_months"].dropna()
non_drivable_recency = df_clean[df_clean["road_type"] == "non-drivable"]["recency_months"].dropna()

stat, p_val = mannwhitneyu(drivable_recency, non_drivable_recency, alternative="two-sided")
print("[1] Freshness: drivable vs non-drivable (Mann-Whitney U)")
print(f"    p-value = {p_val:.2e} (significant if < 0.05)")

groups = [g["recency_months"].dropna() for _, g in df_hierarchy.groupby("grid_highway")]
stat_k, p_val_k = kruskal(*groups)
print("[2] Recency across highway types (Kruskal-Wallis)")
print(f"    p-value = {p_val_k:.2e} (significant if < 0.05)")

print("\nDone. PNG figures written to the working directory.")
# %%
