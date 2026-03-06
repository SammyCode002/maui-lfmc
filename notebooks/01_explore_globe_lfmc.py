# %% [markdown]
# # Exploring Globe-LFMC 2.0: Does Hawaii Have Data?
#
# **Purpose:** Before we can fine-tune a model for Maui, we need to know
# if there are ANY ground truth LFMC samples in Hawaii. This notebook
# downloads and explores the Globe-LFMC 2.0 dataset to answer that question.
#
# **What we're looking for:**
# - How many samples exist in Hawaii? Maui County specifically?
# - What vegetation types were sampled?
# - What time periods are covered?
# - How does Hawaii compare to the CONUS data used by Johnson et al.?
#
# **If Hawaii has zero samples:** We use the CONUS-trained model as-is
# (transfer learning) and evaluate how well it generalizes to Maui.
#
# **If Hawaii has some samples:** We can fine-tune specifically for
# Hawaiian vegetation types. Even a few dozen samples helps.

# %%
# SETUP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path

# Create data directories
Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

print("Setup complete.")

# %% [markdown]
# ## Step 1: Download Globe-LFMC 2.0
#
# The dataset is hosted on figshare (~50MB Excel file).
# Contains 280,000+ LFMC measurements from 2,000+ sites across 15 countries.

# %%
# DOWNLOAD THE DATASET
# You can also download manually from:
# https://springernature.figshare.com/articles/dataset/Globe-LFMC-2_0/25413790

import requests

GLOBE_LFMC_URL = "https://springernature.figshare.com/ndownloader/files/45049786"
RAW_PATH = Path("data/raw/Globe-LFMC-2.0.xlsx")

if RAW_PATH.exists():
    print(f"Already downloaded: {RAW_PATH} ({RAW_PATH.stat().st_size / 1e6:.1f} MB)")
else:
    print("Downloading Globe-LFMC 2.0 (this takes a minute)...")
    start = time.time()
    resp = requests.get(GLOBE_LFMC_URL, stream=True)
    resp.raise_for_status()
    with open(RAW_PATH, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    elapsed = time.time() - start
    print(f"Downloaded in {elapsed:.0f}s: {RAW_PATH} ({RAW_PATH.stat().st_size / 1e6:.1f} MB)")

# %% [markdown]
# ## Step 2: Load and Inspect the Dataset

# %%
# LOAD THE DATASET
print("Loading Globe-LFMC 2.0 (this takes ~30 seconds for the Excel file)...")
start = time.time()

try:
    df = pd.read_excel(RAW_PATH, sheet_name="LFMC Data")
except Exception:
    print("Sheet 'LFMC Data' not found, trying first sheet...")
    df = pd.read_excel(RAW_PATH, sheet_name=0)

elapsed = time.time() - start
print(f"Loaded in {elapsed:.0f}s")
print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"\nColumn names:\n{list(df.columns)}")

# %%
# INSPECT FIRST FEW ROWS
df.head()

# %%
# FIND THE KEY COLUMNS
# Column names vary between Globe-LFMC versions, so let's be flexible
print("Looking for key columns...\n")

for col in df.columns:
    cl = col.lower()
    if any(word in cl for word in ["lat", "lon", "lfmc", "date", "species", "country", "state"]):
        sample_vals = df[col].dropna().head(3).tolist()
        print(f"  {col}: {sample_vals}")

# %% [markdown]
# ## Step 3: Filter to Hawaii
#
# Globe-LFMC uses lat/lon coordinates. Hawaii's bounding box:
# - Latitude: 18.5 to 22.5
# - Longitude: -161.0 to -154.0
#
# Maui County specifically:
# - Latitude: 20.5 to 21.1
# - Longitude: -156.7 to -155.9

# %%
# STANDARDIZE COLUMN NAMES
# Map whatever column names exist to consistent names
col_map = {}
for col in df.columns:
    cl = col.lower().strip()
    if "latitude" in cl or cl == "lat":
        col_map[col] = "latitude"
    elif "longitude" in cl or cl == "lon":
        col_map[col] = "longitude"
    elif "lfmc" in cl and ("value" in cl or "%" in cl or cl == "lfmc"):
        col_map[col] = "lfmc_value"
    elif "sampling" in cl and "date" in cl:
        col_map[col] = "sampling_date"
    elif "species" in cl:
        col_map[col] = "species"
    elif "country" in cl:
        col_map[col] = "country"
    elif "state" in cl and "united" not in cl:
        col_map[col] = "state"

df = df.rename(columns=col_map)
print(f"Mapped columns: {col_map}")

# %%
# THE BIG QUESTION: DOES HAWAII HAVE DATA?

# Hawaii state bounds
hawaii_mask = (
    (df["latitude"] >= 18.5) & (df["latitude"] <= 22.5) &
    (df["longitude"] >= -161.0) & (df["longitude"] <= -154.0)
)
hawaii_df = df[hawaii_mask]

# Maui County bounds
maui_mask = (
    (df["latitude"] >= 20.5) & (df["latitude"] <= 21.1) &
    (df["longitude"] >= -156.7) & (df["longitude"] <= -155.9)
)
maui_df = df[maui_mask]

print("=" * 50)
print("HAWAII DATA CHECK")
print("=" * 50)
print(f"Total Globe-LFMC samples:  {len(df):,}")
print(f"Samples in Hawaii state:   {len(hawaii_df):,}")
print(f"Samples in Maui County:    {len(maui_df):,}")
print("=" * 50)

if len(hawaii_df) == 0:
    print("\nNO HAWAII DATA FOUND.")
    print("This means we'll use transfer learning from CONUS-trained model.")
    print("This is actually the expected result and still a valid project!")
else:
    print(f"\nFOUND {len(hawaii_df)} HAWAII SAMPLES!")
    print("\nHawaii sample details:")
    display_cols = [c for c in ["latitude", "longitude", "lfmc_value", "sampling_date", "species"] if c in hawaii_df.columns]
    print(hawaii_df[display_cols].to_string())

# %%
# CHECK NEARBY: US TERRITORIES AND PACIFIC ISLANDS
# Maybe there are Pacific samples that could help?

if "country" in df.columns:
    us_df = df[df["country"].str.lower().str.contains("united states|usa|us", na=False)]
    print(f"US samples total: {len(us_df):,}")

    if "state" in df.columns:
        state_counts = us_df["state"].value_counts()
        print(f"\nTop 10 US states by sample count:")
        print(state_counts.head(10))
        if "hawaii" in state_counts.index.str.lower().values:
            print(f"\nHawaii count by state field: {state_counts[state_counts.index.str.lower() == 'hawaii'].sum()}")

# %% [markdown]
# ## Step 4: Analyze CONUS Data (for baseline)
#
# Even if Hawaii has no data, we need to understand the CONUS training data
# because that's what the model learns from. Johnson et al. used 41,214
# CONUS samples from 2017-2023.

# %%
# FILTER TO CONUS 2017-2023 (matching Johnson et al.)
conus_mask = (
    (df["latitude"] >= 24.0) & (df["latitude"] <= 49.5) &
    (df["longitude"] >= -125.0) & (df["longitude"] <= -66.0)
)
conus_df = df[conus_mask].copy()

# Add date features
if "sampling_date" in conus_df.columns:
    conus_df["sampling_date"] = pd.to_datetime(conus_df["sampling_date"], errors="coerce")
    conus_df["year"] = conus_df["sampling_date"].dt.year
    conus_df["month"] = conus_df["sampling_date"].dt.month

    # Filter to 2017-2023
    conus_df = conus_df[(conus_df["year"] >= 2017) & (conus_df["year"] <= 2023)]

print(f"CONUS samples (2017-2023): {len(conus_df):,}")
print(f"(Johnson et al. reported 41,214 after aggregation)")

if "lfmc_value" in conus_df.columns:
    print(f"\nLFMC statistics:")
    print(f"  Mean:   {conus_df['lfmc_value'].mean():.1f}%")
    print(f"  Median: {conus_df['lfmc_value'].median():.1f}%")
    print(f"  Std:    {conus_df['lfmc_value'].std():.1f}%")
    print(f"  Min:    {conus_df['lfmc_value'].min():.1f}%")
    print(f"  Max:    {conus_df['lfmc_value'].max():.1f}%")

# %%
# VISUALIZE: LFMC DISTRIBUTION
if "lfmc_value" in conus_df.columns:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(conus_df["lfmc_value"].dropna(), bins=80, color="#4CAF50", edgecolor="white", alpha=0.8)
    ax.set_xlabel("LFMC Value (%)")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Distribution of LFMC Values in CONUS (2017-2023)")
    ax.axvline(x=100, color="red", linestyle="--", alpha=0.7, label="100% (equal water/dry weight)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("data/processed/lfmc_distribution_conus.png", dpi=150)
    plt.show()
    print("Saved: data/processed/lfmc_distribution_conus.png")

# %%
# VISUALIZE: SEASONAL BREAKDOWN (reproducing Figure 1 from the paper)
if "month" in conus_df.columns:
    season_map = {12: "Winter", 1: "Winter", 2: "Winter",
                  3: "Spring", 4: "Spring", 5: "Spring",
                  6: "Summer", 7: "Summer", 8: "Summer",
                  9: "Autumn", 10: "Autumn", 11: "Autumn"}
    conus_df["season"] = conus_df["month"].map(season_map)

    season_order = ["Winter", "Spring", "Summer", "Autumn"]
    season_counts = conus_df["season"].value_counts().reindex(season_order)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(season_counts.index, season_counts.values, color="#E57373", edgecolor="white")
    for bar, count in zip(bars, season_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f"{count:,}", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("# of Samples")
    ax.set_title("Meteorological Season Breakdown of Globe-LFMC 2.0 (CONUS, 2017-2023)")
    plt.tight_layout()
    plt.savefig("data/processed/season_breakdown_conus.png", dpi=150)
    plt.show()
    print("Saved: data/processed/season_breakdown_conus.png")

# %%
# VISUALIZE: SAMPLE LOCATIONS MAP
fig, ax = plt.subplots(figsize=(14, 8))

# Plot all CONUS samples
scatter = ax.scatter(
    conus_df["longitude"], conus_df["latitude"],
    c=conus_df["lfmc_value"] if "lfmc_value" in conus_df.columns else "blue",
    cmap="RdYlGn", s=3, alpha=0.5, vmin=30, vmax=180,
)
plt.colorbar(scatter, ax=ax, label="LFMC (%)", shrink=0.6)

# Highlight Hawaii location (even if no data there)
hawaii_box = plt.Rectangle((-161, 18.5), 7, 4, fill=False, edgecolor="red", linewidth=2, linestyle="--")
ax.add_patch(hawaii_box)
ax.annotate("Hawaii\n(your project!)", xy=(-157.5, 20), fontsize=10,
            color="red", fontweight="bold", ha="center")

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Globe-LFMC 2.0 Sample Locations (CONUS, 2017-2023)")
ax.set_xlim(-130, -60)
ax.set_ylim(22, 52)
plt.tight_layout()
plt.savefig("data/processed/sample_locations_conus.png", dpi=150)
plt.show()
print("Saved: data/processed/sample_locations_conus.png")

# %% [markdown]
# ## Step 5: Save Processed Data
#
# Save the filtered CONUS data as CSV for training pipeline.
# If Hawaii data exists, save that too.

# %%
# SAVE PROCESSED DATA

# CONUS West (where most samples are, matches Johnson et al.)
conus_west_mask = conus_df["longitude"] <= -100.0
conus_west = conus_df[conus_west_mask]
conus_west.to_csv("data/processed/globe_lfmc_conus_west.csv", index=False)
print(f"Saved CONUS-West: {len(conus_west):,} samples")

# Full CONUS
conus_df.to_csv("data/processed/globe_lfmc_conus.csv", index=False)
print(f"Saved full CONUS: {len(conus_df):,} samples")

# Hawaii (if any)
if len(hawaii_df) > 0:
    hawaii_df.to_csv("data/processed/globe_lfmc_hawaii.csv", index=False)
    print(f"Saved Hawaii: {len(hawaii_df):,} samples")

# Maui (if any)
if len(maui_df) > 0:
    maui_df.to_csv("data/processed/globe_lfmc_maui.csv", index=False)
    print(f"Saved Maui: {len(maui_df):,} samples")

# Summary stats
summary = {
    "total_globe_lfmc": len(df),
    "conus_2017_2023": len(conus_df),
    "conus_west": len(conus_west),
    "hawaii": len(hawaii_df),
    "maui_county": len(maui_df),
}
with open("data/processed/data_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSummary: {json.dumps(summary, indent=2)}")

# %% [markdown]
# ## Next Steps
#
# Based on the Hawaii data check above:
#
# **If Hawaii has 0 samples (most likely):**
# 1. Use CONUS-west data to reproduce Johnson et al. results (Phase 3)
# 2. Apply the trained model directly to Maui satellite imagery
# 3. The project becomes: "How well does a CONUS-trained LFMC model
#    generalize to Hawaiian vegetation?"
#
# **If Hawaii has some samples:**
# 1. Still reproduce CONUS results first (Phase 3)
# 2. Then fine-tune specifically on Hawaii data
# 3. Compare: CONUS-only model vs Hawaii-tuned model
#
# **Either way, next step is:**
# - Download Galileo model weights
# - Set up the training environment
# - Run `src/data/sentinel_download.py` to fetch satellite imagery
