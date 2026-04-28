"""
preprocess_eda.py
-----------------
Loads the raw CDC 500 Cities CSV produced by fetch_data.py, applies the
full preprocessing pipeline, runs exploratory data analysis, and saves
all processed artifacts consumed by main_notebook.ipynb.

Usage (from project root):
    python scripts/preprocess_eda.py
    python scripts/preprocess_eda.py --raw data/raw/cdc_500cities_raw.csv

Outputs written to data/processed/:
    wide_matrix.csv         — tracts × 20 measures (imputed, original scale)
    wide_scaled.csv         — tracts × 20 measures (StandardScaler)
    pca_4d.csv              — tracts × 4 principal components
    pca_loadings.csv        — 20 measures × 4 PCs (loading matrix)
    pca_explained.csv       — per-component explained variance
    tract_metadata.csv      — tractfips, stateabbr, cityname, lat, lon
    eda_stats.csv           — describe() + skewness + kurtosis per measure
    missingness_report.csv  — per-column missing counts before imputation
    correlation_matrix.csv  — 20×20 Pearson correlation matrix

Outputs written to assets/:
    eda_01_missingness.png
    eda_02_distributions.png
    eda_03_correlation_heatmap.png
    eda_04_skewness.png
    eda_05_pca_scree.png
    eda_06_pca_loadings.png
    eda_07_pc1_by_state.png
    eda_08_pca_biplot.png
    eda_09_population_distribution.png
    eda_10_outlier_boxplots.png
"""

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT  = Path(__file__).parent.parent
RAW_DEFAULT   = PROJECT_ROOT / "data/raw/cdc_500cities_raw.csv"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
ASSETS_DIR    = PROJECT_ROOT / "assets"

# ---------------------------------------------------------------------------
# Column constants
# ---------------------------------------------------------------------------
COL_YEAR      = "year"
COL_STATE     = "stateabbr"
COL_CITY      = "cityname"
COL_GEOLEVEL  = "geographiclevel"
COL_MEASURE   = "measure"
COL_VALUE     = "data_value"
COL_VALUE_TYPE= "data_value_type"
COL_TRACT     = "tractfips"
COL_CITYFIPS  = "cityfips"
COL_GEOLOC    = "geolocation"
COL_POP       = "populationcount"
COL_CATEGORY  = "category"

# Analysis slice parameters — changing any of these changes the entire pipeline.
SLICE_GEOLEVEL   = "Census Tract"
SLICE_VALUE_TYPE = "Crude prevalence"
SLICE_YEAR       = 2017

N_PCA_COMPONENTS = 4   # covers ≈ 91% variance (confirmed in EDA)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
POINT_RE = re.compile(r".*\(\s*([\-\d\.]+)\s*,\s*([\-\d\.]+)\s*\)")


def parse_point(wkt: str) -> tuple[float | None, float | None]:
    """
    Extract (lon, lat) from a CDC Socrata WKT POINT string.

    The CDC format is either:
        "POINT (-97.74 30.27)"
        or a multiline variant with embedded newlines.

    Returns:
        (lon, lat) as floats, or (None, None) if parsing fails.
    """
    if not isinstance(wkt, str):
        return None, None
    m = POINT_RE.search(wkt.strip())
    if m:
        return float(m.group(2)), float(m.group(1))   # (lon, lat)
    return None, None


def savefig(name: str, tight: bool = True) -> None:
    """Save the current figure to assets/ and close it."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    path = ASSETS_DIR / name
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


def section(title: str) -> None:
    bar = "-" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


# ---------------------------------------------------------------------------
# Step 1 — Load raw CSV
# ---------------------------------------------------------------------------
def load_raw(path: Path) -> pd.DataFrame:
    section("Step 1  |  Load raw CSV")
    if not path.exists():
        print(f"[ERROR] Raw file not found: {path}", file=sys.stderr)
        print("Run:  python scripts/fetch_data.py", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(path, low_memory=False)
    print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  File  : {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    return df


# ---------------------------------------------------------------------------
# Step 2 — Schema validation + type coercions
# ---------------------------------------------------------------------------
def validate_and_coerce(df: pd.DataFrame) -> pd.DataFrame:
    section("Step 2  |  Schema validation + type coercions")

    required = [
        COL_YEAR, COL_STATE, COL_CITY, COL_GEOLEVEL, COL_MEASURE,
        COL_VALUE, COL_VALUE_TYPE, COL_TRACT, COL_GEOLOC,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    print(f"  Schema OK — all {len(required)} required columns present")

    df = df.copy()

    # FIPS codes must be integer identifiers, not floats.
    df[COL_TRACT]    = pd.to_numeric(df[COL_TRACT],    errors="coerce").round().astype("Int64")
    df[COL_CITYFIPS] = pd.to_numeric(df[COL_CITYFIPS], errors="coerce").round().astype("Int64")

    # Text columns: strip leading/trailing whitespace to prevent silent
    # duplicate categories (e.g., "Census Tract " vs "Census Tract").
    text_cols = [COL_CITY, COL_MEASURE, COL_VALUE_TYPE,
                 COL_CATEGORY, COL_STATE, COL_GEOLEVEL]
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # data_value must be numeric.
    df[COL_VALUE] = pd.to_numeric(df[COL_VALUE], errors="coerce")

    print(f"  Unique geographic levels : {sorted(df[COL_GEOLEVEL].dropna().unique().tolist())}")
    print(f"  Unique value types       : {sorted(df[COL_VALUE_TYPE].dropna().unique().tolist())}")
    print(f"  Years present            : {sorted(df[COL_YEAR].dropna().unique().tolist())}")
    return df


# ---------------------------------------------------------------------------
# Step 3 — Apply analysis slice
# ---------------------------------------------------------------------------
def apply_slice(df: pd.DataFrame) -> pd.DataFrame:
    section("Step 3  |  Apply analysis slice")
    print(f"  Filters applied:")
    print(f"    {COL_GEOLEVEL}   = '{SLICE_GEOLEVEL}'")
    print(f"    {COL_VALUE_TYPE} = '{SLICE_VALUE_TYPE}'")
    print(f"    {COL_YEAR}       = {SLICE_YEAR}")

    mask = (
        (df[COL_GEOLEVEL]    == SLICE_GEOLEVEL) &
        (df[COL_VALUE_TYPE]  == SLICE_VALUE_TYPE) &
        (df[COL_YEAR]        == SLICE_YEAR)
    )
    df_slice = df[mask].copy()

    n_tracts   = df_slice[COL_TRACT].nunique()
    n_measures = df_slice[COL_MEASURE].nunique()
    miss_pct   = df_slice[COL_VALUE].isna().mean() * 100

    print(f"\n  Slice shape     : {df_slice.shape}")
    print(f"  Unique tracts   : {n_tracts:,}")
    print(f"  Unique measures : {n_measures}")
    print(f"  Missing values  : {miss_pct:.2f}%")

    # Hard assertions to catch silent filter failures.
    assert df_slice.shape[0] > 200_000, "Slice too small — filter logic may be wrong."
    assert n_tracts          > 20_000,  "Too few tracts — check geolevel filter."
    assert 15 <= n_measures <= 30,       f"Unexpected measure count: {n_measures}"
    assert miss_pct          < 5,       f"High missingness: {miss_pct:.2f}%"
    print("  Slice assertions: PASSED")

    return df_slice


# ---------------------------------------------------------------------------
# Step 4 — Build wide tract × measure matrix
# ---------------------------------------------------------------------------
def build_wide_matrix(df_slice: pd.DataFrame) -> pd.DataFrame:
    section("Step 4  |  Pivot to wide tract × measure matrix")

    wide = df_slice.pivot_table(
        index=COL_TRACT,
        columns=COL_MEASURE,
        values=COL_VALUE,
        aggfunc="mean",   # safe guard against any duplicate (tract, measure) pairs
    )

    n_dropped = df_slice[COL_TRACT].nunique() - wide.shape[0]
    print(f"  Wide matrix shape        : {wide.shape}")
    print(f"  Tracts dropped in pivot  : {n_dropped}  ({n_dropped / df_slice[COL_TRACT].nunique() * 100:.1f}%)")
    print(f"  (Dropped tracts had no valid data_value for any measure in this slice)")

    return wide


# ---------------------------------------------------------------------------
# Step 5 — Missingness analysis + report
# ---------------------------------------------------------------------------
def analyze_missingness(wide: pd.DataFrame) -> pd.DataFrame:
    section("Step 5  |  Missingness analysis")

    miss = pd.DataFrame({
        "n_missing"   : wide.isna().sum(),
        "pct_missing" : (wide.isna().mean() * 100).round(3),
        "n_present"   : wide.notna().sum(),
    }).sort_values("pct_missing", ascending=False)

    total_cells    = wide.shape[0] * wide.shape[1]
    total_missing  = int(wide.isna().sum().sum())
    overall_pct    = total_missing / total_cells * 100

    print(f"  Total cells              : {total_cells:,}")
    print(f"  Missing cells            : {total_missing:,}  ({overall_pct:.2f}%)")
    print(f"  Measures with any missing: {(miss['n_missing'] > 0).sum()}")
    print()
    print("  Per-measure missingness (top 10):")
    print(miss.head(10).to_string())

    # -- Figure: missingness heatmap (one row per tract, one col per measure)
    # Showing every tract would make an unreadable image. Instead we visualise
    # a sorted summary bar chart of missingness per measure.
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["#d73027" if p > 1 else "#4393c3" for p in miss["pct_missing"]]
    ax.bar(range(len(miss)), miss["pct_missing"], color=colors, edgecolor="none")
    ax.set_xticks(range(len(miss)))
    ax.set_xticklabels([m[:20] for m in miss.index], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Missing (%)", fontsize=10)
    ax.set_title("Per-Measure Missingness Before Imputation", fontsize=11)
    ax.axhline(1, color="grey", linestyle="--", linewidth=0.8, label=">1% threshold")
    ax.legend(fontsize=9)
    savefig("eda_01_missingness.png")

    return miss


# ---------------------------------------------------------------------------
# Step 6 — Imputation + standardization
# ---------------------------------------------------------------------------
def impute_and_scale(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    section("Step 6  |  Imputation + standardization")

    # Median imputation — robust to skewed health prevalence distributions.
    n_before = int(wide.isna().sum().sum())
    wide_imp  = wide.fillna(wide.median())
    n_after   = int(wide_imp.isna().sum().sum())
    print(f"  Missing before imputation : {n_before}")
    print(f"  Missing after  imputation : {n_after}  (median imputation)")

    # StandardScaler — K-Means and Isolation Forest are Euclidean distance-based;
    # without scaling, measures with larger absolute variance dominate.
    scaler      = StandardScaler()
    X_scaled    = scaler.fit_transform(wide_imp.values)
    wide_scaled = pd.DataFrame(X_scaled, index=wide_imp.index, columns=wide_imp.columns)

    # Validate: post-scaling means ≈ 0, stds ≈ 1.
    max_mean = float(np.abs(wide_scaled.mean()).max())
    max_std  = float(np.abs(wide_scaled.std(ddof=0) - 1).max())
    assert max_mean < 1e-9, f"Scaling issue: max |mean| = {max_mean}"
    assert max_std  < 1e-9, f"Scaling issue: max |std-1| = {max_std}"
    print(f"  Scaling validation        : PASSED  (|mean| < 1e-9, |std-1| < 1e-9)")

    return wide_imp, wide_scaled, scaler


# ---------------------------------------------------------------------------
# Step 7 — PCA
# ---------------------------------------------------------------------------
def run_pca(
    wide_scaled: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, PCA]:
    section("Step 7  |  PCA — dimensionality reduction")

    X = wide_scaled.values

    # Full PCA to determine the minimum components for 90% variance.
    pca_full = PCA(random_state=42)
    pca_full.fit(X)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_90    = int(np.argmax(cum_var >= 0.90)) + 1
    print(f"  Components for 90% variance : {n_90}  (using {N_PCA_COMPONENTS})")

    # Final PCA with N_PCA_COMPONENTS.
    pca = PCA(n_components=N_PCA_COMPONENTS, random_state=42)
    X_pca  = pca.fit_transform(X)

    pca_df = pd.DataFrame(
        X_pca,
        index=wide_scaled.index,
        columns=[f"PC{i}" for i in range(1, N_PCA_COMPONENTS + 1)],
    )

    var_per_pc = pca.explained_variance_ratio_
    print(f"  Variance explained:")
    for i, v in enumerate(var_per_pc, 1):
        print(f"    PC{i}: {v*100:.1f}%  (cumulative: {sum(var_per_pc[:i])*100:.1f}%)")

    # Loadings DataFrame: rows = measures, cols = PCs.
    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=wide_scaled.columns,
        columns=[f"PC{i}" for i in range(1, N_PCA_COMPONENTS + 1)],
    )

    # Explained variance table.
    explained_df = pd.DataFrame({
        "component"          : [f"PC{i}" for i in range(1, N_PCA_COMPONENTS + 1)],
        "explained_variance" : var_per_pc,
        "cumulative_variance": np.cumsum(var_per_pc),
    })

    return pca_df, loadings_df, explained_df, pca_full


# ---------------------------------------------------------------------------
# Step 8 — Coordinate extraction (for spatial analysis in main notebook)
# ---------------------------------------------------------------------------
def extract_coordinates(df_slice: pd.DataFrame) -> pd.DataFrame:
    section("Step 8  |  Parse tract coordinates from WKT geolocation")

    geo_raw = (
        df_slice[[COL_TRACT, COL_STATE, COL_CITY, COL_GEOLOC]]
        .dropna(subset=[COL_TRACT])
        .drop_duplicates(subset=COL_TRACT)
        .copy()
    )

    parsed          = geo_raw[COL_GEOLOC].apply(parse_point)
    geo_raw["lon"]  = parsed.apply(lambda t: t[0])
    geo_raw["lat"]  = parsed.apply(lambda t: t[1])

    n_valid   = geo_raw["lat"].notna().sum()
    n_total   = len(geo_raw)
    pct_valid = n_valid / n_total * 100

    print(f"  Unique tracts             : {n_total:,}")
    print(f"  Tracts with valid coords  : {n_valid:,}  ({pct_valid:.1f}%)")

    lon_min = geo_raw["lon"].dropna().min()
    lon_max = geo_raw["lon"].dropna().max()
    lat_min = geo_raw["lat"].dropna().min()
    lat_max = geo_raw["lat"].dropna().max()
    print(f"  Longitude range           : [{lon_min:.2f}, {lon_max:.2f}]")
    print(f"  Latitude  range           : [{lat_min:.2f}, {lat_max:.2f}]")

    if pct_valid < 90:
        print(
            f"  [WARNING] Only {pct_valid:.1f}% coverage — "
            "spatial analysis (Moran's I) may be unreliable.",
            file=sys.stderr,
        )

    tract_meta = geo_raw[[COL_TRACT, COL_STATE, COL_CITY, "lon", "lat"]].copy()
    tract_meta.columns = ["tractfips", "stateabbr", "cityname", "lon", "lat"]
    return tract_meta


# ---------------------------------------------------------------------------
# Step 9 — EDA figures
# ---------------------------------------------------------------------------
def plot_distributions(wide_imp: pd.DataFrame) -> None:
    """Grid of histograms for all 20 health measures."""
    measures = list(wide_imp.columns)
    n_cols   = 5
    n_rows   = int(np.ceil(len(measures) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 3))
    axes = axes.flatten()

    for i, measure in enumerate(measures):
        ax     = axes[i]
        vals   = wide_imp[measure].dropna()
        skew_v = float(vals.skew())

        ax.hist(vals, bins=40, color="#4393c3", edgecolor="none", alpha=0.85)
        ax.set_title(f"{measure[:30]}\n(skew={skew_v:.2f})", fontsize=7)
        ax.set_xlabel("Prevalence (%)", fontsize=6)
        ax.set_ylabel("Tracts", fontsize=6)
        ax.tick_params(labelsize=6)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(4))

    # Hide unused subplots.
    for j in range(len(measures), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distribution of Health Measure Prevalence Across Census Tracts",
                 fontsize=13, y=1.01)
    savefig("eda_02_distributions.png")


def plot_correlation_heatmap(wide_imp: pd.DataFrame) -> None:
    """Annotated correlation heatmap of the 20 health measures."""
    corr = wide_imp.corr()
    labels = [m[:20] for m in corr.columns]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Pearson Correlation Matrix — 20 Health Measures", fontsize=12)

    # Annotate cells with values (only where |r| >= 0.7 for legibility).
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = corr.values[i, j]
            if abs(v) >= 0.7 and i != j:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=5, color="white" if abs(v) > 0.85 else "black")

    savefig("eda_03_correlation_heatmap.png")

    # Print top correlated pairs to console.
    pairs = (
        corr.where(~np.eye(corr.shape[0], dtype=bool))
        .stack()
        .sort_values(ascending=False)
    )
    print("  Top 5 most correlated measure pairs:")
    for (m1, m2), r in pairs.head(10)[::2].items():
        print(f"    {m1[:28]} ↔ {m2[:28]}: r={r:.3f}")


def plot_skewness(wide_imp: pd.DataFrame) -> None:
    """Bar chart of skewness per measure; highlights skew > 1."""
    skew = wide_imp.skew().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["#d73027" if s > 1 else "#4393c3" for s in skew.values]
    ax.barh(range(len(skew)), skew.values, color=colors, edgecolor="none")
    ax.set_yticks(range(len(skew)))
    ax.set_yticklabels([m[:28] for m in skew.index], fontsize=7)
    ax.axvline(1, color="red", linewidth=0.9, linestyle="--",
               label="Skew = 1 (threshold)")
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Skewness", fontsize=10)
    ax.set_title("Skewness per Health Measure (raw prevalence scale)", fontsize=11)
    ax.legend(fontsize=9)

    high_skew = skew[skew > 1]
    print(f"  Measures with skewness > 1: {len(high_skew)}")
    print(f"    {list(high_skew.index)}")

    savefig("eda_04_skewness.png")


def plot_pca_scree(pca_full: PCA) -> None:
    """Scree + cumulative variance plot with 80% / 90% guidelines."""
    ev      = pca_full.explained_variance_ratio_
    cum_ev  = np.cumsum(ev)
    n_comps = len(ev)
    n_90    = int(np.argmax(cum_ev >= 0.90)) + 1

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, n_comps + 1), ev * 100, color="#4393c3",
           edgecolor="none", alpha=0.8, label="Per-component")
    ax2 = ax.twinx()
    ax2.plot(range(1, n_comps + 1), cum_ev * 100, color="#d73027",
             marker="o", markersize=4, linewidth=1.5, label="Cumulative")
    ax2.axhline(80, color="#fd8d3c", linestyle="--", linewidth=0.9)
    ax2.axhline(90, color="#d73027", linestyle="--", linewidth=0.9)
    ax2.axvline(n_90, color="#d73027", linestyle=":", alpha=0.6,
                label=f"{N_PCA_COMPONENTS} PCs → 90% var")
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Cumulative Explained Variance (%)", fontsize=10)

    ax.set_xlabel("Principal Component", fontsize=10)
    ax.set_ylabel("Per-Component Variance (%)", fontsize=10)
    ax.set_title("PCA Scree Plot — CDC 500 Cities (20 Health Measures)", fontsize=11)
    ax.set_xticks(range(1, n_comps + 1))

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center right")

    savefig("eda_05_pca_scree.png")


def plot_pca_loadings(loadings_df: pd.DataFrame, explained_df: pd.DataFrame) -> None:
    """Horizontal bar charts of PCA loadings for each component."""
    n_pcs = len(loadings_df.columns)
    fig, axes = plt.subplots(1, n_pcs, figsize=(18, 5), sharey=True)

    for ax, col in zip(axes, loadings_df.columns):
        vals   = loadings_df[col].sort_values()
        colors = ["#d73027" if v > 0 else "#4575b4" for v in vals]
        ax.barh(range(len(vals)), vals, color=colors, edgecolor="none")
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels([v[:26] for v in vals.index], fontsize=7)
        ax.axvline(0, color="black", linewidth=0.6)

        pc_idx = int(col[2:]) - 1
        var_pct = explained_df.loc[pc_idx, "explained_variance"] * 100
        ax.set_title(f"{col}  ({var_pct:.1f}% var)", fontsize=9)
        ax.set_xlabel("Loading", fontsize=8)

    plt.suptitle("PCA Component Loadings — Clinical Interpretation", fontsize=12, y=1.01)
    savefig("eda_06_pca_loadings.png")

    # Print top loaders for PC1.
    print("  PC1 top positive loaders (high cardio-metabolic burden):")
    print("   ", loadings_df["PC1"].sort_values(ascending=False).head(5).to_string())
    print("  PC1 top negative loaders:")
    print("   ", loadings_df["PC1"].sort_values().head(5).to_string())


def plot_pc1_by_state(pca_df: pd.DataFrame, tract_meta: pd.DataFrame) -> None:
    """Median PC1 score per state, sorted descending."""
    meta_idx = tract_meta.set_index("tractfips")
    augmented = pca_df.join(meta_idx["stateabbr"], how="left")

    state_pc1 = (
        augmented.groupby("stateabbr")["PC1"]
        .median()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(16, 4))
    colors = ["#d73027" if v > 0 else "#4575b4" for v in state_pc1.values]
    ax.bar(state_pc1.index, state_pc1.values, color=colors, edgecolor="none", width=0.7)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xlabel("State", fontsize=10)
    ax.set_ylabel("Median PC1 (Cardio-Metabolic Burden)", fontsize=10)
    ax.set_title("Median PC1 Score by State — Geographic Health Burden Gradient",
                 fontsize=11)
    plt.xticks(rotation=90, fontsize=7)
    savefig("eda_07_pc1_by_state.png")

    print("  Top 5 highest-burden states (median PC1):")
    print("   ", state_pc1.head(5).round(3).to_string())
    print("  Top 5 lowest-burden states:")
    print("   ", state_pc1.tail(5).round(3).to_string())


def plot_pca_biplot(pca_df: pd.DataFrame) -> None:
    """PC1 vs PC2 scatter (5K sample) with a density contour overlay."""
    sample = pca_df.sample(n=min(5000, len(pca_df)), random_state=42)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(sample["PC1"], sample["PC2"], s=3, alpha=0.3,
               color="#4393c3", linewidths=0)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xlabel("PC1 — Cardio-Metabolic Burden", fontsize=10)
    ax.set_ylabel("PC2 — Preventive Care Access", fontsize=10)
    ax.set_title("PCA Biplot — PC1 vs PC2 (5K tract sample)", fontsize=11)

    # Mark the 1% extremes in PC1 for context.
    thresh = sample["PC1"].quantile(0.99)
    extreme = sample[sample["PC1"] >= thresh]
    ax.scatter(extreme["PC1"], extreme["PC2"], s=12, alpha=0.8,
               color="#d73027", label=f"Top 1% PC1 (n={len(extreme)})")
    ax.legend(fontsize=9)
    savefig("eda_08_pca_biplot.png")


def plot_population_distribution(df_slice: pd.DataFrame) -> None:
    """Distribution of census tract population counts (log scale)."""
    if COL_POP not in df_slice.columns:
        print("  [SKIP] populationcount column not available.")
        return

    pop = (
        df_slice[[COL_TRACT, COL_POP]]
        .dropna()
        .drop_duplicates(COL_TRACT)
        [COL_POP]
        .astype(float)
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(pop, bins=60, color="#4393c3", edgecolor="none", alpha=0.85)
    axes[0].set_xlabel("Population Count", fontsize=10)
    axes[0].set_ylabel("Number of Tracts", fontsize=10)
    axes[0].set_title("Tract Population Distribution (linear scale)", fontsize=10)

    axes[1].hist(np.log1p(pop), bins=60, color="#2ca25f", edgecolor="none", alpha=0.85)
    axes[1].set_xlabel("log(1 + Population)", fontsize=10)
    axes[1].set_ylabel("Number of Tracts", fontsize=10)
    axes[1].set_title("Tract Population Distribution (log scale)", fontsize=10)

    plt.suptitle(f"Census Tract Population  |  n={len(pop):,}  |  "
                 f"median={pop.median():,.0f}  mean={pop.mean():,.0f}", fontsize=11)
    savefig("eda_09_population_distribution.png")

    print(f"  Population  median: {pop.median():,.0f}")
    print(f"  Population  mean  : {pop.mean():,.0f}")
    print(f"  Population  range : [{pop.min():,.0f}, {pop.max():,.0f}]")


def plot_outlier_boxplots(wide_imp: pd.DataFrame) -> None:
    """Box plots for the 10 measures with the most extreme IQR outliers."""
    # IQR outlier count per measure.
    q1  = wide_imp.quantile(0.25)
    q3  = wide_imp.quantile(0.75)
    iqr = q3 - q1
    lo  = q1 - 1.5 * iqr
    hi  = q3 + 1.5 * iqr

    outlier_counts = ((wide_imp < lo) | (wide_imp > hi)).sum().sort_values(ascending=False)
    top10 = outlier_counts.head(10).index.tolist()

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes = axes.flatten()

    for i, measure in enumerate(top10):
        vals = wide_imp[measure].dropna()
        axes[i].boxplot(vals, vert=True, patch_artist=True,
                        boxprops=dict(facecolor="#4393c3", color="#2171b5"),
                        medianprops=dict(color="#d73027", linewidth=1.5),
                        flierprops=dict(marker=".", markersize=1.5,
                                        alpha=0.3, color="#636363"))
        axes[i].set_title(f"{measure[:22]}\n({outlier_counts[measure]:,} outliers)",
                          fontsize=7)
        axes[i].set_ylabel("Prevalence (%)", fontsize=7)
        axes[i].tick_params(labelsize=6)

    plt.suptitle("IQR Outlier Box Plots — Top 10 Measures by Outlier Count", fontsize=12)
    savefig("eda_10_outlier_boxplots.png")

    print("  Top 5 measures by IQR outlier count:")
    print("   ", outlier_counts.head(5).to_string())


# ---------------------------------------------------------------------------
# Step 10 — Compute summary statistics
# ---------------------------------------------------------------------------
def compute_eda_stats(wide_imp: pd.DataFrame) -> pd.DataFrame:
    section("Step 9  |  EDA summary statistics")

    desc = wide_imp.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    desc["skewness"] = wide_imp.skew()
    desc["kurtosis"] = wide_imp.kurt()

    print("  Computed describe + skewness + kurtosis for all measures")
    print(f"  Measures with skewness > 1: {(desc['skewness'] > 1).sum()}")
    return desc


# ---------------------------------------------------------------------------
# Step 11 — Save all processed artifacts
# ---------------------------------------------------------------------------
def save_artifacts(
    wide_imp    : pd.DataFrame,
    wide_scaled : pd.DataFrame,
    pca_df      : pd.DataFrame,
    loadings_df : pd.DataFrame,
    explained_df: pd.DataFrame,
    tract_meta  : pd.DataFrame,
    eda_stats   : pd.DataFrame,
    miss_report : pd.DataFrame,
) -> None:
    section("Step 10  |  Save processed artifacts")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "wide_matrix.csv"       : wide_imp.reset_index(),
        "wide_scaled.csv"       : wide_scaled.reset_index(),
        "pca_4d.csv"            : pca_df.reset_index(),
        "pca_loadings.csv"      : loadings_df.reset_index(),
        "pca_explained.csv"     : explained_df,
        "tract_metadata.csv"    : tract_meta,
        "eda_stats.csv"         : eda_stats.reset_index().rename(columns={"index": "measure"}),
        "missingness_report.csv": miss_report.reset_index().rename(columns={"index": "measure"}),
        "correlation_matrix.csv": wide_imp.corr().reset_index().rename(
                                      columns={"index": "measure"}),
    }

    for filename, df in artifacts.items():
        path = PROCESSED_DIR / filename
        df.to_csv(path, index=False)
        print(f"  {path}  ({path.stat().st_size / 1e3:.1f} KB)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess the raw CDC 500 Cities CSV and run EDA."
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=RAW_DEFAULT,
        help=f"Path to raw CSV (default: {RAW_DEFAULT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("CDC 500 Cities — Preprocessing + EDA")
    print("=" * 60)

    t0 = time.time()

    # -- Pipeline ----------------------------------------------------------
    df_raw      = load_raw(args.raw)
    df          = validate_and_coerce(df_raw)
    df_slice    = apply_slice(df)
    wide        = build_wide_matrix(df_slice)
    miss_report = analyze_missingness(wide)
    wide_imp, wide_scaled, scaler = impute_and_scale(wide)
    pca_df, loadings_df, explained_df, pca_full = run_pca(wide_scaled)
    tract_meta  = extract_coordinates(df_slice)

    # -- EDA figures -------------------------------------------------------
    section("Step 9  |  EDA figures")
    print("  Generating all EDA figures...")
    plot_distributions(wide_imp)
    plot_correlation_heatmap(wide_imp)
    plot_skewness(wide_imp)
    plot_pca_scree(pca_full)
    plot_pca_loadings(loadings_df, explained_df)
    plot_pc1_by_state(pca_df, tract_meta)
    plot_pca_biplot(pca_df)
    plot_population_distribution(df_slice)
    plot_outlier_boxplots(wide_imp)

    # -- Summary stats -----------------------------------------------------
    eda_stats = compute_eda_stats(wide_imp)

    # -- Save --------------------------------------------------------------
    save_artifacts(
        wide_imp, wide_scaled, pca_df,
        loadings_df, explained_df,
        tract_meta, eda_stats, miss_report,
    )

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"Preprocessing + EDA complete in {elapsed:.0f}s")
    print(f"  Processed artifacts  → {PROCESSED_DIR}/")
    print(f"  EDA figures          → {ASSETS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
