# Health Burden Profiling of U.S. Census Tracts

This project applies unsupervised data mining and spatial statistics to the CDC 500 Cities dataset to uncover geographic patterns in chronic disease burden across 26,968 U.S. census tracts. Preprocessing and EDA are handled by standalone scripts; the main notebook focuses entirely on analysis.

👉 **Start here:** [`main_notebook.ipynb`](main_notebook.ipynb)

---

## Project Video

🎥 [Watch the project walkthrough video](https://youtu.be/UotCJakMsm0)

---

## Python Version

Python 3.11.9

---

## Research Questions

| RQ | Question | Technique |
|---|---|---|
| **RQ1** | How many distinct health burden profiles exist among U.S. census tracts in PCA-reduced space, and what defines each profile? | K-Means + Agglomerative Hierarchical Clustering (course) |
| **RQ2** | Which census tracts exhibit anomalously high multi-indicator health burden beyond what their cluster membership predicts? | Isolation Forest (course) |
| **RQ3** | Does the cardio-metabolic burden score exhibit significant positive spatial autocorrelation, and where are the local hotspots? | Global + Local Moran's I / LISA (external) |

---

## Data

**Dataset:** [CDC 500 Cities: Local Data for Better Health, 2019 Release](https://data.cdc.gov/500-Cities-Places/500-Cities-Local-Data-for-Better-Health-2019-relea/6vp6-wxuq/about_data)

| Property | Detail |
|---|---|
| Source | CDC Open Data Portal (Socrata API) |
| Full dataset size | ~810K rows × 24 columns |
| Analysis slice | Census Tract level, Crude prevalence, Year 2017 |
| Wide matrix | 26,968 tracts × 20 health measures |
| License | U.S. Government Open Data (public domain) |

Raw and processed CSVs are excluded from version control (see `.gitignore`) because of file size. They are fully reproducible by running the two scripts below.

---

## How to Reproduce

### Option A — VS Code / Local

**Step 1 — Clone the repo and install dependencies**
```bash
git clone https://github.com/Rishabh-Pagaria/data_mininng_and_analysis.git
cd data_mininng_and_analysis
pip install -r requirements.txt
```

**Step 2 — Download the raw data** (~3 min, downloads ~810K rows)
```bash
python Scripts/fetch_data.py
# Output: data/raw/cdc_500cities_raw.csv
#         data/raw/download_meta.json
```

For a quick development subset (first 100K rows):
```bash
python Scripts/fetch_data.py --max-rows 100000
```

**Step 3 — Preprocess and run EDA** (~2 min)
```bash
python Scripts/preprocess_eda.py
# Output: data/processed/*.csv   (9 processed artifact files)
#         assets/eda_*.png        (10 EDA figures)
```

**Step 4 — Open the main notebook**

Open `main_notebook.ipynb` in VS Code or JupyterLab and run all cells top to bottom. No API calls happen in the notebook — it loads everything from `data/processed/`.

---

### Option B — Google Colab

**Step 1 — Clone the repo inside Colab**
```python
!git clone https://github.com/Rishabh-Pagaria/data_mininng_and_analysis.git
%cd data_mininng_and_analysis
```

**Step 2 — Install dependencies**
```python
!pip install -r requirements.txt
```

**Step 3 — Download the raw data**
```python
!python Scripts/fetch_data.py
```

**Step 4 — Preprocess and run EDA**
```python
!python Scripts/preprocess_eda.py
```

**Step 5 — Open the main notebook**

In Colab, go to `File → Open notebook → Upload` and select `main_notebook.ipynb`. Then run all cells. Since the data was fetched into the cloned folder in your Colab session, the paths will resolve correctly.

> Note: Colab sessions reset when disconnected. If your session restarts, re-run Steps 3 and 4 to regenerate the processed files before running the main notebook.

---

## Key Dependencies

| Package | Version |
|---|---|
| Python | 3.11.9 |
| pandas | 2.x |
| numpy | 1.x |
| scikit-learn | 1.x |
| scipy | 1.x |
| matplotlib | 3.x |
| esda | 2.x |
| libpysal | 4.x |
| requests | 2.x |

See [`requirements.txt`](requirements.txt) for the complete list with pinned versions.

---

## Results Summary

- **RQ1:** K-Means identifies **k = 2** as the optimal partition (Silhouette = 0.38, Davies–Bouldin = 1.04), confirmed by a Ward-linkage dendrogram. U.S. census tracts split along a single cardio-metabolic burden gradient: Cluster 1 (high burden) shows elevated stroke, diabetes, CKD, COPD, and lower preventive care utilization.

- **RQ2:** Isolation Forest flags approximately **540 tracts (2%)** as multi-indicator anomalies at the reference contamination level — tracts that are extreme across multiple PCA dimensions simultaneously. These tracts concentrate in Southeast and Appalachian states and are stable across contamination thresholds (≥ 95% overlap between 0.01 and 0.02 sets).

- **RQ3:** Global Moran's I is strongly positive and highly significant (I > 0.5, p < 0.001), robust across K ∈ {4, 8, 16}. LISA identifies dense **HH hotspot clusters** in the Southeast/Appalachian region and **LL cold spots** in the Mountain West — a spatial structure that K-Means and Isolation Forest alone cannot reveal.

---

## Repository Structure

```
.
├── main_notebook.ipynb            # Final curated analysis notebook (start here)
├── README.md
├── requirements.txt               # Full environment export (Python 3.11.9)
├── .gitignore
├── LICENSE
│
├── Scripts/
│   ├── fetch_data.py              # Step 1: downloads raw CDC data → data/raw/
│   └── preprocess_eda.py          # Step 2: preprocessing + EDA → data/processed/ + assets/
│
├── data/
│   ├── raw/
│   │   ├── cdc_500cities_raw.csv  # produced by fetch_data.py (gitignored)
│   │   └── download_meta.json     # provenance metadata (committed)
│   └── processed/                 # produced by preprocess_eda.py (gitignored)
│       ├── wide_matrix.csv        # tracts × 20 measures (imputed)
│       ├── wide_scaled.csv        # tracts × 20 measures (StandardScaler)
│       ├── pca_4d.csv             # tracts × 4 principal components
│       ├── pca_loadings.csv       # 20 measures × 4 PCs
│       ├── pca_explained.csv      # per-component explained variance
│       ├── tract_metadata.csv     # tractfips, stateabbr, cityname, lat, lon
│       ├── eda_stats.csv          # describe() + skewness + kurtosis
│       ├── missingness_report.csv # per-column missing counts pre-imputation
│       └── correlation_matrix.csv # 20×20 Pearson correlation matrix
│
├── assets/                        # EDA figures produced by preprocess_eda.py
│   ├── eda_01_missingness.png
│   ├── eda_02_distributions.png
│   ├── eda_03_correlation_heatmap.png
│   ├── eda_04_skewness.png
│   ├── eda_05_pca_scree.png
│   ├── eda_06_pca_loadings.png
│   ├── eda_07_pc1_by_state.png
│   ├── eda_08_pca_biplot.png
│   ├── eda_09_population_distribution.png
│   └── eda_10_outlier_boxplots.png
│
└── checkpoints/
    ├── checkpoint_1.ipynb         # Dataset selection, initial EDA, insights
    └── checkpoint_2.ipynb         # RQ formation, method planning, prototype runs
```