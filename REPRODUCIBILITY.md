# Reproducibility Guide

## 1. Two reproducibility levels

This repository separates two tasks:

1. **Showcase reproduction:** install the lightweight application requirements and run the packaged Streamlit interface with the included compact artifacts.
2. **Experimental reproduction:** obtain the source dataset and execute the scientific pipeline with the full machine-learning, deep-learning, geospatial, statistical, and visualization environment.

The showcase is not a substitute for rerunning every experiment.

## 2. Application environment

Create a clean environment from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Locate and run the Streamlit entry point:

```powershell
Get-ChildItem -Path .\app -Recurse -Filter "*.py" | Select-String -Pattern "st.set_page_config"
```

Then run the returned entry file:

```powershell
streamlit run <path-to-entry-point>
```

## 3. Full scientific environment

The research pipeline requires these package families:

- Python 3.10 or later recommended;
- TensorFlow / Keras;
- pandas and NumPy;
- scikit-learn;
- SciPy;
- pyproj;
- H3;
- matplotlib and seaborn;
- Folium;
- KaggleHub when using automated dataset retrieval.

The research source specifies minimum versions rather than an immutable lockfile. Exact hardware, TensorFlow build, GPU driver, and accelerator runtime can affect deep-learning repeatability. Reproducing the statistical conclusion therefore requires repeated runs, not comparison of a single neural score.

## 4. Source data

The raw US Accidents dataset is not distributed in this repository. Obtain it from the original provider and comply with its access terms. Preserve an immutable local copy and record:

- dataset release or snapshot identifier;
- download date;
- file size;
- SHA-256 checksum;
- schema and row count before filtering.

Do not commit the full raw dataset to Git.

## 5. Expected execution order

1. Configure the study domain and search space.
2. Set deterministic seeds and record the runtime environment.
3. Load and chronologically order accident records.
4. Build backward-looking STPC labels.
5. Project coordinates and generate spatial units.
6. Build weighted adjacency.
7. Aggregate to region-hours and create sliding windows.
8. Partition chronologically and purge split boundaries.
9. Fit scaling on the training partition only.
10. Run validation topology search.
11. Lock the selected configuration.
12. Train and evaluate final baselines on held-out data.
13. Repeat neural runs and conduct the paired test.
14. Run MAUP, ablation, and transfer diagnostics.
15. save metrics, configuration, seeds, plots, models, and checksums.

## 6. Leakage checklist

Before accepting a result, verify that:

- STPC labels use only earlier events;
- records are sorted before target construction;
- core-benchmark KMeans fitting is recorded as full-domain retrospective fitting;
- the zonation protocol is recorded separately as train-era spatial fitting;
- city-level humidity imputation is recorded as a pre-split retrospective step;
- train, validation, and test segments are chronological;
- overlapping windows are purged at split boundaries;
- the scaler is fitted only on training data;
- topology is selected from validation data only;
- operating thresholds are derived from validation data only;
- the test set is evaluated only after configuration lock;
- no result file marked superseded is used as authoritative evidence.

## 7. Artifact integrity

Generate checksums for packaged artifacts:

```powershell
Get-ChildItem .\models, .\data, .\maps -Recurse -File |
  Get-FileHash -Algorithm SHA256 |
  Sort-Object Path |
  Format-Table Hash, Path -AutoSize
```

Inspect Git-tracked files:

```powershell
git ls-files
```

Check for unexpectedly large files before push:

```powershell
Get-ChildItem -Recurse -File |
  Where-Object Length -ge 50MB |
  Sort-Object Length -Descending |
  Select-Object @{Name='MiB';Expression={[math]::Round($_.Length / 1MB, 2)}}, FullName
```

A hosting warning is not an integrity check. Record checksums and confirm that required files are tracked explicitly.

## 8. Minimum experiment record

For each run, retain:

- commit hash;
- dataset checksum;
- city/domain definition;
- date range present in the filtered data;
- feature list;
- target window and radius;
- node count and clustering seed;
- adjacency rule and `k`;
- sequence length;
- split boundaries and purge length;
- scaler-fit partition;
- imputation-fit scope;
- spatial-partition fit scope;
- model hyperparameters;
- training seed;
- package versions;
- hardware/runtime details;
- validation and test prevalence;
- AUC-PR and secondary metrics;
- threshold-selection rule;
- artifact paths and checksums.

## 9. Reproduction claims

Use precise language:

- **Reproduced:** the same pipeline, data snapshot, configuration, and evaluation protocol yield materially consistent findings across the prescribed repeated runs.
- **Replicated:** an independently implemented pipeline or another data snapshot yields consistent conclusions.
- **Executed:** the code ran successfully, without implying result equivalence.

Do not claim exact numerical reproducibility unless the full environment, data snapshot, seeds, and artifacts are controlled and verified.
