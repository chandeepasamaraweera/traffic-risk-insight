# Data Documentation

## 1. Source

The research uses the US Accidents dataset developed by Sobhan Moosavi and collaborators and distributed through Kaggle. This repository does not redistribute the full raw dataset.

Users must obtain the source data independently and follow the provider's terms, attribution requirements, and any applicable institutional data-management rules.

## 2. Fields used by the core pipeline

| Source field | Role | Processing |
|---|---|---|
| `ID` | Source-event identifier | Retained for event traceability during preprocessing. |
| `City` | Study-domain filter | Used for city-specific subsets where applicable. |
| `Start_Time` | Event chronology | Parsed, validated, sorted, and used for hourly aggregation. |
| `End_Time` | Source event timing | Parsed and validated in the research pipeline. |
| `Start_Lat` | Event location | Required for distance search and spatial representation. |
| `Start_Lng` | Event location | Required for distance search and spatial representation. |
| `Severity` | Context feature | Converted to numeric and aggregated as mean severity. |
| `Distance(mi)` | Reported incident extent | Converted to numeric and aggregated as mean reported distance. |
| `Humidity(%)` | Weather context | Median-imputed at city/domain level in the supplied core implementation, then aggregated as mean humidity. |

## 3. Derived fields

| Derived field | Definition |
|---|---|
| `Is_STPC_<radius>` | Positive when an event has an earlier event within the chosen distance and preceding two-hour window. |
| `Region_ID` | Spatial macro-region assigned by KMeans or the documented alternative zonation. |
| `Time_Step` | Start time floored to one-hour resolution. |
| `Total_Accidents` | Number of events in a region-hour. |
| `STPC_Count` | Number of STPC-positive events in a region-hour. |
| `Mean_Severity` | Mean reported severity in a region-hour. |
| `Mean_Distance` | Mean reported incident distance/extent in a region-hour. |
| `Mean_Humidity` | Mean humidity in a region-hour. |

The node-hour target is binary: it is positive when `STPC_Count > 0` in the prediction hour.

## 4. Missing-data handling

The pipeline removes records without valid event times or coordinates because target construction requires both chronology and location. Humidity is filled using the processed city/domain median before the chronological model split in the supplied core implementation. Severity and reported distance are converted to numeric values with documented fallbacks. This makes humidity imputation a retrospective preprocessing step rather than a training-only estimate.

These operations are pragmatic data-engineering controls. They do not reconstruct unavailable traffic-state measurements.

## 5. Data limitations

The source data do not contain:

- verified primary-secondary crash relationships;
- lane-level road topology;
- direction of travel;
- continuous traffic speed, flow, or density;
- queue length or queue-tail position;
- lane closure and blockage duration;
- incident clearance time;
- a verified operational influence area.

Accordingly, `Is_STPC` is a proximity-based research proxy. It must not be relabelled as a confirmed secondary crash.

## 6. Geographic and temporal representativeness

Reported events reflect the coverage, reporting processes, source integrations, and missingness of the dataset. Results may capture exposure, reporting density, local event concentration, or recurring context in addition to short-term post-incident risk.

A model evaluated in one city, spatial partition, or historical period is not validated for another. New domains require fresh data profiling, topology selection, held-out evaluation, calibration assessment, and sensitivity analysis.

## 7. Repository data policy

The curated repository may contain compact demonstration records, result tables, maps, and frozen inference artifacts. It should not contain:

- the complete raw national dataset;
- credentials or API tokens;
- local absolute paths;
- temporary notebook checkpoints;
- personally identifying information not present in the approved source;
- untracked large artifacts required for application execution.

## 8. Data provenance record

For every processed dataset or demonstration artifact, record:

- source dataset name and release/snapshot;
- acquisition method and date;
- source checksum;
- filtering rule;
- preprocessing script and commit hash;
- output row and column counts;
- output checksum;
- intended use: training, validation, testing, demonstration, or archival evidence.

## 9. Interpretation boundary

The dataset supports predictive association at the configured region-hour scale. It does not support legal attribution, reconstruction of individual crash chains, or proof that geographic neighbours exchange traffic influence.
