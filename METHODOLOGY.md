# Methodology

## 1. Study design

This study is a quantitative, predictive, and representation-sensitive evaluation of next-hour proximate traffic risk. The research artifact is the full analytical pipeline: target construction, spatial aggregation, graph construction, tensorization, baseline modelling, graph modelling, evaluation, robustness analysis, and interpretation.

The central comparison is deliberately baseline-first. The experiment does not ask whether a graph neural network can be fitted. It asks whether message passing over an event-derived graph adds held-out predictive value beyond linear, nonlinear tabular, and adjacency-free temporal alternatives.

## 2. Data source and study domains

The source data are drawn from the US Accidents event dataset. The core urban benchmark covers Los Angeles, Chicago, and Miami. Indianapolis provides a supporting urban domain. Iowa is used for transfer and native-rural experiments.

The source is an accident event stream, not a traffic-state database. It provides event timestamps, coordinates, reported severity, reported distance/extent, and weather context. It does not provide verified primary-secondary crash links, lane-level topology, travel direction, continuous speed, flow, density, queue length, lane blockage, or incident clearance records.

## 3. Preprocessing

For each study domain, records are:

1. filtered to the relevant city or experimental area;
2. parsed to valid timestamps;
3. ordered chronologically;
4. checked for valid coordinates;
5. cleaned for the selected numerical fields;
6. retained only when the spatial and temporal fields required by the pipeline are available.

Humidity is median-imputed at the processed city/domain level before the chronological model split in the supplied core implementation. Severity and reported distance are converted to numerical values using the documented fallbacks. Consequently, the principal benchmark is retrospective: its scaler and model fitting are split-controlled, but its city-level humidity statistic is not a purely training-era estimate. The dedicated zonation protocol applies a stricter train-era spatial fit.

## 4. Secondary-crash construct and operational proxy

A secondary crash is conceptually a crash that results from an earlier incident, either at the incident scene or in the queue or backup associated with it. Confirming that relationship normally requires information about traffic state, queue development, roadway direction, incident duration, or case-level investigation. Those variables are not available in the source event stream.

The study therefore operationalizes a narrower construct: the STPC proxy. The proxy is designed for candidate screening and next-hour ranking. It must not be interpreted as ground truth for secondary crashes.

### 4.1 STPC proxy

The outcome is a **Spatio-Temporally Proximate Collision (STPC) proxy**. For an accident at time `t`, the algorithm searches only earlier records. The event is labelled positive when at least one prior event occurs within the selected geographic radius and within the preceding two hours.

The event-level search uses Haversine distance through a BallTree. Candidate radii are evaluated as experimental parameters. The backward-only rule prevents a future accident from contributing to an earlier event's label.

The proxy marks reproducible proximity. It does not establish physical influence, traffic-shockwave propagation, legal attribution, or causal linkage.

## 5. Spatial representation

### 5.1 Metric projection

Coordinates are projected dynamically into the appropriate Universal Transverse Mercator coordinate system for the study domain. Metric coordinates are used for clustering and centroid-distance calculations.

### 5.2 Macro-regions

KMeans groups accident coordinates into macro-regions. Each cluster becomes a node. In the core tensor generator, KMeans is fitted to the full processed city event set before chronological window splitting. This gives a stable retrospective city partition but allows later-event locations to influence the spatial boundaries. In the dedicated zonation experiment, KMeans is instead fitted on the window-aligned training era and then applied to the full period. These procedures answer different questions and must not be described as identical. Node count remains a modelling choice because it changes event density, target prevalence, temporal histories, and graph structure.

### 5.3 Event-derived adjacency

A weighted k-nearest-neighbour graph connects macro-region centroids. Edge weights are based on inverse metric distance. Self-loops are added and the adjacency is symmetrically normalized before graph convolution.

The graph approximates geographic neighbourhood, not verified road connectivity. Rivers, grade separation, one-way systems, restricted-access links, travel direction, and corridor structure are not observed directly.

## 6. Region-hour tensorization

Events are aggregated to hourly node observations. The default feature vector contains:

1. `Total_Accidents`
2. `Mean_Severity`
3. `Mean_Distance`
4. `Mean_Humidity`

Missing node-hours are represented in a continuous hourly index. The binary node target indicates whether at least one STPC-positive event occurs in the prediction hour.

The default temporal input contains four consecutive hourly observations. The target is the following hour, producing a next-hour, multi-node prediction task.

## 7. Experimental controls

### 7.1 Chronological partitioning

Samples are partitioned in chronological order into 70% training, 15% validation, and 15% test segments. Random shuffling is not used.

### 7.2 Overlap purge

Sliding windows overlap by construction. A purge equal to the sequence length is removed at validation and test boundaries so input hours are not shared across adjacent partitions.

### 7.3 Scaling and preprocessing scope

Min-max feature scaling is fitted on the training partition only and then applied to validation and test data. This control does not make every upstream preprocessing step training-only. In the supplied core pipeline, city-level humidity imputation and the principal KMeans partition precede the chronological split. This is a disclosed internal-validity limitation, not a fully prospective preprocessing design.

### 7.4 Model and threshold selection

Topology is selected using validation AUC-PR. The held-out test partition is not used to choose node count, radius, or neighbourhood size. Operating thresholds for neural models are derived from validation predictions and then frozen before test evaluation.

## 8. Models

### Logistic Regression

A multi-output linear baseline with class balancing. It tests whether the engineered input contains a simple linear ranking signal.

### Random Forest

A multi-output nonlinear baseline. It tests whether feature thresholds and interactions explain performance without graph message passing.

### LSTM

An adjacency-free temporal baseline. Node-feature inputs are flattened within each time step, preserving sequence order while withholding the adjacency matrix.

### GCN-GRU

The graph model applies two graph-convolution blocks with normalized adjacency, residual addition, batch normalization, and dropout. A GRU processes the resulting temporal representation, followed by dense layers that produce one probability per node.

### GraphSAGE

GraphSAGE is included as a supporting comparator in the native-rural Iowa experiment. It is not part of the locked three-city core benchmark.

## 9. Imbalance handling and metrics

The target is rare. Accuracy is therefore not used as the primary basis for scientific conclusions.

- **Primary metric:** scikit-learn Average Precision, reported by the study as AUC-PR.
- **Secondary measures:** precision, recall, and F1 at a validation-selected threshold.
- **Neural loss:** weighted binary cross-entropy.
- **Traditional models:** class-balanced estimators.

AUC-PR must be read alongside test prevalence because the no-skill reference level depends on the positive-class frequency.

## 10. Uncertainty and robustness

The experimental design includes:

- repeated LSTM and GCN-GRU training runs;
- paired, one-sided Wilcoxon signed-rank testing for the pre-specified GCN-GRU versus LSTM comparison;
- topology search over node count, STPC radius, and neighbourhood size;
- MAUP scale analysis;
- repeated KMeans zonation analysis;
- comparison with H3-derived spatial units;
- feature-removal ablations;
- temporal-history ablations;
- adjacency-form ablations;
- urban transfer and native-rural experiments.

## 11. Interpretation protocol

A graph advantage supports the narrower conclusion that the tested event-derived adjacency contains useful predictive association. It does not verify physical traffic propagation.

A graph disadvantage does not invalidate graph learning for traffic safety. It indicates that the tested event-derived representation did not add enough information beyond the baselines under the available target, features, data partitions, and optimization procedure.

## 12. Principal validity threats

- **Construct validity:** STPC is a proximity proxy rather than a verified secondary-crash label.
- **Spatial validity:** centroid distance is not road-network connectivity.
- **MAUP:** spatial scale and boundary construction can change conclusions.
- **Temporal validity:** reporting practices and event distributions may drift over time.
- **Retrospective preprocessing:** full-domain KMeans fitting and city-level humidity imputation can incorporate later-period distributional information into the representation, although not into model fitting or target chronology.
- **External validity:** city-specific performance does not establish transferability.
- **Measurement validity:** reported severity, distance, and weather variables are incomplete proxies.
- **Statistical validity:** rare positives and neural run variation limit precision.

These limitations are part of the research result, not exceptions to it.
