# Traffic Risk Insight

## Learning proximate secondary traffic risk from accident event streams under incomplete road topology

A road crash can create a hazardous period after the initial event. Reduced capacity, braking, merging, emergency response, driver distraction, and a growing queue may expose approaching traffic to another crash. In traffic-incident management, a **secondary crash** is a crash that occurs as a result of an earlier incident, either within the original incident scene or within the queue or backup produced by that incident.

Traffic Risk Insight investigates whether the **next-hour, region-level risk of a potentially secondary event** can be ranked from historical accident records when the evidence needed to confirm true secondary crashes is unavailable. The source data do not identify primary-secondary crash pairs and do not include continuous speed, flow, queue, lane-blockage, clearance, travel-direction, or native road-network information. The project therefore predicts **proximate secondary traffic risk**, not confirmed secondary crashes.

The project evaluates a deliberately cautious question: **does an event-derived spatial graph add predictive value beyond strong non-graph baselines?** Accident coordinates are aggregated into macro-regions, connected through distance-based adjacency, and evaluated against linear, nonlinear tabular, and adjacency-free temporal models. The target is a chronology-aware **Spatio-Temporally Proximate Collision (STPC) proxy**. A later accident is STPC-positive when at least one strictly earlier accident falls within the selected spatial radius and the preceding two-hour window. This is a reproducible screening rule for proximity. It does not establish that the earlier accident caused the later one.

> **Research boundary:** This repository supports offline retrospective analysis and demonstration. It is not a real-time traffic-control system, an emergency-dispatch tool, or a method for legal or causal crash reconstruction.

## Secondary crash, STPC proxy, and model target

These terms are related but not interchangeable:

| Term | Meaning in this project |
|---|---|
| Secondary crash | A crash resulting from an earlier incident within the incident scene or the resulting queue or backup. This requires evidence of an operational relationship. |
| STPC event | A later crash that is geographically close to at least one strictly earlier crash and occurs within two hours. It is a proxy label derived from the available event stream. |
| Region-hour target | A binary next-hour outcome indicating whether a macro-region contains at least one STPC-positive event. |
| Model output | A ranking score for the next-hour STPC proxy at each macro-region, not a verified probability that a secondary crash will occur. |

This separation is the central validity safeguard of the study. Spatial and temporal proximity can identify plausible candidates, but proximity alone cannot confirm queue-mediated causation.

## Research question

> To what extent can proximate secondary traffic risk be predicted from accident event streams when verified secondary-crash labels and native road topology are unavailable?

The study addresses three supporting questions:

1. How can a chronology-aware STPC proxy be constructed without treating proximity as causality?
2. Does event-derived graph topology improve prediction beyond Logistic Regression, Random Forest, and an adjacency-free LSTM?
3. How sensitive are the findings to node count, neighbourhood size, clustering initialization, and spatial zonation?

## Main findings

The locked three-city benchmark does not show a consistent advantage for GCN-GRU. Random Forest performs best in Los Angeles and Chicago, while LSTM performs best in Miami.

| City | Logistic Regression | Random Forest | LSTM | GCN-GRU | Best model |
|---|---:|---:|---:|---:|---|
| Los Angeles | 0.0214 | **0.0332** | 0.0262 | 0.0250 | Random Forest |
| Chicago | 0.0070 | **0.0117** | 0.0106 | 0.0098 | Random Forest |
| Miami | 0.0447 | 0.0526 | **0.0710** | 0.0644 | LSTM |

Values are held-out **Average Precision (AP)** scores, reported in the study as AUC-PR.Values are held-out **Average Precision (AP)** scores, reported in the study as AUC-PR. AUC-PR is the primary metric because the target is severely imbalanced. Scores should be interpreted relative to each city's positive-class prevalence and experimental setting, not compared as if the cities were interchangeable.GraphSAGE was evaluated only as a supporting alternative graph-aggregation operator in the Iowa C2 native-rural experiment. Under that configuration, it did not outperform Random Forest, LSTM, or the tested ST-GNN. It is therefore not included in the locked three-city urban benchmark. AUC-PR is the primary metric because the target is severely imbalanced. Scores should be interpreted relative to each city's positive-class prevalence and experimental setting, not compared as if the cities were interchangeable.

The broader evidence indicates that:

- accident event streams contain limited but measurable temporal and contextual ranking information;
- geographic proximity is not a reliable substitute for verified road connectivity;
- coarser spatial representations are generally easier to learn from, but may smooth local detail;
- performance depends on spatial scale, zonation, temporal history, and adjacency weighting;
- stronger results from H3-based reruns demonstrate sensitivity to spatial representation, not universal superiority of H3;
- added graph complexity should be justified by held-out improvement over non-graph baselines.

See [SCIENTIFIC_CONTRIBUTION.md](SCIENTIFIC_CONTRIBUTION.md) for the full novelty statement, [RESULTS.md](RESULTS.md) for interpretation rules, [METHODOLOGY.md](METHODOLOGY.md) for the experimental design, and [TERMINOLOGY.md](TERMINOLOGY.md) for the distinction between secondary crashes and the STPC proxy.

## Scientific contribution

This work makes a **methodological and empirical contribution to representation-bounded traffic-risk modelling**. It does not propose GCN-GRU as a new neural architecture, and it does not claim to identify verified secondary crashes. Its contribution is the design and evaluation of a defensible learning framework for a setting in which both the target and the road graph are incomplete.

### 1. A chronology-aware proxy for secondary-risk research

The study formalizes the Spatio-Temporally Proximate Collision (STPC) proxy using only strictly earlier accidents within a selected spatial radius and a two-hour temporal window. This converts an event archive without confirmed primary-secondary links into a reproducible prediction target while preserving temporal direction and explicitly separating proximity from causality.

### 2. An event-derived spatio-temporal representation

The pipeline converts irregular accident points into hourly macro-region observations and constructs a graph without pretending that the graph is a road network. Dynamic UTM projection supports metric clustering, KMeans defines macro-regions, and weighted k-nearest-neighbour adjacency provides a testable approximation of spatial association. The representation itself is treated as a hypothesis.

### 3. A baseline-first test of graph value

The experimental design assigns a distinct scientific role to each model:

- Logistic Regression tests linear signal;
- Random Forest tests nonlinear contextual interactions;
- LSTM tests temporal dependence without adjacency;
- GCN-GRU tests the additional value of event-derived message passing.

This hierarchy makes it possible to assess whether graph structure contributes beyond simpler explanations for predictive performance.

### 4. Representation sensitivity as primary evidence

The study integrates topology search, repeated neural runs, paired Wilcoxon testing, feature and temporal ablations, urban-to-rural transfer, native-rural redesign, KMeans reruns, and H3 comparison. MAUP is therefore examined as part of model validity rather than treated as a minor preprocessing issue.

### 5. A negative-result contribution

The locked urban benchmark does not support a consistent GCN-GRU advantage. That finding is scientifically useful: it shows that geographic proximity inferred from accident events did not consistently substitute for verified road topology under the tested conditions. The main implication is that **spatial representation should be validated before greater graph-model complexity is expected to improve prediction**.

### 6. Sparse-output engineering safeguard

Fine spatial partitions created node-specific training targets that sometimes contained only one observed class. Standard binary Logistic Regression and Random Forest estimators cannot fit a decision boundary for such outputs.

The SafeNodeClassifier wrapper checks each node target before training. When both classes are present, it fits the requested estimator normally. When only one class is present, it records that class and returns a constant, probability-compatible output instead of attempting an invalid model fit.
This safeguard allows the complete multi-output baseline experiment to run without synthesizing positive observations or concealing sparsity-related failures. It is an engineering and reproducibility contribution, not a statistical solution to limited minority-class evidence.

### 7. A reproducible research artefact

The contribution includes the complete analytical artefact rather than only a trained model: proxy construction, preprocessing, graph generation, tensorization, baseline comparison, held-out evaluation, statistical testing, sensitivity analysis, artifact traceability, and a Streamlit interface for historical replay and clearly labelled synthetic demonstration.

### Contribution boundary

The contribution is not a novel graph-neural-network architecture, a verified secondary-crash detector, a causal traffic-propagation model, or a deployment-ready warning system. Its scientific value lies in the controlled evaluation of target construction, event-derived topology, baseline performance, spatial sensitivity, and the limits of graph learning when labels and road structure are incomplete. The SafeNodeClassifier is presented separately as an engineering safeguard rather than as architectural novelty.

## Analytical workflow

```text
Accident event stream
        |
        v
City/domain filtering and chronological ordering
        |
        v
Backward-looking STPC proxy construction
        |
        v
Metric projection and spatial aggregation
        |
        v
Weighted k-nearest-neighbour graph construction
        |
        v
Hourly region-feature tensorization
        |
        v
Chronological train / validation / test evaluation
        |
        v
LR, RF, LSTM and GCN-GRU comparison
        |
        v
Repeated runs, statistical tests, ablations and MAUP checks
```

## Models

| Model | Scientific role |
|---|---|
| Logistic Regression | Tests whether the engineered representation contains a simple linear ranking signal. |
| Random Forest | Tests nonlinear tabular interactions without explicit sequence or graph modelling. |
| LSTM | Tests temporal dependence without adjacency-based message passing. |
| GCN-GRU | Tests whether event-derived neighbourhood aggregation contributes beyond temporal learning. |
| GraphSAGE | Supporting graph comparator used in the native-rural Iowa experiment. |

## Input representation

Each region-hour is represented by four aggregated features:

- recent accident count;
- mean reported severity;
- mean reported incident distance/extent;
- mean humidity.

Four hourly observations form the default input window, and the model predicts whether each region is STPC-positive in the next hour. These variables are predictive proxies. Reported distance is not treated as measured queue length, severity is not treated as a direct measure of disruption, and humidity is not assigned a causal interpretation.

## Repository structure

```text
traffic-risk-insight/
|-- app/                 # Streamlit application and supporting utilities
|-- data/                # Curated demonstration and result data
|-- maps/                # Spatial artifacts used by the application
|-- models/              # Packaged model and replay artifacts used by the showcase
|-- notebooks/           # Research pipeline notebook with cleared outputs
|-- README.md            # Project overview and entry point
|-- METHODOLOGY.md       # Experimental design and validity controls
|-- RESULTS.md           # Locked benchmark and interpretation
|-- REPRODUCIBILITY.md   # Environment, execution and verification guidance
|-- DATA.md              # Dataset scope, schema and data-governance notes
|-- TERMINOLOGY.md       # Definitions and permitted claim boundaries
|-- SCIENTIFIC_CONTRIBUTION.md # Detailed contribution and novelty statement
|-- DOCUMENTATION.md     # Guide to the documentation and source-of-truth order
|-- requirements.txt     # Application runtime dependencies
`-- .gitignore           # Exclusions and explicit artifact exceptions
```

The public showcase repository is intentionally curated. It is not a byte-for-byte archive of every intermediate experiment or development artifact.

## Run the Streamlit application

### Prerequisites

- Python 3.10 or later is recommended.
- Git is required to clone the repository.
- The packaged artifacts under `models/`, `data/`, and `maps/` must remain in their repository-relative locations.

### Setup

```powershell
git clone <repository-url>
cd traffic-risk-insight
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Launch

From the repository root, locate the Streamlit entry point:

```powershell
Get-ChildItem -Path .\app -Recurse -Filter "*.py" | Select-String -Pattern "st.set_page_config"
```

Then run the file returned by that command:

```powershell
streamlit run <path-to-entry-point>
```

## Research reproducibility

The application dependencies in `requirements.txt` are intended for the showcase. Full experimental reproduction requires the broader scientific environment described in [REPRODUCIBILITY.md](REPRODUCIBILITY.md), including TensorFlow, scikit-learn, SciPy, geospatial tooling, H3, plotting libraries, and access to the source dataset.

Core safeguards include:

- strict chronological ordering;
- backward-only STPC construction;
- chronological train, validation, and test partitions;
- a purge boundary to prevent overlapping input windows across splits;
- training-only feature scaling;
- validation-only topology and threshold selection;
- a held-out test set for final comparison;
- repeated neural runs and paired Wilcoxon testing;
- MAUP scale and zonation sensitivity analysis;
- ablation tests for features, temporal history, and adjacency design.

### Retrospective evaluation boundary

The core execution pipeline performs city-level humidity imputation and fits the main KMeans partition before the chronological model split. Feature scaling, model selection, thresholds, and final evaluation remain partition-controlled, but the spatial partition and imputation statistic can reflect the full historical city sample. The thesis therefore correctly characterizes the evaluation as retrospective rather than fully prospective. Train-era spatial fitting is used in the dedicated zonation protocol, not uniformly throughout the core benchmark.

## Responsible use

This work estimates a research proxy from historical event records. It must not be used to:

- identify fault or legal responsibility;
- claim that one crash caused another;
- automate emergency response or enforcement;
- represent output probabilities as calibrated operational risk without external validation;
- deploy to a new city or data source without domain-specific evaluation.

Any operational extension would require verified road topology, directional traffic information, continuous traffic-state measurements, prospective validation, monitoring for geographic and temporal drift, and review by relevant transport-safety stakeholders.

## Author

**M.A.D.N.Chandeepa Samaraweera**  
Bachelor of Data Science  
Department of Computer and Data Science, NSBM Green University, Sri Lanka

## Supervisor

**Thilini Bakmeedeniya**  
Department of Computer and Data Science, NSBM Green University, Sri Lanka

## Academic context

This repository accompanies the undergraduate thesis:

> *Learning Proximate Secondary Traffic Risk from Accident Event Streams under Incomplete Road Topology: An Event-Derived Spatio-Temporal Graph Learning Approach* (2026).

Extended Abstract: https://www.researchgate.net/publication/408624558_Event-derived_graph_learning_for_proximate_secondary_traffic_risk_prediction_under_incomplete_road_topology

## Dataset acknowledgement

The research uses the **US Accidents** dataset developed by Sobhan Moosavi and collaborators and distributed through Kaggle. Dataset access and use remain subject to the terms and citation requirements of the original provider. Raw national data are not included in this repository.




