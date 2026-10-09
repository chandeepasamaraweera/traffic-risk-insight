# Scientific Contribution

## Contribution statement

This research contributes a chronology-aware, representation-sensitive framework for studying next-hour proximate secondary traffic risk from accident event streams when verified secondary-crash labels, continuous traffic-state measurements, and native road-network topology are unavailable.

The work is not presented as architectural novelty. Its scientific contribution is the controlled construction and evaluation of the target, spatial representation, baseline hierarchy, and validity safeguards required to test graph learning under incomplete topology.

## 1. Problem-level contribution

Most traffic graph-learning studies assume that nodes and edges have operational meaning. This study examines the more constrained case in which the available observations are irregular accident events and neither confirmed secondary-crash chains nor a road graph are supplied. It frames the task as a joint measurement and representation problem rather than only a model-selection problem.

## 2. Target-construction contribution

The study defines a chronology-aware Spatio-Temporally Proximate Collision (STPC) proxy. For each event, only strictly earlier accidents may satisfy the selected spatial radius and preceding two-hour condition. This design:

- preserves temporal direction;
- prevents future accidents from creating earlier labels;
- provides a reproducible screening target;
- avoids representing proximity as verified causality.

The target is subsequently aggregated to the region-hour level for next-hour ranking.

## 3. Representation contribution

The pipeline constructs a testable spatial abstraction from an event stream:

1. geographic coordinates are projected to a city-appropriate UTM system;
2. accident points are aggregated into KMeans macro-regions;
3. region centroids are connected through weighted k-nearest-neighbour adjacency;
4. node observations are regularized to continuous hourly sequences;
5. recent accident count, mean severity, mean reported extent, and mean humidity form the feature representation.

The graph is explicitly described as event-derived rather than road-derived. This distinction makes representation fidelity an empirical question.

## 4. Comparative-design contribution

The study uses a baseline hierarchy in which each model tests a different source of signal:

- Logistic Regression: linear structure;
- Random Forest: nonlinear contextual interactions;
- LSTM: temporal dependence without adjacency;
- GCN-GRU: temporal learning with event-derived graph aggregation;
- GraphSAGE: a supporting alternative aggregation operator in the native-rural Iowa experiment.

This design prevents graph performance from being judged only against weak controls.

## 5. Evaluation contribution

The evaluation combines:

- chronological train, validation, and test partitions;
- purge gaps for overlapping input windows;
- training-fitted scaling;
- validation-only topology and threshold selection;
- held-out Average Precision, reported as AUC-PR;
- repeated neural runs;
- a paired, one-sided Wilcoxon comparison of GCN-GRU and LSTM;
- topology, feature, temporal-history, and adjacency sensitivity;
- urban transfer and native-rural redesign;
- MAUP scale and zonation analysis using repeated KMeans and H3.

This makes spatial representation sensitivity part of the principal evidence rather than an afterthought. The contribution remains bounded by retrospective preprocessing: the core city-level KMeans partition and humidity imputation are fitted before chronological splitting, while the dedicated zonation protocol uses train-era spatial fitting.

## 6. Empirical contribution

The locked three-city urban benchmark does not support a consistent GCN-GRU advantage. Random Forest produces the highest held-out score in Los Angeles and Chicago, while LSTM leads in Miami. Supporting Indianapolis and native-rural Iowa results are also led by Random Forest. Coarser spatial systems are generally more learnable, and results change with zonation, KMeans initialization, temporal history, and adjacency weighting.

The empirical contribution is therefore a boundary result: accident event streams contain limited temporal and contextual ranking information, but the tested geographic adjacency does not consistently replace verified road topology.

## 7. Theoretical implication

The results support a representation-first principle:

> Representation fidelity should be validated before additional graph-model complexity is expected to provide reliable gains.

A graph model can only exploit relationships encoded by its nodes and edges. When those units are inferred from event density and centroid distance, graph message passing may combine geographic neighbours that are not operationally connected. Model sophistication does not remove that measurement limitation.

## 8. Practical contribution

The immediate use is retrospective, analyst-oriented screening. Region-hour rankings, topology comparisons, and spatial-sensitivity views can support investigation, data-quality review, and identification of periods or areas requiring closer examination.

The current evidence does not support autonomous alerts, emergency dispatch, legal attribution, or direct transfer to a new jurisdiction. Operational use would require road-anchored topology, traffic-state data, confirmed secondary incidents, prospective validation, calibration and drift monitoring, governance, and human oversight.

## 9. Research artefact

The submitted artefact comprises:

- the research pipeline;
- saved experimental and model artifacts;
- authoritative-result and provenance logic;
- a curated software showcase;
- a Streamlit interface for historical frozen inference and explicitly synthetic demonstration;
- documentation that separates verified findings from interpretation.

## 10. What is not claimed

This research does not claim:

- a novel graph neural-network architecture;
- verified identification of secondary crashes;
- recovery of the true road network;
- causal estimation of traffic-shockwave propagation;
- universally superior KMeans or H3 zonation;
- a calibrated live-risk service;
- deployment readiness.

These boundaries are part of the scientific contribution because they define what the evidence can support.
