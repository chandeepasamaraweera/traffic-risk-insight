# Results and Interpretation

## 1. Evidence boundary

The primary evidence is the locked held-out benchmark for Los Angeles, Chicago, and Miami. Validation-arena results were used to select spatial configurations and must not be reported as final held-out model performance. Superseded exploratory outputs are retained only for development provenance and are excluded from the final conclusions.

The benchmark evaluates next-hour, region-level ranking of the Spatio-Temporally Proximate Collision (STPC) proxy. It does not evaluate confirmed secondary-crash identification, reconstruct causal crash chains, or measure physical traffic propagation.

## 2. Locked three-city urban benchmark

| City | Logistic Regression | Random Forest | LSTM | GCN-GRU | Best model |
|---|---:|---:|---:|---:|---|
| Los Angeles | 0.0214 | **0.0332** | 0.0262 | 0.0250 | Random Forest |
| Chicago | 0.0070 | **0.0117** | 0.0106 | 0.0098 | Random Forest |
| Miami | 0.0447 | 0.0526 | **0.0710** | 0.0644 | LSTM |

The values are held-out scikit-learn Average Precision scores, reported by the study as AUC-PR. Logistic Regression and Random Forest are deterministic fitted baselines. The reported LSTM and GCN-GRU values summarize the locked repeated-run neural results.

Average Precision must be interpreted within each city's target prevalence, spatial representation, and experimental configuration. The scores do not establish that one city is inherently safer, riskier, or more predictable than another.

## 3. City-level interpretation

### 3.1 Los Angeles

Random Forest achieved the highest held-out Average Precision at 0.0332. LSTM ranked second at 0.0262, followed by GCN-GRU at 0.0250 and Logistic Regression at 0.0214.

The result is consistent with useful nonlinear structure in the engineered tabular representation. GCN-GRU did not improve on either Random Forest or the adjacency-free LSTM. This does not establish which physical mechanism produced the predictive signal because recent accident activity, reported severity, reported incident extent, humidity, exposure, and reporting patterns are not causally separated by the available data.

### 3.2 Chicago

Random Forest achieved the highest held-out Average Precision at 0.0117. LSTM reached 0.0106, GCN-GRU reached 0.0098, and Logistic Regression reached 0.0070.

The supported conclusion is limited: the tested event-derived adjacency did not add predictive value over the strongest Chicago baselines. The result does not show that spatial dependence is absent, nor does it identify graph mismatch as the sole explanation. Proxy-label uncertainty, spatial partitioning, feature construction, model specification, optimization, and temporal conditions remain relevant limitations.

### 3.3 Miami

LSTM achieved the highest held-out Average Precision at 0.0710. GCN-GRU ranked second at 0.0644, followed by Random Forest at 0.0526 and Logistic Regression at 0.0447.

Miami provides the clearest evidence of useful temporal dependence in the locked urban benchmark. However, the graph component did not improve on the adjacency-free temporal baseline. Supporting ablation results further show that temporal history and adjacency weighting were consequential modelling choices, rather than neutral implementation details.

## 4. Cross-city conclusion

GCN-GRU did not achieve the highest held-out score in any of the three cities. The reported paired neural comparisons also did not support the pre-specified alternative that GCN-GRU outperformed the adjacency-free LSTM.

The cross-city evidence supports three conclusions:

1. accident event streams contain limited but measurable temporal and contextual ranking information;
2. the tested event-derived adjacency did not provide a consistent advantage over strong non-graph baselines;
3. greater graph-model complexity did not compensate for uncertainty in the target and spatial representation.

This does not imply that traffic risk is non-spatial or that graph learning is unsuitable for road-safety research. It means that geographic adjacency inferred from accident-event coordinates did not consistently substitute for verified road topology under the evaluated representations.

## 5. Supporting-domain evidence

The Indianapolis and Iowa experiments extend the analysis beyond the locked three-city urban benchmark. They serve as supporting evidence and must be interpreted within their own domain definitions and experimental phases.

Random Forest led the supporting Indianapolis comparison and the native-rural Iowa C2 comparison. The Iowa C1 urban-to-rural experiment acted as a transfer stress test and motivated a native-rural redesign rather than supporting direct reuse of an urban representation.

GraphSAGE was evaluated only as a supporting alternative graph-aggregation operator in the Iowa C2 native-rural experiment. It did not outperform Random Forest, LSTM, or the tested ST-GNN under that configuration. This is an experiment-specific result and must not be generalized into a conclusion about GraphSAGE as an architecture.

The supporting-domain results should not be merged numerically with the locked urban benchmark as though all experiments used an identical spatial design, target configuration, or transfer condition.

## 6. Spatial-representation findings

Spatial representation was a material source of uncertainty rather than a neutral preprocessing choice.

- **Scale:** Coarser node systems were generally more learnable because they retained more events and positive history per node. Finer systems increased sparsity and the frequency of single-class node outputs.
- **STPC radius:** Wider candidate radii increased the number of proxy-positive observations and often improved ranking, but they also changed the target definition. A selected radius is an empirical screening parameter, not a verified physical influence boundary.
- **Neighbourhood size:** Changing the number of graph neighbours altered which regional histories were combined during message passing.
- **KMeans initialization:** Repeated KMeans fitting changed region boundaries, node histories, centroids, and graph edges, producing measurable performance variation.
- **H3 comparison:** H3 exceeded the corresponding KMeans reruns in the reported zonation experiments. This demonstrates sensitivity to zonation; it does not establish hexagonal cells as true traffic units or prove universal H3 superiority.
- **Adjacency weighting:** Binary and inverse-distance adjacency behaved differently across the reported domains, showing that edge-weight usefulness was context-dependent.

The central spatial finding is that the graph representation must itself be validated. A graph model cannot recover operational road connectivity merely through greater architectural complexity.

## 7. Feature and ablation interpretation

Feature importance and ablation answer different questions. Random Forest impurity importance describes how a fitted ensemble used the available variables. An ablation measures what happens after a variable or architectural component is removed and the model is retrained. Neither analysis establishes causality.

| Feature | Predictive role | Interpretation boundary |
|---|---|---|
| Total accidents | Recent node-hour event activity | Does not separate disruption from ordinary exposure or reporting density |
| Mean severity | Context of recent reported incidents | Is not a direct measure of capacity loss, blockage, or clearance time |
| Mean reported distance | Reported incident extent | Is not observed queue length |
| Mean humidity | Environmental context | Does not isolate a causal weather mechanism |

The Miami and Iowa ablation results were not identical. Their differences show that feature composition, temporal history, and adjacency design require domain-specific validation. No single ablation setting should be presented as universally optimal.

## 8. Statistical interpretation

The primary metric is scikit-learn Average Precision, reported by the study as AUC-PR. It summarizes positive-class ranking under severe imbalance: positive region-hours should receive higher scores, and high-scoring observations should contain a greater proportion of positives.

Average Precision does not provide:

- a causal estimate;
- a verified secondary-crash probability;
- operational calibration by itself;
- a universal basis for comparing cities with different prevalence and representations;
- evidence that a validation-selected threshold is appropriate for live deployment.

Repeated neural runs quantify sensitivity to random initialization on the same historical split. They do not quantify uncertainty across independently sampled cities, datasets, or future periods.

For each reported result, preserve the city or domain, experiment phase, split, target prevalence, topology, feature order, temporal history, model version, and whether the value is deterministic or summarized across repeated runs.

## 9. Retrospective evaluation boundary

The benchmark uses chronological model partitions, a purge between overlapping windows, training-fitted feature scaling, validation-controlled model selection and threshold selection, and held-out test evaluation. These controls reduce temporal leakage in model fitting and comparison.

The evaluation is nevertheless retrospective rather than fully prospective. In the supplied core pipeline, city-level humidity imputation and the principal KMeans partition are established before the chronological model split. They can therefore reflect the full historical city sample. The dedicated zonation protocol uses train-era spatial fitting, but that stricter procedure is not applied uniformly throughout the core benchmark.

This limitation restricts prospective claims. Future work should fit all learned preprocessing on training-era data, reserve an untouched prospective evaluation period, and evaluate agreement with externally confirmed secondary incidents.

## 10. Claims supported by the evidence

The results support the following statements:

- the STPC proxy contains limited but measurable ranking signal in the evaluated domains;
- temporal and nonlinear contextual information are useful in several domains;
- GCN-GRU does not show a consistent advantage in the locked urban benchmark;
- spatial scale, zonation, temporal history, and adjacency weighting affect performance;
- event-derived geographic adjacency is not a reliable substitute for verified road topology under the tested conditions;
- representation fidelity should be improved and validated before greater graph-model complexity is expected to provide reliable gains.

## 11. Claims not supported by the evidence

Do not state that:

- STPC-positive events are verified secondary crashes;
- one crash was shown to cause another;
- the graph reconstructs the road network;
- GCN-GRU is the best overall model;
- GraphSAGE is generally inferior to other graph architectures;
- H3 is proven superior to KMeans for traffic modelling;
- a model score is a calibrated operational risk probability;
- the application is ready for autonomous alerts, dispatch, enforcement, or traffic control;
- performance transfers to an untested city, country, period, or data source.

## 12. Recommended reporting statement

> Across the locked three-city urban benchmark, Random Forest achieved the highest held-out Average Precision in Los Angeles and Chicago, while LSTM led in Miami. The reported paired neural comparisons did not support the pre-specified alternative that GCN-GRU outperformed the adjacency-free LSTM. The results indicate that accident event streams contain limited temporal and contextual ranking information, but the tested geographic adjacency did not provide a consistent substitute for verified road topology under the evaluated representations.
