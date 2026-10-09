# Results and Interpretation

## 1. Authoritative result boundary

The public summary uses the locked held-out benchmark for Los Angeles, Chicago, and Miami. Validation-arena results are used for topology selection and must not be reported as final held-out model performance. Superseded exploratory Chicago outputs are retained only for provenance and must not be used for the final Chicago conclusion.

## 2. Locked three-city benchmark

| City | Logistic Regression | Random Forest | LSTM | GCN-GRU | Highest held-out AUC-PR |
|---|---:|---:|---:|---:|---|
| Los Angeles | 0.0214 | **0.0332** | 0.0262 | 0.0250 | Random Forest |
| Chicago | 0.0070 | **0.0117** | 0.0106 | 0.0098 | Random Forest |
| Miami | 0.0447 | 0.0526 | **0.0710** | 0.0644 | LSTM |

The values are scikit-learn Average Precision scores on held-out chronological test data and are reported by the study as AUC-PR.

## 2.1 Source consistency note

The authoritative Chicago LSTM value used by this repository is **0.0106 ± 0.0003**. The value **0.1060** appears once in the thesis benchmark table, but it conflicts with the thesis discussion, conclusion, Streamlit benchmark figure, and the independently printed locked benchmark table. It is treated as a typographical inconsistency. No other result is changed by this correction.

## 3. What the benchmark supports

### Los Angeles

Random Forest produces the highest reported held-out AUC-PR. The result is consistent with useful nonlinear structure in the engineered tabular representation. GCN-GRU does not improve on Random Forest or LSTM in this benchmark.

### Chicago

Random Forest again ranks first, followed by LSTM and GCN-GRU. All absolute scores are low. The correct conclusion is not that Chicago is inherently less predictable than another city; the data prevalence, representation, and domain conditions differ.

### Miami

LSTM produces the highest held-out AUC-PR, with GCN-GRU second. This supports temporal dependence in the available representation, but the graph component does not provide the best result.

### Cross-city conclusion

None of the reported one-sided Wilcoxon tests in the locked urban benchmark supports the pre-specified alternative that GCN-GRU outperforms LSTM. The GCN-GRU also does not demonstrate a consistent advantage across the core urban benchmark. Model complexity alone does not compensate for missing road topology. The results favour validating the target and spatial representation before adding graph message passing.

## 4. Supporting experiments

The thesis also reports supporting evidence from Indianapolis and Iowa. Random Forest leads the supporting Indianapolis and native-rural Iowa comparisons. Iowa additionally includes GraphSAGE as a supporting graph comparator and urban-to-rural transfer stress testing.

These results broaden the representation analysis, but they should not be merged numerically with the three-city table without preserving the distinct experimental design and domain definition.

## 5. Spatial sensitivity

Spatial representation is a material source of uncertainty.

- Coarser node systems are generally more learnable because each node contains more events, though local detail is reduced.
- Repeated KMeans runs show that initialization and boundary placement affect performance.
- H3-based reruns can outperform corresponding KMeans reruns, but this demonstrates zonation sensitivity. It does not establish that H3 is a universally superior traffic representation.
- Neighbourhood size and inverse-distance weighting affect which histories are mixed during message passing.

## 6. How to read the reported AUC-PR measure

The implementation uses Average Precision to summarize positive-class ranking under imbalance. It answers whether positive region-hours tend to receive higher scores and whether high-scoring observations are positive.

It does not provide:

- a causal estimate;
- a calibrated operational probability by itself;
- a universal score that can be compared across cities without prevalence context;
- evidence that a chosen threshold is appropriate for live deployment.

For every reported model result, preserve the city/domain, split, prevalence, topology, model version, and whether the value is a single deterministic score or a mean over repeated runs.

## 7. Claims that should not be made

Do not state that:

- STPC events are verified secondary crashes;
- the graph reconstructs the road network;
- GCN-GRU is the best model overall;
- H3 is proven superior to KMeans;
- the model is ready for live traffic management;
- output scores establish why a crash occurred;
- performance transfers to untested cities, periods, or datasets.

## 8. Recommended reporting sentence

> Across the locked three-city benchmark, Random Forest achieved the highest held-out AUC-PR in Los Angeles and Chicago, while LSTM led in Miami. GCN-GRU did not show a consistent advantage, indicating that geographic adjacency inferred from accident-event coordinates did not reliably substitute for verified road topology under the tested conditions.
