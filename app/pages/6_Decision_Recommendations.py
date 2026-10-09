import streamlit as st

st.title("Decision Recommendations and Limitations")

st.markdown("""
## Core Research Interpretation

The integrated experimental evidence supports one central conclusion:

> Better spatial representation should come before deeper graph architectures.

The dashboard does **not** claim that graph learning is ineffective in general.  
Instead, the evidence suggests that graph-learning utility is constrained by topology fidelity when the graph is inferred from accident-event coordinates.

---

## What the Prototype Supports

### 1. Event-derived topology is representation-sensitive

Topology search shows that performance changes with:

- number of macro-region nodes
- STPC radius
- k-nearest-neighbor connectivity

This means graph construction is not neutral preprocessing.

### 2. Non-graph baselines can remain strong

The benchmark results show that LSTM and Random Forest can outperform the GCN-GRU model in some cities.

This suggests that:

- local temporal memory can be highly informative
- nonlinear tabular interactions can preserve useful signal
- approximate graph propagation can introduce structural noise

### 3. MAUP affects the learning problem itself

Changing spatial units affects:

- node histories
- adjacency edges
- edge weights
- graph sparsity
- message-passing structure

Therefore, MAUP is not only a GIS issue.  
It is a model-level representation issue.

### 4. Spatial morphology is qualitative evidence only

The maps show STPC-labelled proximate event patterns.

They do **not** prove:

- causal secondary crashes
- physical shockwave propagation
- queue-verified incident chains

---

## Scientific Safeguards

Use these terms:

- **STPC-labelled proximate events**
- **proximate secondary traffic risk**
- **event-derived graph representation**
- **incomplete road topology**
- **representation fidelity**
- **offline decision-support prototype**

Avoid these terms:

- causal secondary crash detection
- proven cascade
- shockwave proof
- real-time prediction
- live deployed system

---

## Practical Recommendations

Future work should prioritize:

1. Map-matched accident locations  
2. Directional road-network priors  
3. Traffic speed / flow data  
4. Queue length or incident clearance records  
5. Better spatial representation before deeper graph models  

---

## System Limitations

This prototype:

- uses precomputed experimental outputs
- does not retrain models
- does not perform live inference
- does not verify causality
- does not claim operational deployment

---

## Final System Message

This system is best understood as an **offline analytical decision-support prototype** for exploring how event-derived graph learning behaves under incomplete road topology.
""")