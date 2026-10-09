import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TrafficRisk-Insight",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# HOMEPAGE
# ============================================================

st.title("TrafficRisk-Insight")

st.subheader(
    "Offline Decision Support for Proximate Secondary Traffic Risk"
)

st.markdown(
    """
**TrafficRisk-Insight** is an offline analytical decision-support
prototype for exploring traffic-risk experiments under incomplete
road topology.

The application combines:

- precomputed research evidence;
- historical test-window replay;
- frozen city-specific model inference;
- node-level spatial visualization;
- validation-derived threshold analysis;
- reproducibility verification.

The system supports validated model packages for:

- **Chicago, Illinois**
- **Miami, Florida**, using the corrected Florida-only rebuild
"""
)


# ============================================================
# SYSTEM STATUS
# ============================================================

st.header("System Status")

status_col_1, status_col_2, status_col_3, status_col_4 = st.columns(4)

with status_col_1:
    st.metric(
        "Supported cities",
        "2",
    )

with status_col_2:
    st.metric(
        "Spatial nodes",
        "100 per city",
    )

with status_col_3:
    st.metric(
        "Historical input",
        "4 hours",
    )

with status_col_4:
    st.metric(
        "Execution mode",
        "Offline",
    )

st.success(
    "Chicago and the corrected Miami, Florida-only model packages "
    "have passed geographic, structural, and checkpoint-reproduction "
    "validation."
)


# ============================================================
# AVAILABLE MODULES
# ============================================================

st.header("Available Modules")

module_col_1, module_col_2 = st.columns(2)

with module_col_1:
    st.markdown(
        """
### 1. Topology Search

Explore how node count, STPC radius, and graph connectivity
affect experimental performance.

### 2. Model Benchmark

Compare Logistic Regression, Random Forest, spatially blind LSTM,
and static-kNN ST-GNN results.

### 3. MAUP Sensitivity

Examine how alternative spatial zoning choices affect the
learning problem.

### 4. Ablation Explorer

Inspect the contribution of model and representation components.
"""
    )

with module_col_2:
    st.markdown(
        """
### 5. Spatial Morphology

Visualize qualitative geographic patterns in STPC-labelled
proximate events.

### 6. Decision Recommendations

Review scientific interpretations, limitations, safeguards,
and future recommendations.

### 7. Frozen Model Inference

Select a historical date and hour, run the frozen city-specific
ST-GNN, inspect node-level results, and compare saved models.
"""
    )


# ============================================================
# FROZEN MODEL INFERENCE OVERVIEW
# ============================================================

st.header("Frozen Model Inference")

st.markdown(
    """
The Frozen Model Inference page allows the user to:

1. select Chicago or Miami;
2. choose a historical test date and hour;
3. run fresh inference using the restored ST-GNN checkpoint;
4. replay saved model probabilities;
5. verify fresh predictions against saved scores;
6. inspect the highest-risk spatial nodes;
7. view threshold status on a geographic map;
8. compare saved results from all four models.
"""
)

st.info(
    "Select **Frozen Model Inference** from the left navigation "
    "menu to open the historical inference page."
)


# ============================================================
# SCIENTIFIC FRAMING
# ============================================================

st.header("Scientific Framing")

st.markdown(
    """
This project studies **proximate secondary traffic risk**, not
verified causal secondary crashes.

The STPC target is a chronology-aware spatial and temporal
proximity proxy. The STPC target identifies events occurring near
earlier events under the configured space-time rule.

The STPC target does not prove:

- incident-to-incident causality;
- physical shockwave propagation;
- queue spillback;
- verified secondary crashes;
- operational emergency conditions.
"""
)


# ============================================================
# CORE RESEARCH MESSAGE
# ============================================================

st.header("Core Research Message")

st.success(
    "Under incomplete road topology, graph-learning utility is "
    "constrained by the fidelity of the spatial representation "
    "used to construct the graph."
)

st.markdown(
    """
The project therefore emphasizes:

- representation quality;
- topology sensitivity;
- chronological evaluation;
- class-imbalance handling;
- threshold provenance;
- model reproducibility;
- cautious scientific interpretation.
"""
)


# ============================================================
# SAFEGUARDS
# ============================================================

st.header("Important Safeguards")

safeguard_col_1, safeguard_col_2 = st.columns(2)

with safeguard_col_1:
    st.markdown(
        """
**The prototype does:**

- use precomputed experiment outputs;
- load frozen pre-trained model weights;
- run historical offline inference;
- compare saved model scores;
- verify checkpoint reproducibility;
- visualize node-level probabilities.
"""
    )

with safeguard_col_2:
    st.markdown(
        """
**The prototype does not:**

- ingest live traffic feeds;
- provide real-time traffic prediction;
- retrain models from Streamlit;
- verify causal secondary crashes;
- use a true road-network topology;
- claim operational deployment.
"""
    )

st.warning(
    "Validation-derived thresholds are experimental research "
    "operating points. They are not operational emergency-warning "
    "thresholds."
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "TrafficRisk-Insight | Offline decision-support prototype | "
    "Historical frozen-model inference"
)