import streamlit as st
import pandas as pd
from utils.data_loader import load_csv
import plotly.express as px

st.title("Topology Search Explorer")

st.markdown("""
This page shows the validation-selected event-derived topology configuration for each city/domain.

The topology is defined by:

- **N**: number of macro-region nodes
- **radius**: STPC spatial proximity threshold
- **k**: k-nearest-neighbor graph connectivity

**Important interpretation:**  
The selected topology is not claimed to be the true road graph.  
It is the best-performing representation within the tested search space.
""")

DATA_PATH = "data/topology_search/topology_search.csv"

try:
    df = load_csv(DATA_PATH)

    city_options = sorted(df["city"].unique())
    selected_city = st.selectbox("Select city/domain", city_options)

    filtered = df[df["city"] == selected_city].copy()

    st.subheader(f"Champion Topology — {selected_city}")

    st.dataframe(
        filtered[
            [
                "city",
                "N",
                "radius",
                "k",
                "validation_auc_pr",
                "precision_at_70_recall",
                "f1",
                "notes",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    row = filtered.iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Nodes (N)", int(row["N"]))
    col2.metric("Radius", f"{row['radius']} mi")
    col3.metric("k-neighbors", int(row["k"]))
    col4.metric("Validation AUC-PR", f"{row['validation_auc_pr']:.4f}")

    st.markdown("### Topology Summary")

    st.info(
        f"{selected_city} uses **N={int(row['N'])}**, "
        f"**radius={row['radius']} mi**, and "
        f"**k={int(row['k'])}**."
    )

    fig = px.bar(
        filtered,
        x="city",
        y="validation_auc_pr",
        text="validation_auc_pr",
        title=f"Validation AUC-PR — {selected_city}",
    )

    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(
        xaxis_title="City/Domain",
        yaxis_title="Validation AUC-PR",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.warning("""
Topology search bounds representation uncertainty.  
It does not prove that the selected graph is the true physical road topology.
""")

except FileNotFoundError:
    st.error(f"Data file not found: {DATA_PATH}")

except Exception as e:
    st.error("An error occurred while loading topology-search data.")
    st.exception(e)
