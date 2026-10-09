import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_loader import load_csv

st.title("MAUP / Zonation Sensitivity Explorer")

st.markdown("""
This page shows how spatial discretization affects event-derived graph learning.

**MAUP interpretation:**  
The Modifiable Areal Unit Problem is not only a GIS issue here.  
In this project, changing spatial units changes:

- node histories
- centroid positions
- k-nearest-neighbor edges
- edge weights
- graph sparsity
- message-passing structure

Therefore, MAUP directly affects the learning problem.
""")

DATA_PATH = "data/maup/maup.csv"

try:
    df = load_csv(DATA_PATH)

    city_options = sorted(df["city"].unique())
    selected_city = st.selectbox("Select city/domain", city_options)

    filtered = df[df["city"] == selected_city].copy()
    row = filtered.iloc[0]

    st.subheader(f"MAUP Sensitivity — {selected_city}")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("KMeans Runs", int(row["kmeans_runs"]))
    col2.metric("KMeans CV %", f"{row['kmeans_cv_percent']:.2f}%")
    col3.metric("H3 AUC-PR", f"{row['h3_auc_pr']:.4f}")
    col4.metric("H3 Percentile", f"{row['h3_percentile']:.0f}")

    st.markdown("### Source Data")

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### KMeans CV % by City/Domain")

    fig_cv = px.bar(
        df,
        x="city",
        y="kmeans_cv_percent",
        text="kmeans_cv_percent",
        title="KMeans Zonation Variability"
    )

    fig_cv.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig_cv.update_layout(
        xaxis_title="City/Domain",
        yaxis_title="KMeans CV (%)"
    )

    st.plotly_chart(fig_cv, use_container_width=True)

    st.markdown("### H3 AUC-PR by City/Domain")

    fig_h3 = px.bar(
        df,
        x="city",
        y="h3_auc_pr",
        text="h3_auc_pr",
        title="H3 Zonation AUC-PR"
    )

    fig_h3.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig_h3.update_layout(
        xaxis_title="City/Domain",
        yaxis_title="H3 AUC-PR"
    )

    st.plotly_chart(fig_h3, use_container_width=True)

    st.info(row["interpretation"])

    st.warning("""
Topology search and zonation analysis do not prove a true road graph.
They bound uncertainty caused by spatial discretization.
""")

except FileNotFoundError:
    st.error(f"Data file not found: {DATA_PATH}")

except Exception as e:
    st.error("An error occurred while loading MAUP data.")
    st.exception(e)