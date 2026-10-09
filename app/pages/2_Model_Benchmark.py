import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_loader import load_csv

st.title("Model Benchmark Comparison")

st.markdown("""
This page compares model performance across cities using held-out AUC-PR.

**Important interpretation:**  
A higher AUC-PR indicates stronger rare-event ranking performance.  
These results do not prove causal secondary-crash detection.
""")

DATA_PATH = "data/benchmark/benchmark.csv"

try:
    df = load_csv(DATA_PATH)

    city_options = sorted(df["city"].unique())
    selected_city = st.selectbox("Select city", city_options)

    filtered = df[df["city"] == selected_city].copy()

    st.subheader(f"Benchmark Results — {selected_city}")

    filtered["auc_pr_display"] = filtered["auc_pr"].map(lambda x: f"{x:.4f}")

    st.dataframe(
        filtered[["model", "auc_pr_display", "notes"]],
        use_container_width=True,
        hide_index=True
    )

    fig = px.bar(
        filtered,
        x="model",
        y="auc_pr",
        text="auc_pr",
        title=f"AUC-PR by Model — {selected_city}"
    )

    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(yaxis_title="AUC-PR", xaxis_title="Model")

    st.plotly_chart(fig, use_container_width=True)

    best_row = filtered.sort_values("auc_pr", ascending=False).iloc[0]

    st.success(
        f"Best model for {selected_city}: "
        f"{best_row['model']} with AUC-PR = {best_row['auc_pr']:.4f}"
    )

    st.warning("""
STPC labels are chronology-aware proximity labels.
They are not verified causal secondary-crash labels.
""")

except FileNotFoundError:
    st.error(f"Data file not found: {DATA_PATH}")

except Exception as e:
    st.error("An error occurred while loading benchmark data.")
    st.exception(e)