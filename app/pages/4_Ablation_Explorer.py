import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_loader import load_csv

st.title("Ablation Study Explorer")

st.markdown("""
This page displays exact ablation outputs extracted from the source result tables.

The ablation module currently includes:

- Miami ExperimentB UrbanNative Phase6C ablation
- Iowa ExperimentC2 RuralNative Phase6C ablation

No Los Angeles, Chicago, or Indianapolis ablation table exists in the uploaded 19-table source file.
""")

DATA_PATH = "data/ablation/ablation.csv"

try:
    df = load_csv(DATA_PATH)

    city_options = sorted(df["city"].unique())
    selected_city = st.selectbox("Select city/domain", city_options)

    filtered = df[df["city"] == selected_city].copy()

    st.subheader(f"Ablation Results — {selected_city}")

    model_options = ["All"] + sorted(filtered["model_name"].unique())
    selected_model = st.selectbox("Select model", model_options)

    if selected_model != "All":
        filtered = filtered[filtered["model_name"] == selected_model].copy()

    display_df = filtered.copy()

    display_df["test_auc_pr_mean"] = display_df["test_auc_pr_mean"].map(lambda x: f"{x:.6f}")
    display_df["test_auc_pr_std"] = display_df["test_auc_pr_std"].map(lambda x: f"{x:.6f}")
    display_df["test_f1_mean"] = display_df["test_f1_mean"].map(lambda x: f"{x:.6f}")
    display_df["delta_auc_pr_vs_full"] = display_df["delta_auc_pr_vs_full"].map(lambda x: f"{x:.6f}")

    st.dataframe(
        display_df[
            [
                "ablation_name",
                "model_name",
                "test_auc_pr_mean",
                "test_auc_pr_std",
                "test_f1_mean",
                "delta_auc_pr_vs_full",
                "seq_len",
                "adjacency_mode",
                "n_runs",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### AUC-PR by Ablation Experiment")

    fig_auc = px.bar(
        filtered,
        x="ablation_name",
        y="test_auc_pr_mean",
        color="model_name",
        barmode="group",
        text="test_auc_pr_mean",
        title=f"Ablation AUC-PR Comparison — {selected_city}",
    )

    fig_auc.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig_auc.update_layout(
        xaxis_title="Ablation Experiment",
        yaxis_title="Test AUC-PR Mean",
        xaxis_tickangle=-30,
    )

    st.plotly_chart(fig_auc, use_container_width=True)

    st.markdown("### Delta AUC-PR vs Full Model")

    fig_delta = px.bar(
        filtered,
        x="ablation_name",
        y="delta_auc_pr_vs_full",
        color="model_name",
        barmode="group",
        text="delta_auc_pr_vs_full",
        title=f"Delta AUC-PR vs Full Model — {selected_city}",
    )

    fig_delta.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig_delta.update_layout(
        xaxis_title="Ablation Experiment",
        yaxis_title="Δ AUC-PR vs Full",
        xaxis_tickangle=-30,
    )

    st.plotly_chart(fig_delta, use_container_width=True)

    best_row = filtered.sort_values("test_auc_pr_mean", ascending=False).iloc[0]
    worst_delta_row = filtered.sort_values("delta_auc_pr_vs_full", ascending=True).iloc[0]

    st.success(
        f"Highest AUC-PR in this selection: "
        f"{best_row['ablation_name']} / {best_row['model_name']} "
        f"= {best_row['test_auc_pr_mean']:.6f}"
    )

    st.info(
        f"Most negative delta vs full model: "
        f"{worst_delta_row['ablation_name']} / {worst_delta_row['model_name']} "
        f"= {worst_delta_row['delta_auc_pr_vs_full']:.6f}"
    )

    st.warning("""
Ablation results are model-design sensitivity evidence.
They do not prove causal secondary-crash mechanisms.
""")

except FileNotFoundError:
    st.error(f"Data file not found: {DATA_PATH}")

except Exception as e:
    st.error("An error occurred while loading ablation data.")
    st.exception(e)