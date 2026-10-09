from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Real-Time Deployment Roadmap",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PAGE STYLING
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1450px;
            padding-top: 1.6rem;
            padding-bottom: 2rem;
        }

        .roadmap-stage {
            min-height: 180px;
            padding: 1rem;
            border-radius: 14px;
            border: 1px solid rgba(128, 128, 128, 0.28);
            background-color: rgba(100, 120, 160, 0.10);
        }

        .roadmap-number {
            display: inline-flex;
            width: 34px;
            height: 34px;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            background-color: #5B7CFA;
            color: white;
            font-weight: 700;
            margin-bottom: 0.65rem;
        }

        .roadmap-stage h4 {
            margin: 0 0 0.45rem 0;
        }

        .roadmap-stage p {
            margin: 0;
            line-height: 1.45;
        }

        .summary-card {
            min-height: 200px;
            padding: 1rem;
            border-radius: 14px;
            border: 1px solid rgba(128, 128, 128, 0.28);
            background-color: rgba(100, 120, 160, 0.08);
        }

        .use-case-card {
            border-top: 4px solid #5B7CFA;
        }

        .benefits-card {
            border-top: 4px solid #35B779;
        }

        .takeaways-card {
            border-top: 4px solid #E5A93D;
        }

        .summary-card h4 {
            margin: 0 0 0.6rem 0;
        }

        .summary-card p {
            margin: 0.42rem 0;
            line-height: 1.42;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.title("Real-Time Data and Deployment Roadmap")

st.caption(
    "A concise pathway from the completed research prototype "
    "to a monitored real-world deployment."
)

st.info(
    "The Synthetic September Demo demonstrates the complete inference "
    "workflow. A future real-time system would replace the synthetic "
    "source with validated live traffic and weather feeds while retaining "
    "the frozen spatial topology, fitted scaler, four-hour input contract, "
    "saved model, and validation-derived threshold."
)


# ============================================================
# PROPOSED REAL-TIME WORKFLOW
# ============================================================

st.subheader("Proposed real-time workflow")

roadmap_stages = [
    {
        "title": "Live Data Feeds",
        "description": (
            "Receive timestamped traffic incidents and weather "
            "observations from trusted external sources."
        ),
    },
    {
        "title": "Validate and Aggregate",
        "description": (
            "Clean records, reject invalid observations, assign "
            "events to frozen regions, and calculate hourly features."
        ),
    },
    {
        "title": "Rolling 4-Hour Window",
        "description": (
            "Maintain the latest four complete hourly snapshots "
            "in the exact trained feature and region order."
        ),
    },
    {
        "title": "Saved Model Inference",
        "description": (
            "Apply the frozen city-specific scaler and ST-GNN "
            "without retraining to generate 100 regional scores."
        ),
    },
    {
        "title": "Dashboard and Alerts",
        "description": (
            "Refresh the risk map, highlight validation-threshold "
            "exceedances, and preserve an auditable prediction log."
        ),
    },
]

stage_columns = st.columns(
    len(roadmap_stages),
    gap="small",
)

for stage_number, stage in enumerate(roadmap_stages, start=1):
    with stage_columns[stage_number - 1]:
        st.markdown(
            f"""
            <div class="roadmap-stage">
                <div class="roadmap-number">{stage_number}</div>
                <h4>{stage["title"]}</h4>
                <p>{stage["description"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# CORE OPERATING CONTRACT
# ============================================================

st.markdown("### Core operating contract")

metric_columns = st.columns(4)

metric_columns[0].metric(
    "Update cycle",
    "Hourly",
)

metric_columns[1].metric(
    "Historical context",
    "Previous 4 hours",
)

metric_columns[2].metric(
    "Spatial outputs",
    "100 regions",
)

metric_columns[3].metric(
    "Streamlit role",
    "Monitoring interface",
)


# ============================================================
# DEPLOYMENT PATHWAY
# ============================================================

st.subheader("Deployment pathway")

roadmap_progress = pd.DataFrame(
    {
        "Stage": [
            "Research prototype",
            "Synthetic demo",
            "Live-feed integration",
            "Controlled pilot",
            "Operational review",
        ],
        "Position": [1, 2, 3, 4, 5],
        "Status": [
            "Completed",
            "Completed",
            "Next step",
            "Future step",
            "Future step",
        ],
    }
)

connector_data = pd.DataFrame(
    {
        "start": [1, 2, 3, 4],
        "end": [2, 3, 4, 5],
    }
)

connectors = (
    alt.Chart(connector_data)
    .mark_rule(
        color="#7D879B",
        strokeWidth=4,
    )
    .encode(
        x=alt.X(
            "start:Q",
            axis=None,
            scale=alt.Scale(domain=[0.7, 5.3]),
        ),
        x2="end:Q",
        y=alt.value(65),
    )
)

nodes = (
    alt.Chart(roadmap_progress)
    .mark_circle(
        size=1100,
        stroke="white",
        strokeWidth=1.5,
    )
    .encode(
        x=alt.X(
            "Position:Q",
            axis=None,
            scale=alt.Scale(domain=[0.7, 5.3]),
        ),
        y=alt.value(65),
        color=alt.Color(
            "Status:N",
            scale=alt.Scale(
                domain=[
                    "Completed",
                    "Next step",
                    "Future step",
                ],
                range=[
                    "#35B779",
                    "#E5A93D",
                    "#69758D",
                ],
            ),
            legend=alt.Legend(
                title=None,
                orient="top",
            ),
        ),
        tooltip=[
            alt.Tooltip(
                "Stage:N",
                title="Deployment stage",
            ),
            alt.Tooltip(
                "Status:N",
                title="Status",
            ),
        ],
    )
)

labels = (
    alt.Chart(roadmap_progress)
    .mark_text(
        dy=50,
        fontSize=13,
        fontWeight="bold",
    )
    .encode(
        x=alt.X(
            "Position:Q",
            axis=None,
            scale=alt.Scale(domain=[0.7, 5.3]),
        ),
        y=alt.value(65),
        text="Stage:N",
    )
)

roadmap_chart = (
    connectors
    + nodes
    + labels
).properties(
    height=180,
)

st.altair_chart(
    roadmap_chart,
    use_container_width=True,
)

st.caption(
    "The research prototype and synthetic September demonstration are "
    "complete. Controlled live-feed integration is the next development "
    "stage, followed by pilot monitoring and operational evaluation."
)


# ============================================================
# USE CASES, BENEFITS, AND TAKEAWAYS
# ============================================================

summary_columns = st.columns(3, gap="large")

with summary_columns[0]:
    st.markdown(
        """
        <div class="summary-card use-case-card">
            <h4>Real-world use case</h4>
            <p>• Monitor changing hourly traffic-risk patterns.</p>
            <p>• Highlight regions requiring closer operator attention.</p>
            <p>• Compare regional scores as new observations arrive.</p>
            <p>• Support retrospective review using prediction logs.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with summary_columns[1]:
    st.markdown(
        """
        <div class="summary-card benefits-card">
            <h4>Expected benefits</h4>
            <p>• Consistent processing across 100 spatial regions.</p>
            <p>• Faster visibility of changing regional conditions.</p>
            <p>• Reuse of frozen preprocessing and saved models.</p>
            <p>• Clear communication through maps and alerts.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with summary_columns[2]:
    st.markdown(
        """
        <div class="summary-card takeaways-card">
            <h4>Key takeaways</h4>
            <p>• Streamlit is the monitoring interface.</p>
            <p>• Data ingestion and scheduling run separately.</p>
            <p>• Live deployment requires data-quality monitoring.</p>
            <p>• Human review and field validation remain essential.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

st.success(
    "The proposed real-time architecture reuses the validated research "
    "pipeline. Only the synthetic source is replaced by governed live "
    "feeds, followed by automated hourly aggregation, four-hour buffering, "
    "frozen-model inference, monitoring, and dashboard refresh."
)

with st.expander(
    "Research boundary",
    expanded=False,
):
    st.markdown(
        """
- This page presents a deployment roadmap, not an active live-data connection.
- Model outputs remain experimental STPC-proxy risk scores.
- The dashboard is not an operational emergency-warning system.
- Real-world use requires data agreements, continuous quality checks, drift monitoring, human oversight, and further field validation.
        """
    )

st.divider()

st.caption(
    "TrafficRisk-Insight | "
    "Real-time deployment roadmap | "
    "Research prototype"
)
