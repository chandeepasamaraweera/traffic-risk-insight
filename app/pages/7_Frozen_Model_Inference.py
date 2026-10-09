from datetime import datetime, time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st


# ============================================================
# PROJECT PATH DISCOVERY
# ============================================================

PAGE_FILE = Path(__file__).resolve()
APP_DIR = PAGE_FILE.parents[1]


def find_project_root():
    """
    Locate the project root by searching upward for a directory
    containing both the app and models folders.

    This avoids depending on the current terminal directory.
    """

    search_locations = [
        APP_DIR,
        *APP_DIR.parents,
    ]

    for candidate in search_locations:
        app_folder = candidate / "app"
        models_folder = candidate / "models"

        if (
            app_folder.is_dir()
            and models_folder.is_dir()
        ):
            return candidate

    raise FileNotFoundError(
        "Could not locate the TrafficRisk-Insight project root. "
        "Expected a parent directory containing both 'app' and "
        "'models' folders."
    )


PROJECT_ROOT = find_project_root()

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from utils.artifacts import (  # noqa: E402
    audit_package,
    load_city_data,
    validate_loaded,
)

from utils.inference import (  # noqa: E402
    create_and_load_stgnn,
    predict_one,
    verification_stats,
)

with st.sidebar.expander(
    "Resolved project paths",
    expanded=False,
):
    st.write(
        "**Page file:**",
        str(PAGE_FILE),
    )

    st.write(
        "**Application directory:**",
        str(APP_DIR),
    )

    st.write(
        "**Project root:**",
        str(PROJECT_ROOT),
    )

    st.write(
        "**Models directory:**",
        str(PROJECT_ROOT / "models"),
    )

    st.write(
        "**Models directory exists:**",
        (PROJECT_ROOT / "models").is_dir(),
    )

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Frozen Model Inference",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Frozen Model Inference")

st.caption(
    "Historical offline replay using frozen city-specific champion "
    "models. The displayed probabilities are experimental STPC-proxy "
    "risk scores, not verified causal secondary-crash predictions."
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_datetime(value):
    """
    Convert a timestamp into a consistent display format.
    """

    return pd.Timestamp(value).strftime(
        "%Y-%m-%d %H:%M"
    )


def combine_date_and_hour(
    selected_date,
    selected_hour,
):
    """
    Combine the date and hourly time inputs.
    """

    return pd.Timestamp(
        datetime.combine(
            selected_date,
            selected_hour,
        )
    )


def find_nearest_timestamp(
    requested_timestamp,
    valid_timestamps,
):
    """
    Return the nearest valid historical test timestamp.
    """

    valid_series = pd.Series(
        pd.to_datetime(valid_timestamps)
    )

    differences = (
        valid_series - requested_timestamp
    ).abs()

    nearest_position = int(
        differences.argmin()
    )

    return pd.Timestamp(
        valid_series.iloc[nearest_position]
    )


def select_historical_window(
    test_windows,
):
    """
    Allow the user to enter a year, month, date, and hour.

    If an exact historical test window is unavailable, the page
    offers the nearest available timestamp.
    """

    valid_timestamps = pd.DatetimeIndex(
        pd.to_datetime(
            test_windows["target_hour"]
        )
    )

    minimum_timestamp = pd.Timestamp(
        valid_timestamps.min()
    )

    maximum_timestamp = pd.Timestamp(
        valid_timestamps.max()
    )

    st.subheader("Historical target-hour selection")

    st.write(
        "Select a year, month, date, and hour within the valid "
        "historical test period."
    )

    input_col_1, input_col_2 = st.columns(2)

    with input_col_1:
        selected_date = st.date_input(
            "Year, month, and date",
            value=minimum_timestamp.date(),
            min_value=minimum_timestamp.date(),
            max_value=maximum_timestamp.date(),
            format="YYYY-MM-DD",
            help=(
                "The date must fall within the historical test "
                "period for the selected city."
            ),
        )

    with input_col_2:
        selected_hour = st.time_input(
            "Target hour",
            value=time(
                hour=minimum_timestamp.hour,
                minute=0,
            ),
            step=3600,
            help=(
                "Select the hourly target time. The model uses "
                "the four hours immediately before this target."
            ),
        )

    requested_timestamp = combine_date_and_hour(
        selected_date,
        selected_hour,
    )

    exact_match = (
        valid_timestamps
        == requested_timestamp
    )

    if exact_match.any():
        selected_timestamp = requested_timestamp

        st.success(
            "The selected date and hour are available "
            "in the historical test dataset."
        )

    else:
        nearest_timestamp = find_nearest_timestamp(
            requested_timestamp,
            valid_timestamps,
        )

        st.warning(
            "The selected date and hour are not an available model "
            "test window. This may occur at a purged split boundary "
            "or outside the exact hourly coverage."
        )

        nearest_col_1, nearest_col_2 = st.columns(
            [2, 1]
        )

        with nearest_col_1:
            st.info(
                "Nearest available historical target hour: "
                f"{format_datetime(nearest_timestamp)}"
            )

        with nearest_col_2:
            use_nearest = st.checkbox(
                "Use nearest available hour",
                value=True,
            )

        if not use_nearest:
            st.error(
                "Select another date and hour or enable the "
                "nearest available historical hour."
            )

            st.stop()

        selected_timestamp = nearest_timestamp

    matching_rows = test_windows.loc[
        pd.to_datetime(
            test_windows["target_hour"]
        ).eq(selected_timestamp)
    ]

    if matching_rows.empty:
        st.error(
            "The selected timestamp could not be mapped "
            "to a historical test sample."
        )

        st.stop()

    return matching_rows.iloc[0]


def create_spatial_map(
    node_results,
    city_name,
    threshold,
):
    """
    Create the node-level spatial map.

    Marker meaning:

    Red:
        predicted_probability >= threshold

    Blue:
        predicted_probability < threshold

    Marker size:
        predicted_probability
    """

    map_data = node_results.copy()

    numeric_columns = [
        "center_lat",
        "center_lng",
        "predicted_probability",
    ]

    for column in numeric_columns:
        map_data[column] = pd.to_numeric(
            map_data[column],
            errors="coerce",
        )

    map_data = map_data.dropna(
        subset=numeric_columns
    ).copy()

    if map_data.empty:
        raise ValueError(
            "No valid geographic nodes are available "
            "for the spatial map."
        )

    map_data["threshold_status"] = np.where(
        map_data["predicted_flag"],
        "At or above threshold",
        "Below threshold",
    )

    map_data["predicted_flag_text"] = np.where(
        map_data["predicted_flag"],
        "True",
        "False",
    )

    map_data["actual_label_text"] = (
        map_data["actual_label"]
        .astype(int)
        .astype(str)
    )

    map_data["probability_display"] = (
        map_data["predicted_probability"]
        .map(lambda value: f"{value:.4f}")
    )

    map_data["threshold_display"] = (
        f"{threshold:.4f}"
    )

    # Red for threshold-positive nodes.
    # Blue for nodes below the threshold.
    map_data["fill_color"] = map_data[
        "predicted_flag"
    ].apply(
        lambda flagged: (
            [220, 53, 69, 225]
            if flagged
            else [30, 136, 229, 205]
        )
    )

    # White outlines provide visibility on both dark and light maps.
    map_data["outline_color"] = (
        [[255, 255, 255, 245]]
        * len(map_data)
    )

    # Geographic marker radius measured in metres.
    # Low-probability nodes remain visible.
    map_data["marker_radius"] = (
        100
        + map_data[
            "predicted_probability"
        ] * 525
    )

    center_latitude = float(
        map_data["center_lat"].mean()
    )

    center_longitude = float(
        map_data["center_lng"].mean()
    )

    if city_name == "Chicago":
        default_zoom = 8.7
    else:
        default_zoom = 8.5

    view_state = pdk.ViewState(
        latitude=center_latitude,
        longitude=center_longitude,
        zoom=default_zoom,
        pitch=0,
        bearing=0,
    )

    node_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_data,
        get_position=[
            "center_lng",
            "center_lat",
        ],
        get_fill_color="fill_color",
        get_line_color="outline_color",
        get_radius="marker_radius",
        radius_min_pixels=5,
        radius_max_pixels=24,
        line_width_min_pixels=1,
        stroked=True,
        filled=True,
        pickable=True,
        auto_highlight=True,
    )

    tooltip = {
        "html": (
            "<b>Node ID:</b> {node_id}<br/>"
            "<b>Probability:</b> {probability_display}<br/>"
            "<b>Threshold:</b> {threshold_display}<br/>"
            "<b>Status:</b> {threshold_status}<br/>"
            "<b>Predicted flag:</b> {predicted_flag_text}<br/>"
            "<b>Actual label:</b> {actual_label_text}"
        ),
        "style": {
            "backgroundColor": "rgba(30, 30, 30, 0.96)",
            "color": "white",
            "fontSize": "13px",
            "padding": "10px",
            "borderRadius": "7px",
        },
    }

    deck = pdk.Deck(
        layers=[node_layer],
        initial_view_state=view_state,
        map_style=(
            "https://basemaps.cartocdn.com/gl/"
            "voyager-gl-style/style.json"
        ),
        tooltip=tooltip,
    )

    return deck


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Inference controls")

    city = st.selectbox(
        "City package",
        options=[
            "Chicago",
            "Miami",
        ],
        help=(
            "Miami must use the corrected Florida-only "
            "model package."
        ),
    )

    st.divider()

    st.markdown("### Inference scope")

    st.markdown(
        """
- Historical test windows only
- Frozen model weights
- No model retraining
- No live traffic feed
- No causal crash claim
        """
    )


# ============================================================
# PACKAGE AUDIT
# ============================================================

folder, missing_files = audit_package(
    PROJECT_ROOT,
    city,
)

if missing_files:
    st.error(
        f"The {city} model package is incomplete."
    )

    st.write(
        "The following required files are missing:"
    )

    st.code(
        "\n".join(
            str(folder / name)
            for name in missing_files
        ),
        language="text",
    )

    st.info(
        "Copy the listed artifacts into the appropriate "
        "city model folder and reload the page."
    )

    st.stop()


# ============================================================
# CACHED LOADERS
# ============================================================

@st.cache_data(
    show_spinner=False,
)
def cached_city_data(city_name):
    """
    Load the city-specific artifact package.
    """

    return load_city_data(
        PROJECT_ROOT,
        city_name,
    )


@st.cache_resource(
    show_spinner="Loading frozen ST-GNN model...",
)
def cached_stgnn_model(
    city_name,
    weights_modified_time,
):
    """
    Build the city-specific ST-GNN and restore its weights.

    The checkpoint modification time is part of the cache key.
    Replacing a checkpoint therefore invalidates the model cache.
    """

    (
        city_folder,
        city_metadata,
        _,
        _,
        _,
        _,
        _,
        city_adjacency,
    ) = load_city_data(
        PROJECT_ROOT,
        city_name,
    )

    weights_path = (
        city_folder
        / "stgnn_seed42_final_restored.weights.h5"
    )

    return create_and_load_stgnn(
        city_metadata,
        city_adjacency,
        weights_path,
    )


# ============================================================
# LOAD CITY PACKAGE
# ============================================================

try:
    (
        folder,
        metadata,
        thresholds,
        split_info,
        window_index,
        region_metadata,
        arrays,
        adjacency,
    ) = cached_city_data(city)

except Exception as error:
    st.error(
        f"Could not load the {city} artifact package."
    )

    st.exception(error)
    st.stop()


# ============================================================
# PACKAGE VALIDATION
# ============================================================

validation_checks = validate_loaded(
    metadata,
    split_info,
    window_index,
    region_metadata,
    arrays,
    adjacency,
)

if not all(validation_checks.values()):
    st.error(
        "Artifact validation failed. "
        "Inference has been blocked."
    )

    st.json(validation_checks)
    st.stop()


if (
    city == "Miami"
    and metadata.get("clean_rebuild") is not True
):
    st.error(
        "The selected Miami package is not marked "
        "as the corrected Florida-only clean rebuild."
    )

    st.stop()


if city == "Miami":
    expected_filter = (
        "City == Miami AND State == FL"
    )

    if (
        metadata.get("geographic_filter")
        != expected_filter
    ):
        st.error(
            "The selected Miami package does not contain "
            "the required Florida-only geographic filter."
        )

        st.stop()


with st.expander(
    "Package validation",
    expanded=False,
):
    check_table = pd.DataFrame(
        {
            "Validation check": list(
                validation_checks.keys()
            ),
            "Passed": list(
                validation_checks.values()
            ),
        }
    )

    st.dataframe(
        check_table,
        hide_index=True,
        use_container_width=True,
    )

    st.write(
        f"**City:** {metadata['city']}, "
        f"{metadata['state']}"
    )

    st.write(
        "**Topology:** "
        f"N={metadata['champion_nodes']}, "
        f"radius={metadata['champion_radius_mi']} miles, "
        f"k={metadata['champion_k_neighbors']}"
    )

    st.write(
        f"**Test samples:** "
        f"{split_info['test_samples']:,}"
    )

    st.write(
        "**Feature order:** "
        + ", ".join(metadata["features"])
    )


# ============================================================
# TEST WINDOW PREPARATION
# ============================================================

test_windows = (
    window_index.loc[
        window_index["split"].eq("test")
    ]
    .copy()
    .sort_values("split_sample_index")
    .reset_index(drop=True)
)

datetime_columns = [
    "target_hour",
    "history_hour_1",
    "history_hour_2",
    "history_hour_3",
    "history_hour_4",
]

for column in datetime_columns:
    test_windows[column] = pd.to_datetime(
        test_windows[column],
        errors="raise",
    )

if test_windows.empty:
    st.error(
        "No test windows were found in the "
        "historical window index."
    )

    st.stop()


minimum_target_hour = pd.Timestamp(
    test_windows["target_hour"].min()
)

maximum_target_hour = pd.Timestamp(
    test_windows["target_hour"].max()
)


# ============================================================
# VALID HISTORICAL PERIOD
# ============================================================

st.subheader("Historical test period")

period_col_1, period_col_2, period_col_3 = (
    st.columns(3)
)

with period_col_1:
    st.metric(
        "First valid target hour",
        format_datetime(
            minimum_target_hour
        ),
    )

with period_col_2:
    st.metric(
        "Last valid target hour",
        format_datetime(
            maximum_target_hour
        ),
    )

with period_col_3:
    st.metric(
        "Available test windows",
        f"{len(test_windows):,}",
    )


# ============================================================
# DATE AND HOUR SELECTION
# ============================================================

selected_row = select_historical_window(
    test_windows
)

selected_test_index = int(
    selected_row["split_sample_index"]
)

selected_target_hour = pd.Timestamp(
    selected_row["target_hour"]
)


# ============================================================
# SELECTED WINDOW SUMMARY
# ============================================================

st.subheader("Selected historical window")

summary_col_1, summary_col_2, summary_col_3, summary_col_4 = (
    st.columns(4)
)

with summary_col_1:
    st.metric(
        "Local test index",
        selected_test_index,
    )

with summary_col_2:
    st.metric(
        "Target hour",
        format_datetime(
            selected_target_hour
        ),
    )

with summary_col_3:
    st.metric(
        "Spatial nodes",
        metadata["champion_nodes"],
    )

with summary_col_4:
    st.metric(
        "ST-GNN threshold",
        f"{float(thresholds['STGNN_StaticKNN']):.4f}",
    )


history_labels = [
    format_datetime(
        selected_row[
            f"history_hour_{history_index}"
        ]
    )
    for history_index in range(1, 5)
]

st.write("**Four-hour model input history**")

history_table = pd.DataFrame(
    {
        "Sequence position": [
            "Hour 1",
            "Hour 2",
            "Hour 3",
            "Hour 4",
            "Target hour",
        ],
        "Timestamp": (
            history_labels
            + [
                format_datetime(
                    selected_target_hour
                )
            ]
        ),
    }
)

st.dataframe(
    history_table,
    hide_index=True,
    use_container_width=True,
)


# ============================================================
# INFERENCE SOURCE
# ============================================================

st.subheader("Inference source")

inference_mode = st.radio(
    "Choose the prediction source",
    options=[
        "Fresh frozen ST-GNN inference",
        "Saved-score replay",
    ],
    horizontal=True,
    help=(
        "Fresh inference executes the restored neural model. "
        "Saved-score replay retrieves the exported probability row."
    ),
)

saved_stgnn_scores = np.asarray(
    arrays["stgnn_test_scores.npy"][
        selected_test_index
    ],
    dtype=np.float32,
)


# ============================================================
# RUN INFERENCE
# ============================================================

if inference_mode == "Fresh frozen ST-GNN inference":
    weights_path = (
        folder
        / "stgnn_seed42_final_restored.weights.h5"
    )

    try:
        model = cached_stgnn_model(
            city,
            weights_path.stat().st_mtime_ns,
        )

        with st.spinner(
            "Running frozen ST-GNN inference..."
        ):
            predicted_scores = predict_one(
                model,
                arrays["X_test.npy"][
                    selected_test_index:
                    selected_test_index + 1
                ],
            )

    except Exception as error:
        st.error(
            "Frozen ST-GNN inference failed."
        )

        st.exception(error)
        st.stop()

    reproducibility = verification_stats(
        predicted_scores,
        saved_stgnn_scores,
    )

    verification_passed = (
        reproducibility["all_finite"]
        and reproducibility[
            "maximum_absolute_difference"
        ] <= 1e-5
    )

    if verification_passed:
        st.success(
            "Checkpoint verification passed. "
            "The fresh prediction reproduces the saved scores."
        )

    else:
        st.error(
            "Checkpoint verification failed. "
            "The fresh prediction does not reproduce "
            "the saved score row."
        )

        st.json(reproducibility)
        st.stop()

    with st.expander(
        "Reproducibility details",
        expanded=False,
    ):
        reproducibility_table = pd.DataFrame(
            {
                "Measure": [
                    "Maximum absolute difference",
                    "Mean absolute difference",
                    "All values finite",
                    "Accepted tolerance",
                ],
                "Value": [
                    (
                        f"{reproducibility['maximum_absolute_difference']:.10g}"
                    ),
                    (
                        f"{reproducibility['mean_absolute_difference']:.10g}"
                    ),
                    str(
                        reproducibility["all_finite"]
                    ),
                    "0.00001",
                ],
            }
        )

        st.dataframe(
            reproducibility_table,
            hide_index=True,
            use_container_width=True,
        )

else:
    predicted_scores = saved_stgnn_scores

    st.info(
        "Saved-score replay is active. "
        "No neural-network execution was required."
    )


# ============================================================
# NODE-LEVEL RESULTS
# ============================================================

actual_labels = np.asarray(
    arrays["y_test.npy"][
        selected_test_index
    ],
    dtype=int,
)

stgnn_threshold = float(
    thresholds["STGNN_StaticKNN"]
)

node_results = pd.DataFrame(
    {
        "node_id": np.arange(
            len(predicted_scores),
            dtype=int,
        ),
        "predicted_probability": (
            predicted_scores
        ),
        "predicted_flag": (
            predicted_scores
            >= stgnn_threshold
        ),
        "actual_label": actual_labels,
    }
)

node_results = node_results.merge(
    region_metadata,
    left_on="node_id",
    right_on="region_id",
    how="left",
    validate="one_to_one",
)

if node_results[
    [
        "center_lat",
        "center_lng",
    ]
].isna().any().any():
    st.error(
        "One or more prediction nodes could not be matched "
        "to geographic region metadata."
    )

    st.stop()

node_results = node_results.sort_values(
    "predicted_probability",
    ascending=False,
).reset_index(drop=True)


# ============================================================
# WINDOW METRICS
# ============================================================

predicted_positive_count = int(
    node_results["predicted_flag"].sum()
)

predicted_negative_count = int(
    (~node_results["predicted_flag"]).sum()
)

actual_positive_count = int(
    node_results["actual_label"].sum()
)

maximum_probability = float(
    node_results[
        "predicted_probability"
    ].max()
)

mean_probability = float(
    node_results[
        "predicted_probability"
    ].mean()
)

metric_col_1, metric_col_2, metric_col_3, metric_col_4 = (
    st.columns(4)
)

with metric_col_1:
    st.metric(
        "Maximum probability",
        f"{maximum_probability:.4f}",
    )

with metric_col_2:
    st.metric(
        "Mean node probability",
        f"{mean_probability:.4f}",
    )

with metric_col_3:
    st.metric(
        "Nodes above threshold",
        predicted_positive_count,
    )

with metric_col_4:
    st.metric(
        "Actual positive nodes",
        actual_positive_count,
    )


# ============================================================
# TABLE AND SPATIAL MAP
# ============================================================

table_column, map_column = st.columns(
    [1.05, 1.35]
)

with table_column:
    st.subheader("Highest-risk regions")

    display_results = (
        node_results[
            [
                "node_id",
                "predicted_probability",
                "predicted_flag",
                "actual_label",
            ]
        ]
        .head(20)
        .copy()
    )

    st.dataframe(
        display_results,
        hide_index=True,
        use_container_width=True,
        column_config={
            "node_id": (
                st.column_config.NumberColumn(
                    "Node ID",
                    format="%d",
                )
            ),
            "predicted_probability": (
                st.column_config.ProgressColumn(
                    "Predicted probability",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.4f",
                )
            ),
            "predicted_flag": (
                st.column_config.CheckboxColumn(
                    "Predicted flag",
                    help=(
                        "True when probability is at or "
                        "above the validation-derived threshold."
                    ),
                )
            ),
            "actual_label": (
                st.column_config.NumberColumn(
                    "Actual label",
                    format="%d",
                )
            ),
        },
    )

    st.caption(
        "Predicted flag is True when the node probability "
        "is at or above the validation-derived ST-GNN threshold."
    )


with map_column:
    st.subheader("Spatial distribution")

    # Native Streamlit legend.
    # This avoids raw HTML rendering and adapts to both themes.
    legend_col_1, legend_col_2 = st.columns(2)

    with legend_col_1:
        st.error(
            "🔴 RED: At or above threshold\n\n"
            f"Probability ≥ {stgnn_threshold:.4f}"
        )

    with legend_col_2:
        st.info(
            "🔵 BLUE: Below threshold\n\n"
            f"Probability < {stgnn_threshold:.4f}"
        )

    st.caption(
        "Marker size increases with predicted probability. "
        "Every marker represents one spatial node centroid."
    )

    try:
        spatial_deck = create_spatial_map(
            node_results,
            city,
            stgnn_threshold,
        )

        st.pydeck_chart(
            spatial_deck,
            use_container_width=True,
        )

    except Exception as error:
        st.error(
            "The spatial map could not be created."
        )

        st.exception(error)
        st.stop()

    map_metric_col_1, map_metric_col_2 = (
        st.columns(2)
    )

    with map_metric_col_1:
        st.metric(
            "Red flagged nodes",
            predicted_positive_count,
        )

    with map_metric_col_2:
        st.metric(
            "Blue below-threshold nodes",
            predicted_negative_count,
        )

    st.caption(
        "Hover over a marker to view the node ID, probability, "
        "threshold status, predicted flag, and actual label."
    )


# ============================================================
# SAVED-MODEL COMPARISON
# ============================================================

st.subheader(
    "Saved-model comparison for the selected window"
)

saved_model_scores = {
    "Logistic Regression": np.asarray(
        arrays["lr_test_scores.npy"][
            selected_test_index
        ],
        dtype=np.float32,
    ),
    "Random Forest": np.asarray(
        arrays["rf_test_scores.npy"][
            selected_test_index
        ],
        dtype=np.float32,
    ),
    "LSTM Blind": np.asarray(
        arrays["lstm_test_scores.npy"][
            selected_test_index
        ],
        dtype=np.float32,
    ),
    "ST-GNN Static KNN": saved_stgnn_scores,
}

threshold_mapping = {
    "Logistic Regression": float(
        thresholds["LogisticRegression"]
    ),
    "Random Forest": float(
        thresholds["RandomForest"]
    ),
    "LSTM Blind": float(
        thresholds["LSTM_Blind"]
    ),
    "ST-GNN Static KNN": float(
        thresholds["STGNN_StaticKNN"]
    ),
}

comparison_rows = []

for model_name, score_values in (
    saved_model_scores.items()
):
    model_threshold = threshold_mapping[
        model_name
    ]

    comparison_rows.append(
        {
            "model": model_name,
            "maximum_probability": float(
                score_values.max()
            ),
            "mean_probability": float(
                score_values.mean()
            ),
            "threshold": model_threshold,
            "nodes_above_threshold": int(
                (
                    score_values
                    >= model_threshold
                ).sum()
            ),
        }
    )

comparison_results = pd.DataFrame(
    comparison_rows
)

comparison_col_1, comparison_col_2 = (
    st.columns([1.1, 1])
)

with comparison_col_1:
    st.dataframe(
        comparison_results,
        hide_index=True,
        use_container_width=True,
        column_config={
            "model": "Model",
            "maximum_probability": (
                st.column_config.NumberColumn(
                    "Maximum probability",
                    format="%.4f",
                )
            ),
            "mean_probability": (
                st.column_config.NumberColumn(
                    "Mean probability",
                    format="%.4f",
                )
            ),
            "threshold": (
                st.column_config.NumberColumn(
                    "Validation threshold",
                    format="%.4f",
                )
            ),
            "nodes_above_threshold": (
                st.column_config.NumberColumn(
                    "Nodes above threshold",
                    format="%d",
                )
            ),
        },
    )

with comparison_col_2:
    chart_data = (
        comparison_results
        .set_index("model")[
            "maximum_probability"
        ]
        .sort_values(
            ascending=False
        )
    )

    st.bar_chart(
        chart_data,
        horizontal=True,
        use_container_width=True,
    )

st.caption(
    "Each model uses a separate validation-derived threshold. "
    "Raw probabilities should not be treated as perfectly calibrated "
    "or directly interchangeable across models."
)


# ============================================================
# EXPORT RESULTS
# ============================================================

st.subheader("Export selected-window results")

export_results = node_results.copy()

export_results["threshold"] = (
    stgnn_threshold
)

export_results["target_hour"] = (
    selected_target_hour
)

csv_output = export_results.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download node-level results as CSV",
    data=csv_output,
    file_name=(
        f"{city.lower()}_"
        f"{selected_target_hour.strftime('%Y%m%d_%H00')}_"
        "node_predictions.csv"
    ),
    mime="text/csv",
)


# ============================================================
# SCIENTIFIC LIMITATIONS
# ============================================================

with st.expander(
    "Scientific interpretation and limitations",
    expanded=False,
):
    st.markdown(
        """
- Predictions are experimental STPC-proxy risk scores.
- Red markers represent validation-threshold exceedance.
- A red marker does not prove a causal secondary crash.
- Blue markers represent below-threshold model nodes.
- A blue marker does not guarantee the absence of traffic risk.
- The graph is event-derived and is not a true road network.
- Thresholds were selected using validation data.
- Historical test windows are used for retrospective replay.
- The application does not ingest live traffic data.
- The application is not an operational warning system.
        """
    )


st.divider()

st.caption(
    "TrafficRisk-Insight | Frozen historical inference | "
    "Offline research prototype"
)