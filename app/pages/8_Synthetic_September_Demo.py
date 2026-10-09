from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path
import hashlib
import json
import sys
from typing import Any

import altair as alt
import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Synthetic September Demo",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGE_FILE = Path(__file__).resolve()
APP_DIR = PAGE_FILE.parents[1]


def find_project_root() -> Path:
    """Locate the directory containing both app and models."""

    candidates = [
        PAGE_FILE.parent,
        *PAGE_FILE.parents,
    ]

    for candidate in candidates:
        if (
            (candidate / "app").is_dir()
            and (candidate / "models").is_dir()
        ):
            return candidate

    if APP_DIR.name == "app":
        return APP_DIR.parent

    return APP_DIR


PROJECT_ROOT = find_project_root()

if str(APP_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(APP_DIR),
    )


from utils.artifacts import (  # noqa: E402
    audit_package,
    load_city_data,
    validate_loaded,
)


# ============================================================
# CONSTANTS
# ============================================================

FEATURES = [
    "Total_Accidents",
    "Mean_Severity",
    "Mean_Distance",
    "Mean_Humidity",
]

SCENARIOS = [
    "Normal Operations",
    "Morning Rush",
    "Evening Rush",
    "Heavy Rain",
    "Localized Incident Surge",
    "Combined Stress",
]

DEMO_YEAR = 2026
SEPTEMBER_START = date(DEMO_YEAR, 9, 1)
SEPTEMBER_END = date(DEMO_YEAR, 9, 30)
DEFAULT_DATE = date(DEMO_YEAR, 9, 15)
DEFAULT_HOUR = time(18, 0)
SEED = 42


# ============================================================
# DISPLAY HELPERS
# ============================================================

def format_datetime(value: Any) -> str:
    """Format a date-time value consistently."""

    return pd.Timestamp(value).strftime(
        "%Y-%m-%d %H:%M"
    )


def combine_date_and_hour(
    selected_date: date,
    selected_hour: time,
) -> pd.Timestamp:
    """Combine the selected date and time as an hourly timestamp."""

    return pd.Timestamp(
        datetime.combine(
            selected_date,
            selected_hour,
        )
    ).floor("h")


def stable_seed(*parts: Any) -> int:
    """Return a deterministic process-independent seed."""

    text = "|".join(
        str(part)
        for part in parts
    )

    digest = hashlib.sha256(
        text.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:4],
        "little",
        signed=False,
    )


def threshold_from_metadata(
    metadata: dict,
    thresholds: dict,
) -> float:
    """Return the validation-derived ST-GNN threshold."""

    if "STGNN_StaticKNN" in thresholds:
        return float(
            thresholds["STGNN_StaticKNN"]
        )

    nested_thresholds = metadata.get(
        "thresholds",
        {},
    )

    if "STGNN_StaticKNN" in nested_thresholds:
        return float(
            nested_thresholds[
                "STGNN_StaticKNN"
            ]
        )

    raise KeyError(
        "STGNN_StaticKNN threshold was not found."
    )


# ============================================================
# ARTIFACT HELPERS
# ============================================================

def locate_file(
    folder: Path,
    names: list[str],
    required: bool = True,
) -> Path | None:
    """Locate an artifact by exact name."""

    for name in names:
        direct_path = folder / name

        if direct_path.exists():
            return direct_path

    for name in names:
        matches = sorted(
            folder.rglob(name)
        )

        if matches:
            return matches[0]

    if required:
        raise FileNotFoundError(
            "Could not locate any of: "
            + ", ".join(names)
        )

    return None


@st.cache_data(
    show_spinner=False,
)
def cached_city_data(
    project_root_text: str,
    city_name: str,
):
    """Load a city package using the existing artifact utility."""

    return load_city_data(
        Path(project_root_text),
        city_name,
    )


@st.cache_resource(
    show_spinner=False,
)
def cached_scaler(
    folder_text: str,
    modified_time: float,
):
    """Load the original fitted city-specific scaler."""

    del modified_time

    scaler_path = locate_file(
        Path(folder_text),
        [
            "scaler.joblib",
        ],
    )

    return joblib.load(
        scaler_path
    )


# ============================================================
# SYNTHETIC DATA GENERATION
# ============================================================

def normalise_adjacency_for_smoothing(
    adjacency: np.ndarray,
) -> np.ndarray:
    """
    Create a row-stochastic neighbour matrix without
    self-loops for synthetic spatial smoothing.
    """

    matrix = np.asarray(
        adjacency,
        dtype=np.float64,
    ).copy()

    np.fill_diagonal(
        matrix,
        0.0,
    )

    row_sum = matrix.sum(
        axis=1,
        keepdims=True,
    )

    fallback_rows = (
        row_sum[:, 0] <= 0
    )

    matrix = np.divide(
        matrix,
        row_sum,
        out=np.zeros_like(matrix),
        where=row_sum > 0,
    )

    if np.any(fallback_rows):
        fallback_indices = np.where(
            fallback_rows
        )[0]

        matrix[
            fallback_indices,
            fallback_indices,
        ] = 1.0

    return matrix


def scenario_parameters(
    scenario: str,
    hour: int,
) -> dict[str, float]:
    """Return controlled modifiers for the selected scenario."""

    morning_rush = float(
        7 <= hour <= 9
    )

    evening_rush = float(
        16 <= hour <= 19
    )

    parameters = {
        "traffic": (
            1.0
            + 0.45 * morning_rush
            + 0.60 * evening_rush
        ),
        "humidity_add": 0.0,
        "severity_add": 0.0,
        "distance_mult": 1.0,
        "incident_mult": 1.0,
    }

    if scenario == "Morning Rush":
        if 6 <= hour <= 10:
            parameters["traffic"] *= 1.55
        else:
            parameters["traffic"] *= 1.10

    elif scenario == "Evening Rush":
        if 15 <= hour <= 20:
            parameters["traffic"] *= 1.70
        else:
            parameters["traffic"] *= 1.10

    elif scenario == "Heavy Rain":
        parameters.update(
            {
                "traffic": (
                    parameters["traffic"]
                    * 1.30
                ),
                "humidity_add": 23.0,
                "severity_add": 0.18,
                "distance_mult": 1.18,
            }
        )

    elif scenario == "Localized Incident Surge":
        parameters.update(
            {
                "traffic": (
                    parameters["traffic"]
                    * 1.12
                ),
                "severity_add": 0.22,
                "distance_mult": 1.22,
                "incident_mult": 3.20,
            }
        )

    elif scenario == "Combined Stress":
        parameters.update(
            {
                "traffic": (
                    parameters["traffic"]
                    * 1.55
                ),
                "humidity_add": 26.0,
                "severity_add": 0.35,
                "distance_mult": 1.35,
                "incident_mult": 3.80,
            }
        )

    return parameters


def generate_four_hour_window(
    city: str,
    target_hour: pd.Timestamp,
    scenario: str,
    region_metadata: pd.DataFrame,
    adjacency: np.ndarray,
    scaler,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    list[pd.Timestamp],
]:
    """
    Generate a deterministic four-hour synthetic input window.

    The output follows the frozen model contract:
    four hours, 100 regions, and four features.
    """

    required_region_columns = {
        "region_id",
        "center_lat",
        "center_lng",
    }

    missing_region_columns = (
        required_region_columns.difference(
            region_metadata.columns
        )
    )

    if missing_region_columns:
        raise ValueError(
            "Region metadata is missing: "
            + ", ".join(
                sorted(
                    missing_region_columns
                )
            )
        )

    regions = (
        region_metadata
        .copy()
        .sort_values("region_id")
        .reset_index(drop=True)
    )

    expected_region_ids = np.arange(
        100,
        dtype=int,
    )

    actual_region_ids = regions[
        "region_id"
    ].to_numpy(
        dtype=int
    )

    if (
        len(regions) != 100
        or not np.array_equal(
            actual_region_ids,
            expected_region_ids,
        )
    ):
        raise ValueError(
            "Region metadata must contain region_id "
            "0 through 99 exactly once."
        )

    history_hours = [
        target_hour
        - pd.Timedelta(hours=offset)
        for offset in (
            4,
            3,
            2,
            1,
        )
    ]

    number_of_nodes = len(
        regions
    )

    smoother = (
        normalise_adjacency_for_smoothing(
            adjacency
        )
    )

    fitted_maximum = np.asarray(
        scaler.data_max_,
        dtype=np.float64,
    )

    fitted_minimum = np.asarray(
        scaler.data_min_,
        dtype=np.float64,
    )

    if (
        fitted_maximum.shape != (4,)
        or fitted_minimum.shape != (4,)
    ):
        raise ValueError(
            "The fitted city scaler must contain "
            "exactly four features."
        )

    maximum_accidents = int(
        np.floor(
            fitted_maximum[0]
        )
    )

    maximum_severity = float(
        fitted_maximum[1]
    )

    maximum_distance = float(
        fitted_maximum[2]
    )

    maximum_humidity = float(
        fitted_maximum[3]
    )

    regional_random = np.random.default_rng(
        stable_seed(
            city,
            "regional-baseline",
            SEED,
        )
    )

    raw_activity = (
        regional_random.lognormal(
            mean=-0.10,
            sigma=0.42,
            size=number_of_nodes,
        )
    )

    regional_activity = (
        raw_activity
        / raw_activity.mean()
    )

    humidity_offsets = (
        regional_random.normal(
            loc=0.0,
            scale=1.8,
            size=number_of_nodes,
        )
    )

    focus_random = np.random.default_rng(
        stable_seed(
            city,
            target_hour.date(),
            "incident-focus",
            SEED,
        )
    )

    focus_node = int(
        focus_random.integers(
            0,
            number_of_nodes,
        )
    )

    incident_profile = (
        smoother[focus_node].copy()
    )

    if incident_profile.max() > 0:
        incident_profile = (
            incident_profile
            / incident_profile.max()
        )

    incident_profile[
        focus_node
    ] = 1.0

    raw_tensor = np.zeros(
        (
            4,
            number_of_nodes,
            4,
        ),
        dtype=np.float32,
    )

    generated_frames = []

    for time_index, timestamp in enumerate(
        history_hours
    ):
        random_generator = (
            np.random.default_rng(
                stable_seed(
                    city,
                    timestamp,
                    scenario,
                    SEED,
                )
            )
        )

        hour = int(
            timestamp.hour
        )

        parameters = scenario_parameters(
            scenario,
            hour,
        )

        if timestamp.dayofweek >= 5:
            weekend_factor = 0.82
        else:
            weekend_factor = 1.0

        if city == "Chicago":
            base_rate = 0.055
            humidity_baseline = 66.0
        else:
            base_rate = 0.085
            humidity_baseline = 74.0

        event_rate = (
            base_rate
            * regional_activity
            * parameters["traffic"]
            * weekend_factor
        )

        if scenario in {
            "Localized Incident Surge",
            "Combined Stress",
        }:
            event_rate *= (
                1.0
                + incident_profile
                * (
                    parameters[
                        "incident_mult"
                    ]
                    - 1.0
                )
            )

        event_rate = (
            0.70 * event_rate
            + 0.30
            * (
                smoother
                @ event_rate
            )
        )

        accidents = (
            random_generator.poisson(
                np.clip(
                    event_rate,
                    0.0,
                    None,
                )
            )
            .astype(int)
        )

        accidents = np.clip(
            accidents,
            0,
            maximum_accidents,
        )

        has_accident = (
            accidents > 0
        )

        severity = np.zeros(
            number_of_nodes,
            dtype=np.float64,
        )

        distance = np.zeros(
            number_of_nodes,
            dtype=np.float64,
        )

        if np.any(has_accident):
            nonzero_count = int(
                has_accident.sum()
            )

            severity_values = (
                random_generator.beta(
                    2.2,
                    4.8,
                    nonzero_count,
                )
            )

            severity_values = (
                1.0
                + severity_values
                * (
                    maximum_severity
                    - 1.0
                )
            )

            severity_values += (
                parameters[
                    "severity_add"
                ]
            )

            severity[
                has_accident
            ] = np.clip(
                severity_values,
                1.0,
                maximum_severity,
            )

            distance_scale = max(
                maximum_distance * 0.13,
                0.05,
            )

            distance_values = (
                random_generator.gamma(
                    shape=1.7,
                    scale=distance_scale,
                    size=nonzero_count,
                )
            )

            distance_values *= (
                parameters[
                    "distance_mult"
                ]
            )

            distance[
                has_accident
            ] = np.clip(
                distance_values,
                0.01,
                maximum_distance,
            )

        daily_cycle = (
            8.0
            * np.cos(
                2.0
                * np.pi
                * (
                    hour - 5
                )
                / 24.0
            )
        )

        slow_cycle = (
            4.5
            * np.sin(
                2.0
                * np.pi
                * timestamp.dayofyear
                / 11.0
            )
        )

        city_humidity = (
            humidity_baseline
            + daily_cycle
            + slow_cycle
        )

        humidity_noise = (
            random_generator.normal(
                loc=0.0,
                scale=1.2,
                size=number_of_nodes,
            )
        )

        humidity = (
            city_humidity
            + parameters[
                "humidity_add"
            ]
            + humidity_offsets
            + humidity_noise
        )

        humidity = (
            0.75 * humidity
            + 0.25
            * (
                smoother
                @ humidity
            )
        )

        humidity = np.clip(
            humidity,
            0.0,
            maximum_humidity,
        )

        hourly_values = (
            np.column_stack(
                [
                    accidents,
                    severity,
                    distance,
                    humidity,
                ]
            )
            .astype(np.float32)
        )

        raw_tensor[
            time_index
        ] = hourly_values

        hourly_frame = (
            regions.copy()
        )

        hourly_frame.insert(
            0,
            "timestamp",
            timestamp,
        )

        hourly_frame.insert(
            0,
            "city",
            city,
        )

        for (
            feature_index,
            feature_name,
        ) in enumerate(FEATURES):
            hourly_frame[
                feature_name
            ] = hourly_values[
                :,
                feature_index,
            ]

        hourly_frame[
            "scenario"
        ] = scenario

        hourly_frame[
            "is_synthetic"
        ] = True

        hourly_frame[
            "incident_focus_region"
        ] = focus_node

        generated_frames.append(
            hourly_frame
        )

    raw_records = pd.concat(
        generated_frames,
        ignore_index=True,
    )

    scaled_flat = scaler.transform(
        raw_tensor.reshape(
            -1,
            4,
        )
    )

    scaled_tensor = (
        scaled_flat
        .reshape(
            1,
            4,
            number_of_nodes,
            4,
        )
        .astype(np.float32)
    )

    return (
        raw_records,
        raw_tensor,
        scaled_tensor,
        history_hours,
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_synthetic_records(
    records: pd.DataFrame,
    scaled_tensor: np.ndarray,
    scaler,
) -> dict[str, bool]:
    """Validate the synthetic model input without displaying a table."""

    expected_rows = 4 * 100

    complete_region_counts = (
        records
        .groupby("timestamp")[
            "region_id"
        ]
        .nunique()
    )

    zero_accident_rows = (
        records[
            "Total_Accidents"
        ].eq(0)
    )

    fitted_minimum = np.asarray(
        scaler.data_min_,
        dtype=float,
    )

    fitted_maximum = np.asarray(
        scaler.data_max_,
        dtype=float,
    )

    feature_values = (
        records[
            FEATURES
        ]
        .to_numpy(
            dtype=float
        )
    )

    timestamps = pd.Series(
        sorted(
            records[
                "timestamp"
            ].unique()
        )
    )

    timestamp_differences = (
        timestamps
        .diff()
        .dropna()
    )

    return {
        "Required columns present": all(
            column in records.columns
            for column in FEATURES
        ),
        "Exactly 400 region-hour records": (
            len(records)
            == expected_rows
        ),
        "Four consecutive hourly timestamps": (
            records[
                "timestamp"
            ].nunique() == 4
            and timestamp_differences.eq(
                pd.Timedelta(hours=1)
            ).all()
        ),
        "Exactly 100 regions per hour": (
            complete_region_counts.eq(
                100
            ).all()
        ),
        "Region IDs are 0 through 99": (
            set(
                records[
                    "region_id"
                ].unique()
            )
            == set(range(100))
        ),
        "No duplicate timestamp-region rows": (
            not records.duplicated(
                [
                    "timestamp",
                    "region_id",
                ]
            ).any()
        ),
        "No missing or infinite feature values": (
            records[
                FEATURES
            ].notna().all().all()
            and np.isfinite(
                feature_values
            ).all()
        ),
        "Accident counts are nonnegative integers": (
            records[
                "Total_Accidents"
            ].ge(0).all()
            and np.allclose(
                records[
                    "Total_Accidents"
                ],
                np.round(
                    records[
                        "Total_Accidents"
                    ]
                ),
            )
        ),
        "Zero-accident semantic consistency": (
            records.loc[
                zero_accident_rows,
                "Mean_Severity",
            ].eq(0).all()
            and records.loc[
                zero_accident_rows,
                "Mean_Distance",
            ].eq(0).all()
        ),
        "Raw features inside scaler domain": (
            (
                feature_values
                >= fitted_minimum
                - 1e-7
            ).all()
            and (
                feature_values
                <= fitted_maximum
                + 1e-7
            ).all()
        ),
        "Correct final tensor shape": (
            scaled_tensor.shape
            == (
                1,
                4,
                100,
                4,
            )
        ),
        "Final tensor is finite": (
            np.isfinite(
                scaled_tensor
            ).all()
        ),
    }


# ============================================================
# MODEL DEFINITION AND LOADING
# ============================================================

def tensorflow_components():
    """Import TensorFlow only when fresh inference is requested."""

    try:
        import tensorflow as tf
        from tensorflow.keras import (
            Model,
            layers,
        )

    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is required for fresh "
            "ST-GNN inference but is not installed."
        ) from error

    return (
        tf,
        Model,
        layers,
    )


@st.cache_resource(
    show_spinner="Loading frozen ST-GNN model...",
)
def cached_stgnn_model(
    city_name: str,
    metadata_json: str,
    adjacency_bytes: bytes,
    adjacency_shape: tuple[int, int],
    weights_path_text: str,
    weights_modified_time: float,
):
    """Build the city-specific architecture and restore saved weights."""

    del city_name
    del weights_modified_time

    (
        tf,
        Model,
        layers,
    ) = tensorflow_components()

    metadata = json.loads(
        metadata_json
    )

    adjacency = np.frombuffer(
        adjacency_bytes,
        dtype=np.float32,
    ).reshape(
        adjacency_shape
    )

    class SpatioTemporalGCNLayer(
        layers.Layer
    ):
        def __init__(
            self,
            adj_matrix,
            units,
            **kwargs,
        ):
            super().__init__(
                **kwargs
            )

            self.adj_matrix_np = (
                np.asarray(
                    adj_matrix,
                    dtype=np.float32,
                )
            )

            self.units = int(
                units
            )

            self.dense = layers.Dense(
                self.units,
                activation="relu",
            )

        def build(
            self,
            input_shape,
        ):
            self.adj = tf.constant(
                self.adj_matrix_np,
                dtype=tf.float32,
            )

            super().build(
                input_shape
            )

        def call(
            self,
            inputs,
        ):
            graph_output = tf.einsum(
                "ij,btjf->btif",
                self.adj,
                inputs,
            )

            return self.dense(
                graph_output
            )

    sequence_length = int(
        metadata[
            "seq_len"
        ]
    )

    number_of_nodes = int(
        metadata[
            "champion_nodes"
        ]
    )

    number_of_features = len(
        metadata[
            "features"
        ]
    )

    model_input = layers.Input(
        shape=(
            sequence_length,
            number_of_nodes,
            number_of_features,
        ),
        name="Spatio_Temporal_Input",
    )

    graph_layer_1 = (
        SpatioTemporalGCNLayer(
            adjacency,
            32,
            name="GCN_1",
        )(
            model_input
        )
    )

    graph_layer_1 = (
        layers.BatchNormalization()(
            graph_layer_1
        )
    )

    graph_layer_1 = layers.Dropout(
        0.3
    )(
        graph_layer_1
    )

    graph_layer_2 = (
        SpatioTemporalGCNLayer(
            adjacency,
            32,
            name="GCN_2",
        )(
            graph_layer_1
        )
    )

    graph_layer_2 = (
        layers.BatchNormalization()(
            graph_layer_2
        )
    )

    graph_layer_2 = layers.Dropout(
        0.3
    )(
        graph_layer_2
    )

    temporal_input = layers.Reshape(
        (
            sequence_length,
            number_of_nodes * 32,
        )
    )(
        graph_layer_2
    )

    temporal_output = layers.GRU(
        64,
        name="Temporal_GRU",
    )(
        temporal_input
    )

    temporal_output = layers.Dropout(
        0.3
    )(
        temporal_output
    )

    model_output = layers.Dense(
        number_of_nodes,
        activation="sigmoid",
        name="Risk_Prediction",
    )(
        temporal_output
    )

    model = Model(
        model_input,
        model_output,
        name="STGNN_StaticKNN",
    )

    model(
        np.zeros(
            (
                1,
                sequence_length,
                number_of_nodes,
                number_of_features,
            ),
            dtype=np.float32,
        ),
        training=False,
    )

    model.load_weights(
        weights_path_text
    )

    return model


def predict_stgnn(
    model,
    scaled_tensor: np.ndarray,
) -> np.ndarray:
    """Execute fresh frozen ST-GNN inference."""

    predictions = model(
        np.asarray(
            scaled_tensor,
            dtype=np.float32,
        ),
        training=False,
    )

    scores = (
        np.asarray(
            predictions,
            dtype=np.float32,
        )
        .reshape(-1)
    )

    if scores.shape != (100,):
        raise ValueError(
            "Expected 100 node scores, received "
            f"shape {scores.shape}."
        )

    if (
        not np.isfinite(
            scores
        ).all()
        or np.any(
            scores < 0
        )
        or np.any(
            scores > 1
        )
    ):
        raise ValueError(
            "The model returned invalid scores."
        )

    return scores


# ============================================================
# SPATIAL MAP
# ============================================================

def create_spatial_map(
    node_results: pd.DataFrame,
    threshold: float,
) -> pdk.Deck:
    """Create the 100-node regional model-score map."""

    map_data = (
        node_results.copy()
    )

    map_data[
        "radius"
    ] = np.where(
        map_data[
            "predicted_flag"
        ],
        850,
        520,
    )

    map_data[
        "fill_color"
    ] = map_data[
        "predicted_probability"
    ].apply(
        lambda value: [
            int(
                40
                + 210 * value
            ),
            int(
                145
                - 95 * value
            ),
            int(
                225
                - 165 * value
            ),
            210,
        ]
    )

    map_data[
        "line_color"
    ] = map_data[
        "predicted_flag"
    ].apply(
        lambda flag: (
            [
                255,
                45,
                45,
                255,
            ]
            if flag
            else [
                220,
                230,
                245,
                180,
            ]
        )
    )

    map_data[
        "score_display"
    ] = map_data[
        "predicted_probability"
    ].map(
        lambda value: (
            f"{value:.4f}"
        )
    )

    map_data[
        "threshold_display"
    ] = f"{threshold:.4f}"

    map_data[
        "alert_display"
    ] = np.where(
        map_data[
            "predicted_flag"
        ],
        "Above threshold",
        "Below threshold",
    )

    map_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_data,
        get_position=(
            "[center_lng, center_lat]"
        ),
        get_radius="radius",
        get_fill_color="fill_color",
        get_line_color="line_color",
        line_width_min_pixels=1,
        stroked=True,
        filled=True,
        pickable=True,
        opacity=0.88,
    )

    view_state = pdk.ViewState(
        latitude=float(
            map_data[
                "center_lat"
            ].mean()
        ),
        longitude=float(
            map_data[
                "center_lng"
            ].mean()
        ),
        zoom=9.7,
        pitch=0,
    )

    tooltip = {
        "html": (
            "<b>Region:</b> {region_id}<br/>"
            "<b>Model score:</b> {score_display}<br/>"
            "<b>Threshold:</b> {threshold_display}<br/>"
            "<b>Status:</b> {alert_display}"
        )
    }

    return pdk.Deck(
        layers=[
            map_layer,
        ],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style=None,
    )


# ============================================================
# PAGE HEADER
# ============================================================

st.title(
    "Synthetic September Operations Demo"
)

st.caption(
    "End-to-end demonstration using deterministic "
    "synthetic September 2026 region-hour inputs, "
    "frozen preprocessing, and the saved "
    "city-specific ST-GNN."
)

st.info(
    "The inputs on this page are synthetic. The outputs "
    "are experimental STPC-proxy model scores for "
    "controlled demonstration scenarios, not observed "
    "crash outcomes and not verified causal "
    "secondary-crash predictions."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header(
        "September demo controls"
    )

    city = st.selectbox(
        "City",
        [
            "Chicago",
            "Miami",
        ],
        index=0,
    )

    scenario = st.selectbox(
        "Scenario",
        SCENARIOS,
        index=0,
    )

    selected_date = st.date_input(
        "September target date",
        value=DEFAULT_DATE,
        min_value=SEPTEMBER_START,
        max_value=SEPTEMBER_END,
    )

    selected_hour = st.time_input(
        "Target hour",
        value=DEFAULT_HOUR,
        step=3600,
    )

    run_inference = st.button(
        "Run September prediction",
        type="primary",
        use_container_width=True,
    )

    st.markdown(
        """
**Page contract**

- Synthetic September 2026 inputs
- Four preceding hours per target
- Frozen city-specific scaler
- Frozen 100-node ordering
- Fresh ST-GNN inference
- No synthetic ground-truth claim
        """
    )

    with st.expander(
        "Resolved project paths",
        expanded=False,
    ):
        st.write(
            "**Page file:**",
            str(PAGE_FILE),
        )

        st.write(
            "**Project root:**",
            str(PROJECT_ROOT),
        )


# ============================================================
# PACKAGE LOAD AND AUDIT
# ============================================================

folder, missing_files = audit_package(
    PROJECT_ROOT,
    city,
)

if missing_files:
    st.error(
        f"The {city} model package is incomplete: "
        + ", ".join(
            map(
                str,
                missing_files,
            )
        )
    )

    st.stop()


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
    ) = cached_city_data(
        str(PROJECT_ROOT),
        city,
    )

except Exception as error:
    st.error(
        f"Could not load the {city} model package."
    )

    st.exception(
        error
    )

    st.stop()


folder = Path(
    folder
)

adjacency = np.asarray(
    adjacency,
    dtype=np.float32,
)


package_validation_checks = validate_loaded(
    metadata,
    split_info,
    window_index,
    region_metadata,
    arrays,
    adjacency,
)

if not all(
    package_validation_checks.values()
):
    failed_package_checks = [
        check_name
        for (
            check_name,
            passed,
        ) in package_validation_checks.items()
        if not passed
    ]

    st.error(
        "The frozen artifact package failed validation. "
        "Inference has been blocked. Failed checks: "
        + "; ".join(
            failed_package_checks
        )
    )

    st.stop()


if list(
    metadata.get(
        "features",
        [],
    )
) != FEATURES:
    st.error(
        "The city package feature order does not "
        "match the required four-feature contract."
    )

    st.stop()


if (
    int(
        metadata.get(
            "seq_len",
            -1,
        )
    ) != 4
    or int(
        metadata.get(
            "champion_nodes",
            -1,
        )
    ) != 100
):
    st.error(
        "The selected city package does not match "
        "the required (4, 100, 4) model-input contract."
    )

    st.stop()


if adjacency.shape != (
    100,
    100,
):
    st.error(
        "Expected a (100, 100) adjacency matrix, "
        f"received {adjacency.shape}."
    )

    st.stop()


try:
    scaler_path = locate_file(
        folder,
        [
            "scaler.joblib",
        ],
    )

    scaler = cached_scaler(
        str(folder),
        scaler_path.stat().st_mtime,
    )

except Exception as error:
    st.error(
        "The fitted city-specific scaler could not "
        "be loaded."
    )

    st.exception(
        error
    )

    st.stop()


if int(
    getattr(
        scaler,
        "n_features_in_",
        -1,
    )
) != 4:
    st.error(
        "The fitted scaler does not contain exactly "
        "four features."
    )

    st.stop()


if hasattr(
    scaler,
    "feature_names_in_",
):
    if list(
        scaler.feature_names_in_
    ) != FEATURES:
        st.error(
            "The fitted scaler feature order does not "
            "match the model contract."
        )

        st.stop()


try:
    stgnn_threshold = threshold_from_metadata(
        metadata,
        thresholds,
    )

except Exception as error:
    st.error(
        "The ST-GNN validation threshold could not "
        "be loaded."
    )

    st.exception(
        error
    )

    st.stop()


# ============================================================
# GENERATE SELECTED WINDOW
# ============================================================

target_hour = combine_date_and_hour(
    selected_date,
    selected_hour,
)


try:
    (
        raw_records,
        raw_tensor,
        scaled_tensor,
        history_hours,
    ) = generate_four_hour_window(
        city=city,
        target_hour=target_hour,
        scenario=scenario,
        region_metadata=region_metadata,
        adjacency=adjacency,
        scaler=scaler,
    )

    synthetic_checks = (
        validate_synthetic_records(
            raw_records,
            scaled_tensor,
            scaler,
        )
    )

except Exception as error:
    st.error(
        "Synthetic window generation failed."
    )

    st.exception(
        error
    )

    st.stop()


# ============================================================
# SELECTED WINDOW SUMMARY
# ============================================================

st.subheader(
    "Selected synthetic window"
)

summary_columns = st.columns(
    6
)

summary_columns[0].metric(
    "City",
    city,
)

summary_columns[1].metric(
    "Scenario",
    scenario,
)

summary_columns[2].metric(
    "Target hour",
    target_hour.strftime(
        "%b %d, %H:00"
    ),
)

summary_columns[3].metric(
    "Spatial nodes",
    "100",
)

summary_columns[4].metric(
    "Input shape",
    "1 × 4 × 100 × 4",
)

summary_columns[5].metric(
    "ST-GNN threshold",
    f"{stgnn_threshold:.4f}",
)


history_table = pd.DataFrame(
    {
        "Sequence position": [
            "Hour 1",
            "Hour 2",
            "Hour 3",
            "Hour 4",
            "Target hour",
        ],
        "Timestamp": [
            format_datetime(
                timestamp
            )
            for timestamp in history_hours
        ]
        + [
            format_datetime(
                target_hour
            )
        ],
        "Role": (
            [
                "Model input",
            ]
            * 4
            + [
                "Prediction timestamp",
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
# SYNTHETIC INPUT SUMMARY
# ============================================================

st.subheader(
    "Synthetic input summary"
)

input_summary_columns = st.columns(
    5
)

input_summary_columns[0].metric(
    "Feature hours",
    raw_records[
        "timestamp"
    ].nunique(),
)

input_summary_columns[1].metric(
    "Region-hour records",
    f"{len(raw_records):,}",
)

input_summary_columns[2].metric(
    "Regions per hour",
    int(
        raw_records
        .groupby("timestamp")[
            "region_id"
        ]
        .nunique()
        .min()
    ),
)

input_summary_columns[3].metric(
    "Missing feature values",
    int(
        raw_records[
            FEATURES
        ]
        .isna()
        .sum()
        .sum()
    ),
)

input_summary_columns[4].metric(
    "Synthetic seed",
    SEED,
)


preview_columns = [
    "city",
    "timestamp",
    "region_id",
    "center_lat",
    "center_lng",
    *FEATURES,
    "scenario",
    "is_synthetic",
]

synthetic_input_csv = (
    raw_records[
        preview_columns
    ]
    .to_csv(
        index=False
    )
    .encode("utf-8")
)

st.download_button(
    "Download selected four-hour synthetic input CSV",
    data=synthetic_input_csv,
    file_name=(
        f"{city.lower()}_"
        f"{target_hour.strftime('%Y%m%d_%H00')}_"
        f"{scenario.lower().replace(' ', '_')}_"
        "synthetic_input.csv"
    ),
    mime="text/csv",
)


# ============================================================
# PREPROCESSING AND INFERENCE READINESS
# ============================================================

st.subheader(
    "Preprocessing and inference readiness"
)

passed_check_count = int(
    sum(
        bool(value)
        for value in synthetic_checks.values()
    )
)

total_check_count = len(
    synthetic_checks
)

all_synthetic_checks_passed = all(
    synthetic_checks.values()
)


readiness_columns = st.columns(
    3
)

readiness_columns[0].metric(
    "Validation checks passed",
    (
        f"{passed_check_count}/"
        f"{total_check_count}"
    ),
)

readiness_columns[1].metric(
    "Final tensor shape",
    "1 × 4 × 100 × 4",
)

readiness_columns[2].metric(
    "Inference readiness",
    (
        "Ready"
        if all_synthetic_checks_passed
        else "Blocked"
    ),
)


if not all_synthetic_checks_passed:
    failed_synthetic_checks = [
        check_name
        for (
            check_name,
            passed,
        ) in synthetic_checks.items()
        if not passed
    ]

    st.error(
        "Synthetic data validation failed. Saved-model "
        "inference is blocked. Failed checks: "
        + "; ".join(
            failed_synthetic_checks
        )
    )

    st.stop()


st.success(
    "All structural, temporal, semantic, feature-range, "
    "and tensor checks passed. The synthetic window is "
    "ready for frozen-model inference."
)


with st.expander(
    "View frozen preprocessing contract",
    expanded=False,
):
    preprocessing_contract = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Fitted minimum": np.asarray(
                scaler.data_min_,
                dtype=float,
            ),
            "Fitted maximum": np.asarray(
                scaler.data_max_,
                dtype=float,
            ),
            "Scale": np.asarray(
                scaler.scale_,
                dtype=float,
            ),
            "Offset": np.asarray(
                scaler.min_,
                dtype=float,
            ),
        }
    )

    st.dataframe(
        preprocessing_contract,
        hide_index=True,
        use_container_width=True,
    )

    st.code(
        (
            "scaled_value = "
            "raw_value * fitted_scale "
            "+ fitted_offset"
        ),
        language="text",
    )

    st.success(
        "The existing fitted scaler is applied using "
        "transform(). No scaler is fitted on synthetic data."
    )


# ============================================================
# FRESH SAVED-MODEL INFERENCE
# ============================================================

if "synthetic_last_request" not in st.session_state:
    st.session_state[
        "synthetic_last_request"
    ] = None


if "synthetic_scores" not in st.session_state:
    st.session_state[
        "synthetic_scores"
    ] = None


request_key = (
    city,
    scenario,
    str(target_hour),
)


if run_inference:
    try:
        weights_path = locate_file(
            folder,
            [
                "stgnn_seed42_final_restored.weights.h5",
                "stgnn.weights.h5",
                "stgnn_weights.h5",
            ],
        )

        model = cached_stgnn_model(
            city_name=city,
            metadata_json=json.dumps(
                metadata,
                sort_keys=True,
                default=str,
            ),
            adjacency_bytes=np.ascontiguousarray(
                adjacency,
                dtype=np.float32,
            ).tobytes(),
            adjacency_shape=tuple(
                adjacency.shape
            ),
            weights_path_text=str(
                weights_path
            ),
            weights_modified_time=(
                weights_path
                .stat()
                .st_mtime
            ),
        )

        with st.spinner(
            "Running fresh saved-model inference..."
        ):
            predicted_scores = predict_stgnn(
                model,
                scaled_tensor,
            )

        st.session_state[
            "synthetic_scores"
        ] = predicted_scores

        st.session_state[
            "synthetic_last_request"
        ] = request_key

    except Exception as error:
        st.error(
            "Fresh ST-GNN inference failed. "
            "No synthetic predictions were fabricated."
        )

        st.exception(
            error
        )

        st.stop()


if (
    st.session_state[
        "synthetic_last_request"
    ]
    != request_key
):
    st.info(
        "Choose the controls and click "
        "**Run September prediction** to execute "
        "fresh saved-model inference."
    )

    st.stop()


predicted_scores = np.asarray(
    st.session_state[
        "synthetic_scores"
    ],
    dtype=np.float32,
)


node_results = (
    region_metadata
    .copy()
    .sort_values("region_id")
    .reset_index(drop=True)
)

node_results[
    "predicted_probability"
] = predicted_scores

node_results[
    "predicted_flag"
] = (
    predicted_scores
    >= stgnn_threshold
)

node_results[
    "threshold"
] = stgnn_threshold

node_results[
    "target_hour"
] = target_hour

node_results[
    "scenario"
] = scenario


latest_features = raw_records.loc[
    raw_records[
        "timestamp"
    ].eq(
        history_hours[-1]
    ),
    [
        "region_id",
        *FEATURES,
    ],
]


node_results = node_results.merge(
    latest_features,
    on="region_id",
    how="left",
    validate="one_to_one",
)


node_results = (
    node_results
    .sort_values(
        "predicted_probability",
        ascending=False,
    )
    .reset_index(drop=True)
)


# ============================================================
# RESULTS
# ============================================================

st.subheader(
    "Saved ST-GNN prediction results"
)


result_columns = st.columns(
    5
)

result_columns[0].metric(
    "Maximum model score",
    (
        f"{node_results['predicted_probability'].max():.4f}"
    ),
)

result_columns[1].metric(
    "Mean node score",
    (
        f"{node_results['predicted_probability'].mean():.4f}"
    ),
)

result_columns[2].metric(
    "Median node score",
    (
        f"{node_results['predicted_probability'].median():.4f}"
    ),
)

result_columns[3].metric(
    "Regions above threshold",
    int(
        node_results[
            "predicted_flag"
        ].sum()
    ),
)

result_columns[4].metric(
    "Highest-score region",
    int(
        node_results.iloc[
            0
        ][
            "region_id"
        ]
    ),
)


# ============================================================
# HIGHEST-SCORE TABLE AND SPATIAL MAP
# ============================================================

left_column, right_column = st.columns(
    [
        1.05,
        1.45,
    ]
)


with left_column:
    st.markdown(
        "#### Highest-score regions"
    )

    st.dataframe(
        node_results[
            [
                "region_id",
                "predicted_probability",
                "predicted_flag",
                *FEATURES,
            ]
        ].head(12),
        hide_index=True,
        use_container_width=True,
        column_config={
            "region_id":
                st.column_config.NumberColumn(
                    "Region ID",
                    format="%d",
                ),
            "predicted_probability":
                st.column_config.NumberColumn(
                    "Model score",
                    format="%.4f",
                ),
            "predicted_flag":
                st.column_config.CheckboxColumn(
                    "Above threshold",
                ),
            "Total_Accidents":
                st.column_config.NumberColumn(
                    "Accidents",
                    format="%d",
                ),
            "Mean_Severity":
                st.column_config.NumberColumn(
                    "Mean severity",
                    format="%.2f",
                ),
            "Mean_Distance":
                st.column_config.NumberColumn(
                    "Mean distance",
                    format="%.2f",
                ),
            "Mean_Humidity":
                st.column_config.NumberColumn(
                    "Mean humidity",
                    format="%.1f",
                ),
        },
    )


with right_column:
    st.markdown(
        "#### Spatial distribution"
    )

    st.pydeck_chart(
        create_spatial_map(
            node_results,
            stgnn_threshold,
        ),
        use_container_width=True,
    )

    st.caption(
        "Red outlines indicate validation-threshold "
        "exceedance. Marker colour represents the "
        "continuous ST-GNN model score."
    )


# ============================================================
# MODEL-SCORE DISTRIBUTION
# ============================================================

st.markdown(
    "#### Distribution of regional model scores"
)


score_chart_data = node_results[
    [
        "region_id",
        "predicted_probability",
    ]
].copy()


score_chart_data[
    "predicted_probability"
] = pd.to_numeric(
    score_chart_data[
        "predicted_probability"
    ],
    errors="coerce",
)


score_chart_data = score_chart_data.dropna(
    subset=[
        "predicted_probability",
    ]
)


score_chart_data[
    "predicted_probability"
] = score_chart_data[
    "predicted_probability"
].clip(
    lower=0.0,
    upper=1.0,
)


score_histogram = (
    alt.Chart(
        score_chart_data
    )
    .mark_bar(
        color="#79BDF2",
        opacity=0.90,
        cornerRadiusTopLeft=4,
        cornerRadiusTopRight=4,
    )
    .encode(
        x=alt.X(
            "predicted_probability:Q",
            bin=alt.Bin(
                step=0.05,
                extent=[
                    0.0,
                    1.0,
                ],
            ),
            title="ST-GNN model score",
            scale=alt.Scale(
                domain=[
                    0.0,
                    1.0,
                ]
            ),
            axis=alt.Axis(
                format=".2f",
                labelAngle=0,
                values=[
                    0.0,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    1.0,
                ],
                grid=True,
            ),
        ),
        y=alt.Y(
            "count():Q",
            title="Number of regions",
            axis=alt.Axis(
                tickMinStep=1,
                grid=True,
            ),
        ),
        tooltip=[
            alt.Tooltip(
                "predicted_probability:Q",
                bin=alt.Bin(
                    step=0.05,
                    extent=[
                        0.0,
                        1.0,
                    ],
                ),
                title="Model score range",
            ),
            alt.Tooltip(
                "count():Q",
                title="Number of regions",
                format="d",
            ),
        ],
    )
    .properties(
        height=340,
    )
)


threshold_chart_data = pd.DataFrame(
    {
        "threshold": [
            float(
                stgnn_threshold
            ),
        ],
        "threshold_label": [
            (
                "Alert threshold: "
                f"{stgnn_threshold:.4f}"
            )
        ],
    }
)


threshold_line = (
    alt.Chart(
        threshold_chart_data
    )
    .mark_rule(
        color="#FF4B4B",
        strokeWidth=3,
        strokeDash=[
            7,
            5,
        ],
    )
    .encode(
        x=alt.X(
            "threshold:Q",
            scale=alt.Scale(
                domain=[
                    0.0,
                    1.0,
                ]
            ),
        ),
        tooltip=[
            alt.Tooltip(
                "threshold:Q",
                title="Alert threshold",
                format=".4f",
            ),
        ],
    )
)


threshold_label = (
    alt.Chart(
        threshold_chart_data
    )
    .mark_text(
        color="#FF6B6B",
        align="left",
        baseline="top",
        dx=7,
        dy=8,
        fontSize=13,
        fontWeight="bold",
    )
    .encode(
        x=alt.X(
            "threshold:Q",
            scale=alt.Scale(
                domain=[
                    0.0,
                    1.0,
                ]
            ),
        ),
        y=alt.value(5),
        text="threshold_label:N",
    )
)


st.altair_chart(
    (
        score_histogram
        + threshold_line
        + threshold_label
    ),
    use_container_width=True,
)


st.caption(
    "The x-axis shows ST-GNN model scores from 0.00 "
    "to 1.00. Each bar shows how many of the 100 "
    "regions received a score within that range. "
    "The red dashed line shows the validation-derived "
    "alert threshold. These values are experimental "
    "model scores, not guaranteed real-world crash "
    "probabilities."
)


below_threshold_count = int(
    (
        node_results[
            "predicted_probability"
        ]
        < stgnn_threshold
    ).sum()
)

above_threshold_count = int(
    (
        node_results[
            "predicted_probability"
        ]
        >= stgnn_threshold
    ).sum()
)


interpretation_columns = st.columns(
    2
)

interpretation_columns[0].metric(
    "Regions below alert threshold",
    below_threshold_count,
)

interpretation_columns[1].metric(
    "Regions at or above alert threshold",
    above_threshold_count,
)


if above_threshold_count > 0:
    st.warning(
        f"{above_threshold_count} of the 100 regions "
        f"received model scores at or above the "
        f"validation-derived threshold of "
        f"{stgnn_threshold:.4f}. This represents "
        f"model alert status, not a confirmed crash event."
    )

else:
    st.info(
        "No regions crossed the validation-derived "
        "alert threshold for the selected synthetic "
        "scenario and target hour."
    )


# ============================================================
# EXPORT AND LIMITATIONS
# ============================================================

export_columns = [
    "target_hour",
    "scenario",
    "region_id",
    "center_lat",
    "center_lng",
    "predicted_probability",
    "threshold",
    "predicted_flag",
    *FEATURES,
]


result_csv = (
    node_results[
        export_columns
    ]
    .to_csv(
        index=False
    )
    .encode("utf-8")
)


st.download_button(
    "Download node-level synthetic prediction results",
    data=result_csv,
    file_name=(
        f"{city.lower()}_"
        f"{target_hour.strftime('%Y%m%d_%H00')}_"
        f"{scenario.lower().replace(' ', '_')}_"
        "stgnn_predictions.csv"
    ),
    mime="text/csv",
)


with st.expander(
    "Scientific interpretation and limitations",
    expanded=False,
):
    st.markdown(
        """
- Input records are deterministic synthetic scenario data, not observed September events.
- Scores are experimental STPC-proxy model outputs, not verified causal secondary-crash probabilities.
- A threshold exceedance is a model alert, not proof that a crash will occur.
- This page does not use or create synthetic ground-truth labels.
- Accuracy, precision, recall, F1, and calibration are not estimated from this synthetic scenario.
- The fitted city-specific scaler, frozen region ordering, adjacency matrix, saved model weights, and validation-derived threshold are reused without retraining.
- Controlled scenario responses do not establish causation.
        """
    )


st.divider()

st.caption(
    "TrafficRisk-Insight | "
    "Synthetic September operations demo | "
    "Offline research prototype"
)