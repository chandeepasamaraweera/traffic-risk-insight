from pathlib import Path
import re

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Spatial Morphology Viewer",
    page_icon="MAP",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGE_FILE = Path(__file__).resolve()
PROJECT_ROOT = PAGE_FILE.parents[2]
MAPS_DIR = PROJECT_ROOT / "maps"

MAP_CONFIG = {
    "Chicago": {
        "filename": "Chicago.html",
        "primary_triggers": 20,
        "stpc_events": 32,
    },
    "Los Angeles": {
        "filename": "Los_Angeles.html",
        "primary_triggers": 83,
        "stpc_events": 174,
    },
    "Miami": {
        "filename": "Miami.html",
        "primary_triggers": 94,
        "stpc_events": 472,
    },
    "Indianapolis": {
        "filename": "Indianapolis.html",
        "primary_triggers": None,
        "stpc_events": None,
    },
    "Iowa C2": {
        "filename": "Iowa_C2.html",
        "primary_triggers": None,
        "stpc_events": None,
    },
}


def patch_folium_html(html_data: str) -> str:
    if not html_data.strip():
        raise ValueError("The selected HTML map file is empty.")

    meta_tag = (
        '<meta name="referrer" '
        'content="strict-origin-when-cross-origin">'
    )

    if '<meta name="referrer"' not in html_data.lower():
        head_match = re.search(
            r"<head[^>]*>",
            html_data,
            flags=re.IGNORECASE,
        )
        if head_match is None:
            raise ValueError("The map HTML has no head element.")
        position = head_match.end()
        html_data = (
            html_data[:position]
            + "\n    "
            + meta_tag
            + html_data[position:]
        )

    official_url = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    old_urls = [
        "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "https://{s}.tile.osm.org/{z}/{x}/{y}.png",
        "http://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "http://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "http://{s}.tile.osm.org/{z}/{x}/{y}.png",
    ]
    for old_url in old_urls:
        html_data = html_data.replace(old_url, official_url)

    if '"referrerPolicy"' not in html_data:
        html_data = html_data.replace(
            '"minZoom": 0,',
            '"referrerPolicy": '
            '"strict-origin-when-cross-origin",\n'
            '  "minZoom": 0,',
        )

    return html_data


@st.cache_data(show_spinner=False)
def load_and_patch_map(map_path_text: str, modified_time: float) -> str:
    _ = modified_time
    map_path = Path(map_path_text)
    html_data = map_path.read_text(encoding="utf-8")
    return patch_folium_html(html_data)


st.title("Spatial Morphology Viewer")
st.caption(
    "Interactive inspection of empirical event morphology under "
    "the selected city-specific STPC configuration."
)
st.info(
    "This is a retrospective research visualization. It is not "
    "a live traffic map and does not establish crash causality."
)

available_cities = [
    city
    for city, config in MAP_CONFIG.items()
    if (MAPS_DIR / config["filename"]).is_file()
]

missing_cities = [
    city
    for city, config in MAP_CONFIG.items()
    if not (MAPS_DIR / config["filename"]).is_file()
]

if not available_cities:
    st.error("No map HTML files were found.")
    st.write("Expected maps directory:")
    st.code(str(MAPS_DIR), language=None)
    st.stop()

city = st.selectbox("Select city", available_cities)
config = MAP_CONFIG[city]
map_path = MAPS_DIR / config["filename"]

if not map_path.is_file():
    st.error("The selected map file was not found.")
    st.code(str(map_path), language=None)
    st.stop()

if map_path.stat().st_size == 0:
    st.error("The selected map file is empty.")
    st.stop()

try:
    patched_html = load_and_patch_map(
        str(map_path),
        map_path.stat().st_mtime,
    )
except Exception as error:
    st.error("The selected map could not be loaded.")
    st.exception(error)
    st.stop()

components.html(
    patched_html,
    height=720,
    scrolling=False,
)

primary = config["primary_triggers"]
stpc = config["stpc_events"]

if primary is not None and stpc is not None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Primary triggers", f"{primary:,}")
    with col2:
        st.metric("STPC-proximate events", f"{stpc:,}")
    with col3:
        st.metric("Displayed events", f"{primary + stpc:,}")

with st.expander("Map diagnostics", expanded=False):
    st.write("Project root:", str(PROJECT_ROOT))
    st.write("Maps directory:", str(MAPS_DIR))
    st.write("Selected map:", str(map_path))
    st.write("OSM referrer policy: strict-origin-when-cross-origin")
    if missing_cities:
        st.warning(
            "Missing configured map files: "
            + ", ".join(missing_cities)
        )

st.warning(
    "STPC-labelled proximate events are not verified causal "
    "secondary crashes."
)

with st.expander(
    "Scientific interpretation and limitations",
    expanded=False,
):
    st.markdown(
        """
- Blue markers represent primary or isolated trigger events.
- Red markers represent chronology-aware STPC-proximate events.
- Geographic and temporal proximity does not establish causality.
- The displayed day is selected for qualitative inspection.
- The graph is event-derived and is not a true road network.
- The application does not ingest a live traffic feed.
- The visualization is not an operational warning system.
        """
    )

st.divider()
st.caption(
    "TrafficRisk-Insight | Spatial morphology | "
    "Offline research prototype"
)
