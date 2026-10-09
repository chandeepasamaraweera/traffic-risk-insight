from pathlib import Path


# paths.py is located at:
# traffic-risk-system/app/utils/paths.py
#
# parents[0] = app/utils
# parents[1] = app
# parents[2] = traffic-risk-system
PROJECT_ROOT = Path(__file__).resolve().parents[2]

APP_DIR = PROJECT_ROOT / "app"
DATA_DIR = PROJECT_ROOT / "data"
MAPS_DIR = PROJECT_ROOT / "maps"
MODELS_DIR = PROJECT_ROOT / "models"


MAUP_CSV = DATA_DIR / "maup" / "maup.csv"
BENCHMARK_CSV = DATA_DIR / "benchmark" / "benchmark.csv"
TOPOLOGY_SEARCH_CSV = (
    DATA_DIR / "topology_search" / "topology_search.csv"
)
ABLATION_CSV = DATA_DIR / "ablation" / "ablation.csv"