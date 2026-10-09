from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_csv(path):
    """
    Load a CSV relative to the traffic-risk-system project root.

    Example:
        load_csv("data/benchmark/benchmark.csv")
    """
    csv_path = Path(path)

    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path

    csv_path = csv_path.resolve()

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Data file not found: {csv_path}\n"
            f"Project root: {PROJECT_ROOT}\n"
            f"Current working directory: {Path.cwd()}"
        )

    return pd.read_csv(csv_path)