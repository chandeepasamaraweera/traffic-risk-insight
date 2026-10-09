from pathlib import Path
import json
import numpy as np
import pandas as pd

CITY_FILES = {
    "Chicago": {"folder": "chicago", "prefix": "chicago"},
    "Miami": {"folder": "miami", "prefix": "miami"},
}

REQUIRED = [
    "metadata.json", "thresholds.json", "split_info.json", "adjacency_norm.npy",
    "X_test.npy", "y_test.npy", "stgnn_test_scores.npy", "lstm_test_scores.npy",
    "rf_test_scores.npy", "lr_test_scores.npy", "stgnn_seed42_final_restored.weights.h5",
]

def city_dir(project_root: Path, city: str) -> Path:
    return project_root / "models" / CITY_FILES[city]["folder"]

def audit_package(project_root: Path, city: str):
    folder = city_dir(project_root, city)
    prefix = CITY_FILES[city]["prefix"]
    required = REQUIRED + [f"{prefix}_window_index.csv", f"{prefix}_region_metadata.csv"]
    missing = [name for name in required if not (folder/name).exists()]
    return folder, missing

def load_json(path):
    with open(path, encoding="utf-8") as f: return json.load(f)

def load_city_data(project_root: Path, city: str):
    folder, missing = audit_package(project_root, city)
    if missing: raise FileNotFoundError("Missing artifacts: " + ", ".join(missing))
    prefix = CITY_FILES[city]["prefix"]
    metadata = load_json(folder/"metadata.json")
    thresholds = load_json(folder/"thresholds.json")
    split_info = load_json(folder/"split_info.json")
    windows = pd.read_csv(folder/f"{prefix}_window_index.csv", parse_dates=["target_hour", "history_hour_1", "history_hour_2", "history_hour_3", "history_hour_4"])
    regions = pd.read_csv(folder/f"{prefix}_region_metadata.csv")
    arrays = {name: np.load(folder/name, mmap_mode="r", allow_pickle=False) for name in [
        "X_test.npy", "y_test.npy", "stgnn_test_scores.npy", "lstm_test_scores.npy", "rf_test_scores.npy", "lr_test_scores.npy"
    ]}
    adjacency = np.load(folder/"adjacency_norm.npy", allow_pickle=False).astype(np.float32)
    return folder, metadata, thresholds, split_info, windows, regions, arrays, adjacency

def validate_loaded(metadata, split_info, windows, regions, arrays, adjacency):
    n = int(metadata["champion_nodes"])
    expected_test = int(split_info["test_samples"])
    tests = windows.loc[windows["split"].eq("test")]
    checks = {
        "test window count": len(tests) == expected_test,
        "X_test shape": tuple(arrays["X_test.npy"].shape) == (expected_test, metadata["seq_len"], n, len(metadata["features"])),
        "y_test shape": tuple(arrays["y_test.npy"].shape) == (expected_test, n),
        "adjacency shape": tuple(adjacency.shape) == (n, n),
        "region IDs": set(regions["region_id"].astype(int)) == set(range(n)),
    }
    for score in ["stgnn_test_scores.npy", "lstm_test_scores.npy", "rf_test_scores.npy", "lr_test_scores.npy"]:
        checks[score] = tuple(arrays[score].shape) == (expected_test, n)
    return checks
