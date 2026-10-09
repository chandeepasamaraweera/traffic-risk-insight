import numpy as np
from .model_architecture import build_stgnn_model

def create_and_load_stgnn(metadata, adjacency, weights_path):
    model = build_stgnn_model(
        seq_len=int(metadata["seq_len"]), n_nodes=int(metadata["champion_nodes"]),
        n_features=len(metadata["features"]), adj_matrix=adjacency,
    )
    model.load_weights(weights_path)
    return model

def predict_one(model, x):
    return model.predict(np.asarray(x, dtype=np.float32), verbose=0)[0].astype(np.float32)

def verification_stats(fresh, saved):
    difference = np.abs(np.asarray(fresh) - np.asarray(saved))
    return {"maximum_absolute_difference": float(difference.max()), "mean_absolute_difference": float(difference.mean()), "all_finite": bool(np.isfinite(fresh).all())}
