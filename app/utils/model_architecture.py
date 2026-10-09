import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers

class SpatioTemporalGCNLayer(layers.Layer):
    def __init__(self, adj_matrix, units, **kwargs):
        super().__init__(**kwargs)
        self.adj_matrix_np = np.asarray(adj_matrix, dtype=np.float32)
        self.units = units
        self.dense = layers.Dense(units, activation="relu")

    def build(self, input_shape):
        self.adj = tf.constant(self.adj_matrix_np, dtype=tf.float32)
        super().build(input_shape)

    def call(self, inputs):
        x = tf.einsum("ij,btjf->btif", self.adj, inputs)
        return self.dense(x)


def build_stgnn_model(seq_len, n_nodes, n_features, adj_matrix):
    inp = layers.Input((seq_len, n_nodes, n_features), name="Spatio_Temporal_Input")
    g1 = SpatioTemporalGCNLayer(adj_matrix, 32, name="GCN_1")(inp)
    g1 = layers.BatchNormalization()(g1)
    g1 = layers.Dropout(0.3)(g1)
    g2 = SpatioTemporalGCNLayer(adj_matrix, 32, name="GCN_2")(g1)
    g2 = layers.BatchNormalization()(g2)
    g2 = layers.Dropout(0.3)(g2)
    x = layers.Reshape((seq_len, n_nodes * 32))(g2)
    x = layers.GRU(64, name="Temporal_GRU")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_nodes, activation="sigmoid", name="Risk_Prediction")(x)
    return Model(inp, out, name="STGNN_StaticKNN")
