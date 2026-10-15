# src/mlp.py
import torch.nn as nn

def build_mlp(in_dim, num_classes):
    return nn.Sequential(
        nn.Linear(in_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(1024, num_classes))