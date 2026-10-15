# src/mlp.py

import torch
import torch.nn as nn

class DeepMLP(nn.Module):
    """
    Mạng Perceptron Đa Lớp Sâu (3 Hidden Layers) tối ưu cho bài toán 150 class:
    - Kích thước tầng giảm dần: in_dim -> 2048 -> 1024 -> 512 -> num_classes
    - Kích hoạt GELU (mịn hơn ReLU, bảo toàn gradient tốt hơn)
    - BatchNorm1d giúp ổn định phân phối đặc trưng giữa các tầng
    - Dropout bậc thang (0.3 -> 0.25 -> 0.2) chống overfitting
    """
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Dropout(0.30),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.25),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.20),

            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_legacy_mlp(in_dim: int, num_classes: int):
    """Kiến trúc MLP 1 tầng ẩn tương thích với các model cũ đã lưu."""
    return nn.Sequential(
        nn.Linear(in_dim, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(1024, num_classes)
    )


def build_mlp(in_dim: int, num_classes: int, deep: bool = True):
    """
    Factory function xây dựng mô hình MLP.
    - deep=True: Mô hình DeepMLP 3 tầng mới tối ưu cho 150 class.
    - deep=False: Mô hình 1 tầng tương thích checkpoint cũ.
    """
    if deep:
        return DeepMLP(in_dim, num_classes)
    return build_legacy_mlp(in_dim, num_classes)