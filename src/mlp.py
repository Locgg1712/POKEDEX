# src/mlp.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering


class NormedLinear(nn.Module):
    """
    Cosine Similarity Classifier (ProtoNet / ArcFace-style head):
    Chuẩn hóa L2 cả input features lẫn weight vectors trước khi nhân ma trận,
    sau đó nhân với nhiệt độ học được (learnable temperature τ > 0).
    Giúp tầng 512 học biểu diễn metric tốt hơn, tăng khoảng cách giữa các lớp
    tương đồng về hình thái (ví dụ Pidgey vs Pidgeotto, Bulbasaur vs Ivysaur).
    """
    def __init__(self, in_features: int, out_features: int, init_tau: float = 10.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight)
        # Temperature τ được học — bắt đầu từ init_tau, luôn dương nhờ softplus
        self._log_tau = nn.Parameter(torch.tensor(float(init_tau)).log())

    @property
    def tau(self):
        return F.softplus(self._log_tau) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, p=2, dim=-1)
        w_norm = F.normalize(self.weight, p=2, dim=-1)
        return self.tau * F.linear(x_norm, w_norm)


class DeepMLP(nn.Module):
    """
    Mạng Perceptron Đa Lớp Sâu tích hợp Nhánh Tự Nhận Định Tộc/Họ (Self-Discovered Taxonomy):
    - Nếu n_clusters == 0: Hoạt động như DeepMLP 3 tầng tiêu chuẩn (tương thích ngược 100%).
    - Nếu n_clusters > 0:
        + cluster_head: Dự đoán phân phối xác suất thuộc về K cụm hình thái - màu sắc tự sinh.
        + species_head: Kết hợp biểu diễn 512 chiều với xác suất cụm để phân loại 150 loài.
    - use_cosine_head=True: Thay Linear cuối bằng NormedLinear (Cosine Classifier / Metric Learning).
    """
    def __init__(self, in_dim: int, num_classes: int, n_clusters: int = 0, use_cosine_head: bool = False):
        super().__init__()
        self.n_clusters = n_clusters
        self.use_cosine_head = use_cosine_head
        _Head = NormedLinear if use_cosine_head else nn.Linear

        self.backbone = nn.Sequential(
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
        )

        if n_clusters > 0:
            self.cluster_head = nn.Linear(512, n_clusters)
            self.species_head = _Head(512 + n_clusters, num_classes)
        else:
            self.fc = _Head(512, num_classes)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Tự động tương thích với các checkpoint cũ lưu dưới dạng self.net."""
        remapped = {}
        keys_to_del = []
        for k, v in state_dict.items():
            if k.startswith(prefix + "net."):
                sub_k = k[len(prefix + "net."):]
                parts = sub_k.split(".", 1)
                idx = int(parts[0])
                rest = parts[1]
                if idx == 12:
                    new_k = prefix + f"fc.{rest}"
                else:
                    new_k = prefix + f"backbone.{idx}.{rest}"
                remapped[new_k] = v
                keys_to_del.append(k)

        for k in keys_to_del:
            del state_dict[k]
        state_dict.update(remapped)

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, x: torch.Tensor, return_cluster: bool = False):
        feat = self.backbone(x)
        if self.n_clusters > 0:
            cluster_logits = self.cluster_head(feat)
            cluster_probs = torch.softmax(cluster_logits, dim=-1)
            combined = torch.cat([feat, cluster_probs], dim=-1)
            species_logits = self.species_head(combined)
            if return_cluster:
                return species_logits, cluster_logits
            return species_logits
        return self.fc(feat)


def build_legacy_mlp(in_dim: int, num_classes: int):
    """Kiến trúc MLP 1 tầng ẩn tương thích với các model cũ đã lưu."""
    return nn.Sequential(
        nn.Linear(in_dim, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(1024, num_classes)
    )


def build_mlp(
    in_dim: int, num_classes: int, deep: bool = True,
    n_clusters: int = 0, use_cosine_head: bool = False
):
    if deep:
        return DeepMLP(in_dim, num_classes, n_clusters=n_clusters, use_cosine_head=use_cosine_head)
    return build_legacy_mlp(in_dim, num_classes)


# ==============================================================================
# UNSUPERVISED VISUAL TAXONOMY (TỰ ĐỘNG NHẬN ĐỊNH TỘC/HỌ TỪ ĐẶC TRƯNG DSP)
# ==============================================================================

def discover_visual_clusters(X_train_weighted: np.ndarray, y_train: np.ndarray, num_classes: int, n_clusters: int = 24):
    """
    Tự động phân cụm 150 loài Pokémon thành `n_clusters` Tộc/Họ dựa thuần túy trên
    tín hiệu DSP (Hình dáng HOG/Fourier + Kết cấu Gabor/GLCM + Màu sắc HSV/Lab):
    1. Tính vector trọng tâm (centroid) chuẩn hóa L2 cho từng loài trên tập Train.
    2. Chạy phân cụm phân cấp AgglomerativeClustering (Ward linkage).
    3. Trả về mảng ánh xạ `class_to_cluster` kích thước (num_classes,).
    """
    n_clusters = min(n_clusters, max(2, num_classes // 2))
    centroids = np.zeros((num_classes, X_train_weighted.shape[1]), dtype=np.float32)

    for c in range(num_classes):
        mask = (y_train == c)
        if np.any(mask):
            mean_vec = X_train_weighted[mask].mean(axis=0)
            norm = np.linalg.norm(mean_vec)
            centroids[c] = mean_vec / (norm + 1e-8)

    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="ward")
    class_to_cluster = clustering.fit_predict(centroids).astype(np.int64)
    return class_to_cluster


def taxonomy_rerank(species_probs: np.ndarray, cluster_probs: np.ndarray, class_to_cluster: np.ndarray, beta: float = 0.25):
    """
    Suy luận xác suất hợp (Hierarchical Log-Linear Pooling):
        P_final(m) ∝ P_species(m)^(1 - beta) * P_cluster(Cluster(m))^beta
    Không dùng ngưỡng cứng if/else, giúp đẩy mượt mà các loài thuộc đúng Tộc hình thái - màu sắc lên Top-1.
    """
    eps = 1e-8
    c_idx = np.asarray(class_to_cluster, dtype=np.int64)

    if species_probs.ndim == 1:
        prior = cluster_probs[c_idx]
        combined = np.power(np.clip(species_probs, eps, 1.0), 1.0 - beta) * np.power(np.clip(prior, eps, 1.0), beta)
        return combined / np.sum(combined)

    prior = cluster_probs[:, c_idx]
    combined = np.power(np.clip(species_probs, eps, 1.0), 1.0 - beta) * np.power(np.clip(prior, eps, 1.0), beta)
    return combined / np.sum(combined, axis=1, keepdims=True)