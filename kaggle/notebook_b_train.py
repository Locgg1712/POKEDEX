# %% [markdown]
# Notebook B - Huấn luyện mô hình, Thử nghiệm bóc tách (Ablation Study)
# & Phân loại nhận thức tiến hóa (Evolution-Aware Classification).
# Chạy trên môi trường Kaggle (CPU hoặc GPU).

# %%
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(os.environ.get("POKEDEX_ROOT", Path.cwd()))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import FEATURE_GROUPS, FEATURE_DIM
from src.mlp import build_mlp
from src.preprocess import IMAGE_SIZE
from src.evolution import evolution_rerank, get_pokemon_family

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURES_PATH = Path(os.environ.get("POKEDEX_FEATURES", ROOT / "features.npz"))
MODEL_DIR = Path(os.environ.get("POKEDEX_MODEL", ROOT / "Model"))
RESULTS_DIR = Path(os.environ.get("POKEDEX_RESULTS", ROOT / "results"))

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# %%
print(f"Loading features from {FEATURES_PATH}...")
data = np.load(FEATURES_PATH, allow_pickle=False)
X = data["X"].astype(np.float32)
y = data["y"].astype(np.int64)
groups = data["groups"].astype(str)
names = data["names"].astype(str)

if "group_columns" in data:
    saved_groups = json.loads(str(data["group_columns"].item()))
else:
    saved_groups = FEATURE_GROUPS

print(f"Samples: {len(X)}, Dim: {X.shape[1]}, Classes: {len(names)}, Unique Groups: {len(np.unique(groups))}")

# %%
# Chia dữ liệu StratifiedGroupKFold đảm bảo không leak ảnh augment
splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
train_idx, val_idx = next(splitter.split(X, y, groups))

assert set(groups[train_idx]).isdisjoint(set(groups[val_idx])), "Data leakage detected between train and val groups!"
print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")

labels_dict = {i: str(names[i]) for i in range(len(names))}

# %%
def cols(*keys):
    """Lấy danh sách các cột đặc trưng theo tên nhóm."""
    selected = []
    for key in keys:
        if key in saved_groups:
            start, end = saved_groups[key]
            selected.extend(range(int(start), int(end)))
        else:
            print(f"[Cảnh báo] Nhóm đặc trưng '{key}' không có trong saved_groups")
    return np.asarray(selected, dtype=np.int64)


def train_mlp_seed(X_train, y_train, X_val, y_val, num_classes, seed, epochs=80, deep=True):
    """Huấn luyện 1 mạng MLP với seed cố định."""
    torch.manual_seed(seed)
    model = build_mlp(X_train.shape[1], num_classes, deep=deep).to(DEVICE)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=256,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_tensor = torch.from_numpy(X_val).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, epochs=epochs, steps_per_epoch=len(train_loader)
    )
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

    best_acc, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_tensor).argmax(dim=1).cpu().numpy()
        acc = accuracy_score(y_val, val_pred)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_acc


def evaluate_predictions(label, y_true, prob, class_names, apply_rerank=False):
    """Đánh giá toàn diện: Top-1, Top-3, Top-5, Macro-F1."""
    if apply_rerank:
        reranked_probs = []
        for p in prob:
            reranked_probs.append(evolution_rerank(p, labels_dict, alpha=0.35))
        prob = np.array(reranked_probs)

    pred = prob.argmax(axis=1)
    order = np.argsort(prob, axis=1)
    top3 = np.mean(np.any(order[:, -3:] == y_true[:, None], axis=1))
    top5 = np.mean(np.any(order[:, -5:] == y_true[:, None], axis=1))
    acc = accuracy_score(y_true, pred)
    macro_f1 = f1_score(y_true, pred, average="macro", zero_division=0)

    print(f"[{label:30s}] Top-1={acc*100:5.2f}% | Top-3={top3*100:5.2f}% | Top-5={top5*100:5.2f}% | F1={macro_f1*100:5.2f}%")
    return {
        "label": label,
        "accuracy": float(acc),
        "top3": float(top3),
        "top5": float(top5),
        "macro_f1": float(macro_f1),
        "prob": prob,
        "pred": pred
    }


def run_experiment(label, selected_columns, ensemble=False, deep=True, save=False):
    """Chạy một thử nghiệm với tập đặc trưng và mô hình tương ứng."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[train_idx][:, selected_columns]).astype(np.float32)
    X_va = scaler.transform(X[val_idx][:, selected_columns]).astype(np.float32)

    seeds = range(5) if ensemble else [SEED]
    models = []
    for s in seeds:
        m, _ = train_mlp_seed(X_tr, y[train_idx], X_va, y[val_idx], len(names), seed=s, deep=deep)
        models.append(m)

    val_tensor = torch.from_numpy(X_va).to(DEVICE)
    with torch.no_grad():
        all_probs = [torch.softmax(m(val_tensor), dim=1).cpu().numpy() for m in models]
    prob = np.mean(all_probs, axis=0)

    metrics = evaluate_predictions(label, y[val_idx], prob, names)

    if save:
        import joblib
        for i, m in enumerate(models):
            torch.save(m.state_dict(), MODEL_DIR / f"mlp_{i}.pt")
        joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
        meta = {
            "in_dim": len(selected_columns),
            "num_classes": len(names),
            "cols": selected_columns.tolist(),
            "image_size": int(data["image_size"]),
            "metrics": {k: metrics[k] for k in ["accuracy", "top3", "top5", "macro_f1"]}
        }
        joblib.dump(meta, MODEL_DIR / "meta.pkl")
        joblib.dump(labels_dict, MODEL_DIR / "labels.pkl")
        np.save(MODEL_DIR / "validation_prob.npy", prob)
        print(f"Đã lưu mô hình tối ưu vào '{MODEL_DIR}/'")

    return metrics


# %%
# ==============================================================================
# BẢNG THỬ NGHIỆM BÓC TÁCH (ABLATION STUDY) & SO SÁNH
# ==============================================================================

print("\n" + "=" * 80)
print("🚀 BẮT ĐẦU ABLATION STUDY & SO SÁNH CÁC NHÓM ĐẶC TRƯNG DSP")
print("=" * 80)

ablation_configs = [
    ("E00_baseline_hog_hist_fd", cols("hog", "hist", "fd")),
    ("E01_plus_fine_hog", cols("hog", "hog_fine", "hist", "fd")),
    ("E02_plus_spatial_color", cols("hog", "hist", "fd", "sp_hist")),
    ("E03_plus_moments", cols("hog", "hist", "fd", "sp_hist", "moments")),
    ("E04_plus_lbp", cols("hog", "hist", "fd", "sp_hist", "moments", "lbp")),
    ("E05_plus_shape_hu", cols("hog", "hist", "fd", "sp_hist", "moments", "lbp", "geometry", "hu")),
    ("E06_plus_glcm", cols("hog", "hist", "fd", "sp_hist", "moments", "lbp", "geometry", "hu", "glcm")),
    ("E07_plus_gabor", cols("hog", "hist", "fd", "sp_hist", "moments", "lbp", "geometry", "hu", "glcm", "gabor")),
    ("E08_all_dsp_features", cols("hog", "hog_fine", "hist", "fd", "sp_hist", "moments", "lbp", "geometry", "hu", "glcm", "gabor", "fourier_2d")),
    ("E09_all_dsp_deep_mlp", cols("hog", "hog_fine", "hist", "fd", "sp_hist", "moments", "lbp", "geometry", "hu", "glcm", "gabor", "fourier_2d")),
    ("E10_all_dsp_ensemble_5seeds", cols("hog", "hog_fine", "hist", "fd", "sp_hist", "moments", "lbp", "geometry", "hu", "glcm", "gabor", "fourier_2d")),
]

results = []
for label, col_subset in ablation_configs:
    if len(col_subset) == 0:
        continue
    is_ensemble = "ensemble" in label
    is_deep = "deep" in label or "ensemble" in label
    save_model = label == "E10_all_dsp_ensemble_5seeds"

    res = run_experiment(label, col_subset, ensemble=is_ensemble, deep=is_deep, save=save_model)
    results.append(res)

    # Thử nghiệm thêm: Tái xếp hạng nhận thức họ tiến hóa (Evolution Reranking)
    if is_ensemble:
        rerank_label = f"{label} + evolution_rerank"
        rerank_res = evaluate_predictions(rerank_label, y[val_idx], res["prob"], names, apply_rerank=True)
        results.append(rerank_res)

# %%
# ==============================================================================
# LƯU KẾT QUẢ THÍ NGHIỆM VÀO CSV
# ==============================================================================

csv_path = RESULTS_DIR / "experiments.csv"
with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Experiment_Label", "Top1_Accuracy", "Top3_Accuracy", "Top5_Accuracy", "Macro_F1"])
    for r in results:
        writer.writerow([r["label"], f"{r['accuracy']*100:.2f}%", f"{r['top3']*100:.2f}%", f"{r['top5']*100:.2f}%", f"{r['macro_f1']*100:.2f}%"])

print(f"\nĐã ghi kết quả vào: {csv_path}")

# %%
# Vẽ biểu đồ so sánh các thử nghiệm
plt.figure(figsize=(12, 6))
labels = [r["label"] for r in results]
accs = [r["accuracy"] * 100 for r in results]
top3s = [r["top3"] * 100 for r in results]

x_pos = np.arange(len(labels))
width = 0.35

plt.bar(x_pos - width / 2, accs, width, label="Top-1 Accuracy", color="#3498db")
plt.bar(x_pos + width / 2, top3s, width, label="Top-3 Accuracy", color="#2ecc71")

plt.xticks(x_pos, labels, rotation=45, ha="right", fontsize=9)
plt.ylabel("Độ chính xác (%)")
plt.title("So sánh các cấu hình đặc trưng DSP và Mô hình Machine Learning")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "ablation_comparison.png", dpi=200)
plt.close()

print(f"Đã lưu biểu đồ so sánh: {RESULTS_DIR / 'ablation_comparison.png'}")
