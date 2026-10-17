# %% [markdown]
# Notebook B - Huấn luyện mô hình, Thử nghiệm bóc tách (Ablation Study)
# & Tự nhận định Tộc/Họ Không giám sát (Self-Discovered Visual Taxonomy).

# %%
import json
import os
import random
import sys
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(os.environ.get("POKEDEX_ROOT", Path.cwd()))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import FEATURE_GROUPS, compute_block_weights
from src.mlp import build_mlp, discover_visual_clusters, taxonomy_rerank
from src.evolution import evolution_rerank

SEED = 42
N_CLUSTERS = 24  # Số lượng Tộc/Họ để máy tự phân cụm từ đặc trưng DSP

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Tự động phát hiện môi trường Kaggle và dùng đúng đường dẫn input
_KAGGLE_FEATURES = Path("/kaggle/input/notebooks/loc1208/pokedex-ml/POKEDEX/features.npz")
if "POKEDEX_FEATURES" in os.environ:
    FEATURES_PATH = Path(os.environ["POKEDEX_FEATURES"])
elif _KAGGLE_FEATURES.exists():
    FEATURES_PATH = _KAGGLE_FEATURES
elif os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None:
    FEATURES_PATH = _KAGGLE_FEATURES
elif Path("/kaggle/working/features.npz").exists():
    FEATURES_PATH = Path("/kaggle/working/features.npz")
else:
    FEATURES_PATH = ROOT / "features.npz"

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

# Xác định vị trí ảnh gốc (aug_idx == 0) và ảnh lật ngang (aug_idx == 1) trong từng group
# Trong dataset.py, mỗi ảnh gốc sinh ra N_aug biến thể liên tiếp có cùng group_id
is_orig = np.zeros(len(groups), dtype=bool)
is_flip = np.zeros(len(groups), dtype=bool)
prev_g = None
offset_in_group = 0
for i, g in enumerate(groups):
    if g != prev_g:
        offset_in_group = 0
        prev_g = g
    else:
        offset_in_group += 1
    if offset_in_group == 0:
        is_orig[i] = True
    elif offset_in_group == 1:
        is_flip[i] = True

print(f"Total Samples: {len(X)}, Dim: {X.shape[1]}, Classes: {len(names)}, Unique Images: {np.sum(is_orig)}")

# %%
# Chia dữ liệu StratifiedGroupKFold (Zero Data Leakage)
splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
train_idx, val_all_idx = next(splitter.split(X, y, groups))

assert set(groups[train_idx]).isdisjoint(set(groups[val_all_idx])), "Data leakage detected!"

# Tập Validation chuẩn: Đánh giá trên ảnh gốc (kèm cặp ảnh Flip ngang để chạy Flip TTA như thực tế)
val_orig_idx = val_all_idx[is_orig[val_all_idx]]
val_flip_idx = val_all_idx[is_flip[val_all_idx]]
use_val_tta = (len(val_orig_idx) == len(val_flip_idx)) and np.all(groups[val_orig_idx] == groups[val_flip_idx])

print(f"Train samples (with augmentations): {len(train_idx)}")
print(f"Val original images: {len(val_orig_idx)} (Flip TTA enabled: {use_val_tta})")

labels_dict = {i: str(names[i]) for i in range(len(names))}


# %%
def cols(*keys):
    selected = []
    for key in keys:
        if key in saved_groups:
            start, end = saved_groups[key]
            selected.extend(range(int(start), int(end)))
    return np.asarray(selected, dtype=np.int64)


def train_mlp_seed(
    X_train, y_train, X_val_orig, X_val_flip, y_val,
    num_classes, seed, epochs=85, deep=True, class_to_cluster=None,
    use_cosine_head=False
):
    torch.manual_seed(seed)
    n_clusters = int(class_to_cluster.max() + 1) if class_to_cluster is not None else 0
    model = build_mlp(
        X_train.shape[1], num_classes, deep=deep,
        n_clusters=n_clusters, use_cosine_head=use_cosine_head
    ).to(DEVICE)

    if class_to_cluster is not None:
        y_train_cluster = torch.from_numpy(class_to_cluster[y_train]).long()
        train_ds = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train),
            y_train_cluster
        )
    else:
        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))

    train_loader = DataLoader(
        train_ds,
        batch_size=256,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    val_orig_t = torch.from_numpy(X_val_orig).to(DEVICE)
    val_flip_t = torch.from_numpy(X_val_flip).to(DEVICE) if X_val_flip is not None else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.5e-2)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1.2e-3, epochs=epochs, steps_per_epoch=len(train_loader)
    )
    criterion_species = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    criterion_cluster = torch.nn.CrossEntropyLoss(label_smoothing=0.02)

    best_acc, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if class_to_cluster is not None:
                bx, by, bc = [b.to(DEVICE) for b in batch]
                sp_logits, cl_logits = model(bx, return_cluster=True)
                loss = criterion_species(sp_logits, by) + 0.35 * criterion_cluster(cl_logits, bc)
            else:
                bx, by = batch[0].to(DEVICE), batch[1].to(DEVICE)
                loss = criterion_species(model(bx), by)
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            logits = model(val_orig_t)
            if val_flip_t is not None:
                logits = 0.5 * (logits + model(val_flip_t))
            val_pred = logits.argmax(dim=1).cpu().numpy()
            acc = accuracy_score(y_val, val_pred)
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_acc


def train_linear_classifier(X_train, y_train, X_val_orig, X_val_flip=None):
    """
    Huấn luyện Logistic Regression (L2 Softmax / Linear Classifier) đa luồng
    nhanh và cho xác suất hiệu chỉnh tốt để Blending trực giao với DeepMLP.
    """
    from sklearn.linear_model import LogisticRegression
    print("      -> Huấn luyện mô hình Linear Classifier (SVM/Softmax Blending)...")
    clf = LogisticRegression(C=1.0, max_iter=250, solver="lbfgs", random_state=SEED, n_jobs=-1)
    clf.fit(X_train, y_train)
    prob_orig = clf.predict_proba(X_val_orig)
    if X_val_flip is not None:
        prob_flip = clf.predict_proba(X_val_flip)
        prob = 0.5 * (prob_orig + prob_flip)
    else:
        prob = prob_orig
    return clf, prob


def evaluate_predictions(label, y_true, prob):
    pred = prob.argmax(axis=1)
    order = np.argsort(prob, axis=1)
    top3 = np.mean(np.any(order[:, -3:] == y_true[:, None], axis=1))
    top5 = np.mean(np.any(order[:, -5:] == y_true[:, None], axis=1))
    acc = accuracy_score(y_true, pred)
    macro_f1 = f1_score(y_true, pred, average="macro", zero_division=0)

    print(f"[{label:42s}] Top-1={acc*100:5.2f}% | Top-3={top3*100:5.2f}% | Top-5={top5*100:5.2f}% | F1={macro_f1*100:5.2f}%")
    return {
        "label": label,
        "accuracy": float(acc),
        "top3": float(top3),
        "top5": float(top5),
        "macro_f1": float(macro_f1),
        "prob": prob,
        "pred": pred
    }


def run_experiment(
    label, selected_columns, ensemble=False, deep=True,
    use_block_weights=True, use_self_taxonomy=False,
    use_cosine_head=False, n_clusters=N_CLUSTERS,
    svm_blend=False, alpha_svm=0.15, save=False
):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[train_idx][:, selected_columns]).astype(np.float32)
    X_va_orig = scaler.transform(X[val_orig_idx][:, selected_columns]).astype(np.float32)
    X_va_flip = scaler.transform(X[val_flip_idx][:, selected_columns]).astype(np.float32) if use_val_tta else None

    if use_block_weights:
        w = compute_block_weights(saved_groups, selected_columns, total_dim=X.shape[1])
        X_tr = X_tr * w
        X_va_orig = X_va_orig * w
        if X_va_flip is not None:
            X_va_flip = X_va_flip * w
    else:
        w = np.ones(len(selected_columns), dtype=np.float32)

    class_to_cluster = None
    if use_self_taxonomy:
        class_to_cluster = discover_visual_clusters(
            X_tr, y[train_idx], num_classes=len(names), n_clusters=n_clusters
        )

    seeds = range(5) if ensemble else [SEED]
    models = []
    for s in seeds:
        m, _ = train_mlp_seed(
            X_tr, y[train_idx], X_va_orig, X_va_flip, y[val_orig_idx],
            len(names), seed=s, deep=deep, class_to_cluster=class_to_cluster,
            use_cosine_head=use_cosine_head
        )
        models.append(m)

    val_orig_t = torch.from_numpy(X_va_orig).to(DEVICE)
    val_flip_t = torch.from_numpy(X_va_flip).to(DEVICE) if X_va_flip is not None else None

    all_sp_probs = []
    all_cl_probs = []
    with torch.no_grad():
        for m in models:
            if use_self_taxonomy:
                sp1, cl1 = m(val_orig_t, return_cluster=True)
                if val_flip_t is not None:
                    sp2, cl2 = m(val_flip_t, return_cluster=True)
                    sp_p = 0.5 * (torch.softmax(sp1, dim=1) + torch.softmax(sp2, dim=1))
                    cl_p = 0.5 * (torch.softmax(cl1, dim=1) + torch.softmax(cl2, dim=1))
                else:
                    sp_p = torch.softmax(sp1, dim=1)
                    cl_p = torch.softmax(cl1, dim=1)
                all_sp_probs.append(sp_p.cpu().numpy())
                all_cl_probs.append(cl_p.cpu().numpy())
            else:
                sp1 = torch.softmax(m(val_orig_t), dim=1)
                if val_flip_t is not None:
                    sp2 = torch.softmax(m(val_flip_t), dim=1)
                    sp1 = 0.5 * (sp1 + sp2)
                all_sp_probs.append(sp1.cpu().numpy())

    prob = np.mean(all_sp_probs, axis=0)
    if use_self_taxonomy:
        cl_prob = np.mean(all_cl_probs, axis=0)
        prob = taxonomy_rerank(prob, cl_prob, class_to_cluster, beta=0.25)

    svm_model = None
    if svm_blend:
        svm_model, svm_prob = train_linear_classifier(
            X_tr, y[train_idx], X_va_orig, X_va_flip
        )
        prob = (1.0 - alpha_svm) * prob + alpha_svm * svm_prob

    metrics = evaluate_predictions(label, y[val_orig_idx], prob)

    if save:
        import joblib
        for i, m in enumerate(models):
            torch.save(m.state_dict(), MODEL_DIR / f"mlp_{i}.pt")
        if svm_model is not None:
            joblib.dump(svm_model, MODEL_DIR / "svm_blend.pkl")
        joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
        meta = {
            "in_dim": len(selected_columns),
            "num_classes": len(names),
            "cols": selected_columns.tolist(),
            "w": w.tolist(),
            "n_clusters": int(n_clusters) if use_self_taxonomy else 0,
            "class_to_cluster": class_to_cluster.tolist() if class_to_cluster is not None else None,
            "use_cosine_head": bool(use_cosine_head),
            "has_svm_blend": bool(svm_blend),
            "alpha_svm": float(alpha_svm) if svm_blend else 0.0,
            "image_size": int(data["image_size"]),
            "metrics": {k: metrics[k] for k in ["accuracy", "top3", "top5", "macro_f1"]}
        }
        joblib.dump(meta, MODEL_DIR / "meta.pkl")
        joblib.dump(labels_dict, MODEL_DIR / "labels.pkl")
        np.save(MODEL_DIR / "validation_prob.npy", prob)
        print(f"Đã lưu mô hình tối ưu vào '{MODEL_DIR}/'")

    return metrics


# %%
print("\n" + "=" * 85)
print("🚀 BẮT ĐẦU ABLATION STUDY & TỰ NHẬN ĐỊNH TỘC/HỌ KHÔNG GIÁM SÁT (SELF-TAXONOMY)")
print("=" * 85)

all_feature_keys = [
    k for k in (
        "hog", "hog_fine", "hist", "fd", "sp_hist", "moments",
        "lbp", "geometry", "hu", "glcm", "gabor", "fourier_2d", "stage_morph"
    )
    if k in saved_groups
]

ablation_configs = [
    # (label, col_subset, is_ensemble, is_deep, use_block_weights, use_self_taxonomy, use_cosine_head, n_clusters, svm_blend)
    ("E00_baseline_hog_hist_fd", cols("hog", "hist", "fd"), False, False, False, False, False, 0, False),
    ("E02_plus_spatial_color", cols("hog", "hist", "fd", "sp_hist"), False, False, False, False, False, 0, False),
    ("E08_all_dsp_legacy_mlp", cols(*all_feature_keys), False, False, True, False, False, 0, False),
    ("E09_all_dsp_deep_mlp_weighted", cols(*all_feature_keys), False, True, True, False, False, 0, False),
    ("E10a_taxonomy_k24_ce_head", cols(*all_feature_keys), False, True, True, True, False, 24, False),
    ("E10b_taxonomy_k24_cosine_head", cols(*all_feature_keys), False, True, True, True, True, 24, False),
    ("E11_ensemble_5seeds_cosine_tax", cols(*all_feature_keys), True, True, True, True, True, 24, False),
    ("E12_ensemble_svm_blending", cols(*all_feature_keys), True, True, True, True, True, 24, True),
]

results = []
for label, col_subset, is_ens, is_deep, use_bw, use_tax, use_cos, n_cl, svm_b in ablation_configs:
    if len(col_subset) == 0:
        continue
    save_model = (label == "E12_ensemble_svm_blending")
    res = run_experiment(
        label, col_subset,
        ensemble=is_ens, deep=is_deep,
        use_block_weights=use_bw,
        use_self_taxonomy=use_tax,
        use_cosine_head=use_cos,
        n_clusters=n_cl,
        svm_blend=svm_b,
        save=save_model
    )
    results.append(res)

# %%
csv_path = RESULTS_DIR / "experiments.csv"
with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Experiment_Label", "Top1_Accuracy", "Top3_Accuracy", "Top5_Accuracy", "Macro_F1"])
    for r in results:
        writer.writerow([
            r["label"],
            f"{r['accuracy']*100:.2f}%",
            f"{r['top3']*100:.2f}%",
            f"{r['top5']*100:.2f}%",
            f"{r['macro_f1']*100:.2f}%"
        ])

print(f"\nĐã ghi kết quả vào: {csv_path}")

# %%
# Vẽ biểu đồ so sánh các thử nghiệm
plt.figure(figsize=(14, 6))
labels = [r["label"] for r in results]
accs = [r["accuracy"] * 100 for r in results]
top3s = [r["top3"] * 100 for r in results]

x_pos = np.arange(len(labels))
width = 0.35

plt.bar(x_pos - width / 2, accs, width, label="Top-1 Accuracy", color="#3498db")
plt.bar(x_pos + width / 2, top3s, width, label="Top-3 Accuracy", color="#2ecc71")

plt.xticks(x_pos, labels, rotation=35, ha="right", fontsize=9)
plt.ylabel("Độ chính xác (%)")
plt.title("Ablation Study: Tối ưu đặc trưng DSP & Mô hình Phân loại 150 loài Pokémon")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "ablation_comparison.png", dpi=200)
plt.close()

print(f"Đã lưu biểu đồ so sánh: {RESULTS_DIR / 'ablation_comparison.png'}")