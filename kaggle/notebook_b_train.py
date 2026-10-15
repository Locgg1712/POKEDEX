# %% [markdown]
# Notebook B - grouped, stratified MLP training and ablation evaluation.
# The script intentionally uses the first fold of StratifiedGroupKFold so every
# class is represented in validation without leaking augmented copies.

# %%
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

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

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURES_PATH = Path(os.environ.get("POKEDEX_FEATURES", ROOT / "features.npz"))
MODEL_DIR = Path(os.environ.get("POKEDEX_MODEL", ROOT / "Model"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# %%
data = np.load(FEATURES_PATH, allow_pickle=False)
X = data["X"].astype(np.float32)
y = data["y"].astype(np.int64)
groups = data["groups"].astype(str)
names = data["names"].astype(str)
assert X.shape[1] == FEATURE_DIM
assert int(data["image_size"]) == IMAGE_SIZE, (
    f"Feature image size {int(data['image_size'])} != notebook image size {IMAGE_SIZE}"
)
if "group_columns" in data:
    saved_groups = json.loads(str(data["group_columns"].item()))
else:
    saved_groups = FEATURE_GROUPS

splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
train_idx, val_idx = next(splitter.split(X, y, groups))
assert set(groups[train_idx]).isdisjoint(set(groups[val_idx]))
assert set(y[val_idx]) == set(range(len(names))), "Validation is missing a class"
print(f"Train={len(train_idx)}, val={len(val_idx)}, classes={len(names)}")

# %%
def cols(*keys):
    """Return absolute feature columns for an ablation."""
    selected = []
    for key in keys:
        start, end = saved_groups[key]
        selected.extend(range(int(start), int(end)))
    return np.asarray(selected, dtype=np.int64)


def train_one(X_train, y_train, X_val, y_val, seed, weight, epochs=80):
    torch.manual_seed(seed)
    model = build_mlp(X_train.shape[1], len(names)).to(DEVICE)
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
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    noise_scale = torch.from_numpy(weight).to(DEVICE) * 0.1
    best_acc, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            noisy = batch_x + torch.randn_like(batch_x) * noise_scale
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(noisy), batch_y)
            loss.backward()
            optimizer.step()
            scheduler.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(val_tensor).argmax(dim=1).cpu().numpy()
        acc = accuracy_score(y_val, val_pred)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_acc


def probabilities(models, values):
    tensor = torch.from_numpy(values).to(DEVICE)
    with torch.no_grad():
        probs = [torch.softmax(model(tensor), dim=1).cpu().numpy() for model in models]
    return np.mean(probs, axis=0)


def evaluate(label, y_true, prob):
    pred = prob.argmax(axis=1)
    order = np.argsort(prob, axis=1)
    top3 = np.mean(np.any(order[:, -3:] == y_true[:, None], axis=1))
    top5 = np.mean(np.any(order[:, -5:] == y_true[:, None], axis=1))
    report = classification_report(
        y_true, pred, target_names=names, zero_division=0, output_dict=True
    )
    print(f"{label}: accuracy={accuracy_score(y_true, pred):.4f}, "
          f"top3={top3:.4f}, top5={top5:.4f}, "
          f"macro_f1={f1_score(y_true, pred, average='macro', zero_division=0):.4f}")
    confidence = prob.max(axis=1)
    for threshold in (0.5, 0.8):
        mask = confidence >= threshold
        print(f"  confidence>={threshold}: coverage={mask.mean():.4f}, "
              f"accuracy={accuracy_score(y_true[mask], pred[mask]):.4f}"
              if mask.any() else f"  confidence>={threshold}: coverage=0")
    hard = sorted(
        ((name, values["f1-score"]) for name, values in report.items()
         if name in names),
        key=lambda item: item[1],
    )[:10]
    print("  hardest:", ", ".join(f"{name}={score:.3f}" for name, score in hard))
    pairs = Counter((int(true), int(p))
                    for true, p in zip(y_true, pred) if true != p)
    print("  confusion pairs:", [
        (names[a], names[b], count) for (a, b), count in pairs.most_common(10)
    ])
    return pred, report


def run_ablation(label, selected, ensemble=False, weighted=False, save=False):
    scaler = StandardScaler()
    train_x = scaler.fit_transform(X[train_idx][:, selected]).astype(np.float32)
    val_x = scaler.transform(X[val_idx][:, selected]).astype(np.float32)
    weight = np.ones(len(selected), dtype=np.float32)
    if weighted:
        weight[:] = 1.0 / np.sqrt(len(selected))
        for key in saved_groups:
            group_columns = np.arange(saved_groups[key][0], saved_groups[key][1])
            selected_positions = np.flatnonzero(np.isin(selected, group_columns))
            if len(selected_positions):
                weight[selected_positions] = 1.0 / np.sqrt(len(group_columns))
        train_x *= weight
        val_x *= weight
    models = []
    seeds = range(5) if ensemble else [SEED]
    for seed in seeds:
        model, _ = train_one(
            train_x, y[train_idx], val_x, y[val_idx], seed, weight
        )
        models.append(model)
    prob = probabilities(models, val_x)
    pred, report = evaluate(label, y[val_idx], prob)
    if save:
        for i, model in enumerate(models):
            torch.save(model.state_dict(), MODEL_DIR / f"mlp_{i}.pt")
        joblib_dump(scaler, MODEL_DIR / "scaler.pkl")
        w = weight
        meta = {"in_dim": len(selected), "num_classes": len(names),
                "w": w, "cols": selected.tolist(), "image_size": IMAGE_SIZE}
        joblib_dump(meta, MODEL_DIR / "meta.pkl")
        joblib_dump({i: str(name) for i, name in enumerate(names)}, MODEL_DIR / "labels.pkl")
        np.save(MODEL_DIR / "validation_prob.npy", prob)
        with open(MODEL_DIR / "report.txt", "w", encoding="utf-8") as stream:
            stream.write(classification_report(
                y[val_idx], pred, target_names=names, zero_division=0
            ))
    return prob, report


def joblib_dump(value, path):
    import joblib
    joblib.dump(value, path)

# %%
ablation = [
    ("baseline", cols("hog", "hist", "fd")),
    ("plus_sp_hist", cols("hog", "hist", "fd", "sp_hist")),
    ("plus_moments", cols("hog", "hist", "fd", "sp_hist", "moments")),
    ("all_features", cols("hog", "hist", "fd", "sp_hist", "moments", "lbp")),
    ("all_features_weighted", cols("hog", "hist", "fd", "sp_hist", "moments", "lbp")),
    ("all_features_ensemble", cols("hog", "hist", "fd", "sp_hist", "moments", "lbp")),
]
results = []
for label, selected in ablation:
    prob, report = run_ablation(
        label, selected, weighted=label == "all_features_weighted",
        ensemble=label == "all_features_ensemble",
        save=label == "all_features_ensemble",
    )
    pred = prob.argmax(axis=1)
    results.append((label, accuracy_score(y[val_idx], pred),
                    f1_score(y[val_idx], pred, average="macro", zero_division=0)))

# %%
pred = np.asarray(np.load(MODEL_DIR / "validation_prob.npy")).argmax(axis=1)
cm = confusion_matrix(y[val_idx], pred, labels=np.arange(len(names)), normalize="true")
plt.figure(figsize=(18, 16))
sns.heatmap(cm, xticklabels=names, yticklabels=names, cmap="Blues")
plt.title("Confusion matrix (row normalized)")
plt.tight_layout()
plt.savefig(MODEL_DIR / "confusion.png", dpi=160)
plt.close()
confidence = np.load(MODEL_DIR / "validation_prob.npy").max(axis=1)
correct = pred == y[val_idx]
plt.figure(figsize=(9, 5))
plt.hist(confidence[correct], bins=20, alpha=0.7, label="correct")
plt.hist(confidence[~correct], bins=20, alpha=0.7, label="wrong")
plt.xlabel("Confidence")
plt.ylabel("Samples")
plt.legend()
plt.tight_layout()
plt.savefig(MODEL_DIR / "confidence.png", dpi=160)
plt.close()
print("Ablation")
for row in results:
    print(f"{row[0]:25s} accuracy={row[1]:.4f} macro_f1={row[2]:.4f}")

# %%
# Centroids are kept in the original three component spaces used by explain.py.
import shutil

sel = {
    key: np.arange(*map(int, saved_groups[key]))
    for key in ("hog", "hist", "fd")
}
cent = {}
for i, name in enumerate(names):
    class_features = X[y == i]
    cent[str(name)] = {
        key: class_features[:, columns].mean(axis=0)
        for key, columns in sel.items()
    }
joblib_dump(cent, MODEL_DIR / "centroids.pkl")
shutil.make_archive("/kaggle/working/Model", "zip", MODEL_DIR)
print(f"Saved centroids and /kaggle/working/Model.zip")
