# src/train.py

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import StandardScaler

from src.dataset import load_dataset
from src.model import create_base_model
from src.preprocess import IMAGE_SIZE
from src.features import FEATURE_DIM


def train(data_dir="data", model_type="svm", epochs=60, batch_size=128):
    """
    Train model with StratifiedGroupKFold to guarantee zero data leakage
    between original images and their augmented versions.
    """
    print(f"Loading data from '{data_dir}' (IMAGE_SIZE={IMAGE_SIZE})...")
    X, y, labels, groups = load_dataset(data_dir, return_groups=True)
    num_classes = len(labels)
    in_dim = X.shape[1]
    print(f"Total samples: {len(X)}, Features dim: {in_dim}, Classes: {num_classes}")

    # --- Stratified Group Split (Zero Data Leakage) ---
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(splitter.split(X, y, groups))

    # Verification: groups in train must NOT be in val
    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])
    overlap = train_groups.intersection(val_groups)
    assert len(overlap) == 0, f"Critical error: {len(overlap)} groups leaked into validation!"

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    print(f"Train samples: {len(X_train)} ({len(train_groups)} groups)")
    print(f"Val samples:   {len(X_val)} ({len(val_groups)} groups)")

    # --- Feature Scaling (fit ONLY on train) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model_dir = "Model"
    os.makedirs(model_dir, exist_ok=True)

    if model_type == "svm":
        print("\nTraining SVM (RBF kernel)...")
        # Fast parameter search or direct fit depending on dataset size
        if len(X_train_scaled) > 4000:
            print("Large dataset detected: using SVC(C=10.0, kernel='rbf', probability=True)...")
            from sklearn.svm import SVC
            model = SVC(C=10.0, kernel="rbf", probability=True, class_weight="balanced", random_state=42)
            model.fit(X_train_scaled, y_train)
        else:
            base_model = create_base_model()
            param_grid = {"C": [1.0, 10.0, 50.0], "gamma": ["scale"]}
            grid = GridSearchCV(base_model, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
            grid.fit(X_train_scaled, y_train)
            model = grid.best_estimator_
            print("Best SVM params:", grid.best_params_)

        y_probs = model.predict_proba(X_val_scaled)
        y_pred = np.argmax(y_probs, axis=1)

        joblib.dump(model, os.path.join(model_dir, "model.pkl"))

    elif model_type == "mlp":
        print("\nTraining MLP with PyTorch...")
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        from src.mlp import build_mlp

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_mlp(in_dim, num_classes).to(device)

        train_dataset = TensorDataset(torch.from_numpy(X_train_scaled).float(), torch.from_numpy(y_train).long())
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_tensor = torch.from_numpy(X_val_scaled).float().to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, epochs=epochs, steps_per_epoch=len(train_loader)
        )
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

        best_acc = -1.0
        best_state = None

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad(set_to_none=True)
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            model.eval()
            with torch.no_grad():
                val_out = model(val_tensor)
                val_preds = val_out.argmax(dim=1).cpu().numpy()
                acc = accuracy_score(y_val, val_preds)

            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch [{epoch+1:02d}/{epochs}] Loss: {total_loss/len(train_loader):.4f} - Val Acc: {acc*100:.2f}% (Best: {best_acc*100:.2f}%)")

        model.load_state_dict(best_state)
        torch.save(model.state_dict(), os.path.join(model_dir, "mlp.pt"))

        model.eval()
        with torch.no_grad():
            y_probs = torch.softmax(model(val_tensor), dim=1).cpu().numpy()
            y_pred = np.argmax(y_probs, axis=1)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # --- Comprehensive Evaluation ---
    acc = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)

    # Top-K Accuracies
    order = np.argsort(y_probs, axis=1)
    top3 = np.mean(np.any(order[:, -3:] == y_val[:, None], axis=1))
    top5 = np.mean(np.any(order[:, -5:] == y_val[:, None], axis=1))

    print("\n" + "=" * 50)
    print("🎯 EVALUATION RESULTS (Unseen Augmented Groups)")
    print("=" * 50)
    print(f"Top-1 Accuracy: {acc * 100:.2f}%")
    print(f"Top-3 Accuracy: {top3 * 100:.2f}%")
    print(f"Top-5 Accuracy: {top5 * 100:.2f}%")
    print(f"Macro F1-Score: {macro_f1 * 100:.2f}%")
    print("=" * 50)

    target_names = [labels[i] for i in sorted(labels.keys())]
    if len(target_names) <= 20:
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred, target_names=target_names, zero_division=0))

    # --- Save Metadata & Artifacts ---
    meta = {
        "in_dim": in_dim,
        "num_classes": num_classes,
        "image_size": IMAGE_SIZE,
        "model_type": model_type,
        "metrics": {
            "top1": float(acc),
            "top3": float(top3),
            "top5": float(top5),
            "macro_f1": float(macro_f1),
        }
    }
    joblib.dump(meta, os.path.join(model_dir, "meta.pkl"))
    joblib.dump(labels, os.path.join(model_dir, "labels.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    np.save(os.path.join(model_dir, "validation_prob.npy"), y_probs)

    print(f"\nSaved models and artifacts to '{model_dir}/'.")
    return acc, top3, macro_f1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pokédex Model Training")
    parser.add_argument("--data", default="data", help="Path to data directory")
    parser.add_argument("--model", default="svm", choices=["svm", "mlp"], help="Model type to train")
    parser.add_argument("--epochs", type=int, default=60, help="Epochs for MLP")
    args = parser.parse_args()
    train(data_dir=args.data, model_type=args.model, epochs=args.epochs)