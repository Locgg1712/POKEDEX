# src/train.py

from src.dataset import load_dataset
from src.model import create_base_model

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def train(data_dir="data"):
    print("Loading data...")
    X, y, labels = load_dataset(data_dir)

    # ==============================
    # SPLIT
    # ==============================
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"\nTrain: {len(X_train)} | Val: {len(X_val)}")

    # ==============================
    # SCALE (tránh leakage)
    # ==============================
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    # ==============================
    # GIẢM DATA ĐỂ TUNE NHANH
    # ==============================
    subset_size = 2000  
    X_small = X_train[:subset_size]
    y_small = y_train[:subset_size]

    print(f"\nGridSearch trên {subset_size} samples...")

    # ==============================
    # MODEL
    # ==============================
    model = create_base_model()

    # ==============================
    #  GRID NHẸ 
    # ==============================
    param_grid = {
        'C': [10, 100],
        'gamma': ['scale'],
        'kernel': ['rbf']
    }

    grid = GridSearchCV(
        model,
        param_grid,
        cv=3,              #  giảm từ 5 → 3
        verbose=2,
        n_jobs=-1,
        scoring='accuracy'
    )

    # ==============================
    # TRAIN GRID
    # ==============================
    print("\nTraining với GridSearch (FAST MODE)...")
    grid.fit(X_small, y_small)

    best_model = grid.best_estimator_

    print("\nBest params:", grid.best_params_)

    # ==============================
    #  TRAIN LẠI FULL DATA
    # ==============================
    print("\nTraining lại trên full dataset...")
    best_model.fit(X_train, y_train)

    # ==============================
    # EVALUATION
    # ==============================
    y_pred = best_model.predict(X_val)
    acc    = accuracy_score(y_val, y_pred)

    print(f"\nAccuracy: {acc * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(
        y_val, y_pred,
        target_names=[labels[i] for i in sorted(labels)]
    ))

    # ==============================
    # CONFUSION MATRIX
    # ==============================
    cm = confusion_matrix(y_val, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d',
        xticklabels=[labels[i] for i in sorted(labels)],
        yticklabels=[labels[i] for i in sorted(labels)]
    )

    plt.title(f"Confusion Matrix — Accuracy: {acc*100:.1f}%")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # ==============================
    # SAVE MODEL
    # ==============================
    import os
    model_dir = "Model"
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(best_model, os.path.join(model_dir, "model.pkl"))
    joblib.dump(labels,     os.path.join(model_dir, "labels.pkl"))
    joblib.dump(scaler,     os.path.join(model_dir, "scaler.pkl"))

    print("\nSaved model.pkl / labels.pkl / scaler.pkl into Model/")


if __name__ == "__main__":
    train()