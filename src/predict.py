# src/predict.py

import os
import joblib
import numpy as np
from src.preprocess import extract_pokemon
from src.features import extract_features


#  Load model 1 lần khi import — không load lại mỗi lần predict
_model  = None
_labels = None
_scaler = None


def _load_model():
    global _model, _labels, _scaler
    if _model is None:
        model_dir = "Model"
        _model  = joblib.load(os.path.join(model_dir, "model.pkl"))
        _labels = joblib.load(os.path.join(model_dir, "labels.pkl"))
        _scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))


def predict(image_path):
    _load_model()

    img = extract_pokemon(image_path)
    if img is None:
        return "Không đọc được ảnh", 0.0

    feat = extract_features(img)
    feat = _scaler.transform([feat])

    pred = _model.predict(feat)[0]
    prob = _model.predict_proba(feat)[0]

    return _labels[pred], float(max(prob))


if __name__ == "__main__":
    path = input("Ảnh: ")
    name, conf = predict(path)
    print(f"{name} ({conf:.2f})")