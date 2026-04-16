# src/predict.py

import os
import joblib
import numpy as np
from src.preprocess import extract_pokemon
from src.features import extract_feature_components
from src.explain import analyze_features, generate_explanation

# Load model 1 lần khi import
_model  = None
_labels = None
_scaler = None
_centroids = None


def _load_model():
    global _model, _labels, _scaler, _centroids
    if _model is None:
        model_dir = "Model"
        _model  = joblib.load(os.path.join(model_dir, "model.pkl"))
        _labels = joblib.load(os.path.join(model_dir, "labels.pkl"))
        _scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        try:
            _centroids = joblib.load(os.path.join(model_dir, "centroids.pkl"))
        except FileNotFoundError:
            _centroids = {}


def predict(image_path):
    _load_model()

    img = extract_pokemon(image_path)
    if img is None:
        return "Không đọc được ảnh", 0.0, "Không thể trích xuất đặc trưng từ ảnh."

    hog_feat, hist, fd_feat = extract_feature_components(img)
    feat = np.hstack([hog_feat, hist, fd_feat])
    feat_scaled = _scaler.transform([feat])

    pred = _model.predict(feat_scaled)[0]
    prob = _model.predict_proba(feat_scaled)[0]
    
    predicted_name = _labels[pred]
    confidence = float(max(prob))

    # Explanation Generation
    if _centroids:
        c, s, t = analyze_features((hog_feat, hist, fd_feat), predicted_name, _centroids)
        explanation = generate_explanation(predicted_name, c, s, t)
    else:
        explanation = "Không tìm thấy dữ liệu centroids để giải thích."

    return predicted_name, confidence, explanation


if __name__ == "__main__":
    path = input("Ảnh: ")
    name, conf, expl = predict(path)
    print(f"\n==== KẾT QUẢ ====\n{name} ({conf:.2f})\n\n{expl}")