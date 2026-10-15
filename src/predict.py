# src/predict.py

import os
import joblib
import numpy as np
import torch
from src.preprocess import extract_pokemon, IMAGE_SIZE
from src.features import extract_feature_components, extra_features, FEATURE_DIM
from src.explain import analyze_features, generate_explanation
from src.mlp import build_mlp

# Load model 1 lần khi import
_models = None
_labels = None
_scaler = None
_centroids = None
_model_type = None
_meta = None


def _load_model():
    global _models, _labels, _scaler, _centroids, _model_type, _meta
    if _models is None:
        model_dir = "Model"
        _labels = joblib.load(os.path.join(model_dir, "labels.pkl"))
        _scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        svm_path = os.path.join(model_dir, "model.pkl")
        mlp_paths = sorted(
            os.path.join(model_dir, name)
            for name in os.listdir(model_dir)
            if name.startswith("mlp_") and name.endswith(".pt")
        )
        mlp_path = os.path.join(model_dir, "mlp.pt")

        if (mlp_paths or os.path.exists(mlp_path)) and os.path.exists(
            os.path.join(model_dir, "meta.pkl")
        ):
            _meta = joblib.load(os.path.join(model_dir, "meta.pkl"))
            trained_image_size = _meta.get("image_size")
            if trained_image_size is not None and int(trained_image_size) != IMAGE_SIZE:
                raise RuntimeError(
                    f"Model expects image size {trained_image_size}, "
                    f"but preprocessing uses {IMAGE_SIZE}. "
                    "Set POKEDEX_IMAGE_SIZE to the training value."
                )
            mlp_paths = mlp_paths or [mlp_path]
            _models = []
            for path in mlp_paths:
                model = build_mlp(_meta["in_dim"], _meta["num_classes"])
                state_dict = torch.load(path, map_location="cpu", weights_only=True)
                model.load_state_dict(state_dict)
                model.eval()
                _models.append(model)
            _model_type = "mlp"
        elif os.path.exists(svm_path):
            _models = [joblib.load(svm_path)]
            _model_type = "svm"
        else:
            raise FileNotFoundError(
                "Không tìm thấy Model/model.pkl, Model/mlp.pt hoặc Model/mlp_*.pt."
            )

        try:
            _centroids = joblib.load(os.path.join(model_dir, "centroids.pkl"))
        except FileNotFoundError:
            _centroids = {}


def predict(image_path):
    _load_model()

    img = extract_pokemon(image_path)
    if img is None:
        return "Không đọc được ảnh", 0.0, "Không thể trích xuất đặc trưng từ ảnh."

    def make_feature(image):
        components = extract_feature_components(image)
        hog_feat, hist, fd_feat = components
        full = np.hstack([hog_feat, hist, fd_feat, extra_features(image)])
        if full.shape[0] != FEATURE_DIM:
            raise RuntimeError(f"Unexpected feature dimension: {full.shape[0]}")
        expected_dim = int(_meta.get("in_dim", FEATURE_DIM)) if _meta else FEATURE_DIM
        if expected_dim == 2820:
            return full[:2820], components
        cols = _meta.get("cols") if _meta else None
        selected = full[np.asarray(cols, dtype=np.int64)] if cols else full
        return selected, components

    feat, components = make_feature(img)
    flipped_feat, _ = make_feature(np.ascontiguousarray(np.fliplr(img)))
    feat_scaled = _scaler.transform([feat])
    flipped_scaled = _scaler.transform([flipped_feat])
    if _meta and _meta.get("w") is not None:
        weight = np.asarray(_meta["w"])
        feat_scaled = feat_scaled * weight
        flipped_scaled = flipped_scaled * weight

    if _model_type == "svm":
        prob = (
            _models[0].predict_proba(feat_scaled)[0]
            + _models[0].predict_proba(flipped_scaled)[0]
        ) / 2
    else:
        with torch.no_grad():
            tensor = torch.from_numpy(feat_scaled).float()
            flipped_tensor = torch.from_numpy(flipped_scaled).float()
            probabilities = [
                (
                    torch.softmax(model(tensor), dim=1).numpy()[0]
                    + torch.softmax(model(flipped_tensor), dim=1).numpy()[0]
                ) / 2
                for model in _models
            ]
        prob = np.mean(probabilities, axis=0)
    pred = int(np.argmax(prob))

    predicted_name = _labels[pred]
    confidence = float(max(prob))

    # Explanation Generation
    if _centroids:
        c, s, t = analyze_features(components, predicted_name, _centroids)
        explanation = generate_explanation(predicted_name, c, s, t)
    else:
        explanation = "Không tìm thấy dữ liệu centroids để giải thích."

    return predicted_name, confidence, explanation


if __name__ == "__main__":
    path = input("Ảnh: ")
    name, conf, expl = predict(path)
    print(f"\n==== KẾT QUẢ ====\n{name} ({conf:.2f})\n\n{expl}")