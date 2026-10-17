# src/predict.py

import os
import joblib
import numpy as np
import torch

from src.preprocess import extract_pokemon, IMAGE_SIZE
from src.features import (
    extract_features,
    extract_feature_components,
    extra_features,
    spatial_color_hist,
    color_moments,
    lbp_hist,
    shape_geometry,
    hu_moments,
    FEATURE_DIM,
)
from src.explain import analyze_features, generate_explanation
from src.mlp import build_mlp, build_legacy_mlp, taxonomy_rerank
from src.evolution import evolution_rerank, get_family_info

_models = None
_labels = None
_scaler = None
_centroids = None
_model_type = None
_meta = None
_svm_blend = None


def _load_model():
    """Tải model và metadata từ thư mục Model/ (hỗ trợ cả Self-Taxonomy DeepMLP, Cosine Head, SVM Blending, Legacy MLP và SVM)."""
    global _models, _labels, _scaler, _centroids, _model_type, _meta, _svm_blend
    if _models is not None:
        return

    model_dir = "Model"
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Thư mục '{model_dir}' không tồn tại.")

    labels_path = os.path.join(model_dir, "labels.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    if not os.path.exists(labels_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Thiếu labels.pkl hoặc scaler.pkl trong Model/")

    _labels = joblib.load(labels_path)
    _scaler = joblib.load(scaler_path)

    svm_path = os.path.join(model_dir, "model.pkl")
    svm_blend_path = os.path.join(model_dir, "svm_blend.pkl")
    if os.path.exists(svm_blend_path):
        _svm_blend = joblib.load(svm_blend_path)
    else:
        _svm_blend = None

    mlp_paths = sorted(
        os.path.join(model_dir, name)
        for name in os.listdir(model_dir)
        if name.startswith("mlp_") and name.endswith(".pt")
    )
    mlp_path = os.path.join(model_dir, "mlp.pt")

    if (mlp_paths or os.path.exists(mlp_path)) and os.path.exists(os.path.join(model_dir, "meta.pkl")):
        _meta = joblib.load(os.path.join(model_dir, "meta.pkl"))
        trained_image_size = _meta.get("image_size")
        if trained_image_size is not None and int(trained_image_size) != IMAGE_SIZE:
            print(
                f"[Cảnh báo] Model được huấn luyện với IMAGE_SIZE={trained_image_size}, "
                f"nhưng cài đặt hiện tại là {IMAGE_SIZE}."
            )

        mlp_paths = mlp_paths or [mlp_path]
        _models = []
        in_dim = int(_meta["in_dim"])
        num_classes = int(_meta["num_classes"])
        n_clusters = int(_meta.get("n_clusters", 0))

        for path in mlp_paths:
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
            use_cosine_head = bool(_meta.get("use_cosine_head", False)) or any("_log_tau" in k for k in state_dict.keys())
            try:
                model = build_mlp(in_dim, num_classes, deep=True, n_clusters=n_clusters, use_cosine_head=use_cosine_head)
                model.load_state_dict(state_dict)
            except Exception:
                try:
                    model = build_mlp(in_dim, num_classes, deep=True, n_clusters=0, use_cosine_head=use_cosine_head)
                    model.load_state_dict(state_dict)
                except Exception:
                    model = build_legacy_mlp(in_dim, num_classes)
                    model.load_state_dict(state_dict)

            model.eval()
            _models.append(model)
        _model_type = "mlp"

    elif os.path.exists(svm_path):
        _models = [joblib.load(svm_path)]
        _model_type = "svm"
    else:
        raise FileNotFoundError("Không tìm thấy Model/model.pkl, Model/mlp.pt hoặc Model/mlp_*.pt.")

    try:
        _centroids = joblib.load(os.path.join(model_dir, "centroids.pkl"))
    except FileNotFoundError:
        _centroids = {}


def _align_features(img):
    """Trích xuất vector đặc trưng và căn chỉnh kích thước tương thích với mọi phiên bản checkpoint."""
    components = extract_feature_components(img)
    hog_feat, fine_hog, hist, fd_feat = components

    expected_dim = int(_meta.get("in_dim", FEATURE_DIM)) if _meta else FEATURE_DIM
    full = np.hstack([hog_feat, fine_hog, hist, fd_feat, extra_features(img, include_stage_morph=True)])

    if _meta and _meta.get("cols") is not None:
        cols = np.asarray(_meta["cols"], dtype=np.int64)
        if max(cols) < full.shape[0]:
            return full[cols].astype(np.float32), components

    if expected_dim == 3590:
        feat_3590 = np.hstack([hog_feat, fine_hog, hist, fd_feat, extra_features(img, include_stage_morph=False)])
        return feat_3590.astype(np.float32), components
    elif expected_dim == 2820:
        return np.hstack([hog_feat, hist, fd_feat]).astype(np.float32), components
    elif expected_dim == 3157:
        return np.hstack([
            hog_feat, hist, fd_feat,
            spatial_color_hist(img),
            color_moments(img),
            lbp_hist(img),
        ]).astype(np.float32), components
    elif expected_dim == 3498:
        return np.hstack([
            hog_feat, fine_hog, hist, fd_feat,
            spatial_color_hist(img),
            color_moments(img),
            lbp_hist(img),
            shape_geometry(img),
            hu_moments(img),
        ]).astype(np.float32), components

    return full.astype(np.float32), components


def predict(image_path, apply_evolution_rerank=True):
    """
    Dự đoán loài Pokémon từ đường dẫn ảnh.
    Trả về: (predicted_name, confidence, explanation, top3)
    """
    _load_model()

    img = extract_pokemon(image_path)
    if img is None:
        return "Không đọc được ảnh", 0.0, "Không thể đọc hoặc trích xuất Pokémon từ ảnh.", []

    # Trích xuất đặc trưng cho ảnh gốc và ảnh lật ngang (Flip TTA)
    feat, components = _align_features(img)
    flipped_img = np.ascontiguousarray(np.fliplr(img))
    flipped_feat, _ = _align_features(flipped_img)

    feat_scaled = _scaler.transform([feat])
    flipped_scaled = _scaler.transform([flipped_feat])

    if _meta and _meta.get("w") is not None:
        weight = np.asarray(_meta["w"], dtype=np.float32)
        feat_scaled = feat_scaled * weight
        flipped_scaled = flipped_scaled * weight

    if _model_type == "svm":
        prob = (
            _models[0].predict_proba(feat_scaled)[0]
            + _models[0].predict_proba(flipped_scaled)[0]
        ) / 2.0
    else:
        n_clusters = int(_meta.get("n_clusters", 0)) if _meta else 0
        class_to_cluster = _meta.get("class_to_cluster") if _meta else None

        with torch.no_grad():
            tensor = torch.from_numpy(feat_scaled).float()
            flipped_tensor = torch.from_numpy(flipped_scaled).float()

            if n_clusters > 0 and class_to_cluster is not None:
                sp_list, cl_list = [], []
                for model in _models:
                    sp1, cl1 = model(tensor, return_cluster=True)
                    sp2, cl2 = model(flipped_tensor, return_cluster=True)
                    sp_list.append(0.5 * (torch.softmax(sp1, dim=1).numpy()[0] + torch.softmax(sp2, dim=1).numpy()[0]))
                    cl_list.append(0.5 * (torch.softmax(cl1, dim=1).numpy()[0] + torch.softmax(cl2, dim=1).numpy()[0]))
                prob = np.mean(sp_list, axis=0)
                cl_prob = np.mean(cl_list, axis=0)
                prob = taxonomy_rerank(prob, cl_prob, np.asarray(class_to_cluster), beta=0.25)
            else:
                probabilities = [
                    0.5 * (
                        torch.softmax(model(tensor), dim=1).numpy()[0]
                        + torch.softmax(model(flipped_tensor), dim=1).numpy()[0]
                    )
                    for model in _models
                ]
                prob = np.mean(probabilities, axis=0)

        # Hỗ trợ SVM Blending (nếu checkpoint có lưu svm_blend.pkl)
        if _svm_blend is not None:
            svm_prob = (
                _svm_blend.predict_proba(feat_scaled)[0]
                + _svm_blend.predict_proba(flipped_scaled)[0]
            ) / 2.0
            alpha_svm = float(_meta.get("alpha_svm", 0.15)) if _meta else 0.15
            prob = (1.0 - alpha_svm) * prob + alpha_svm * svm_prob

        if apply_evolution_rerank:
            prob = evolution_rerank(prob, _labels, alpha=0.35, uncertainty_threshold=0.60)

    top_indices = np.argsort(prob)[::-1]
    top3 = [(str(_labels[idx]), float(prob[idx])) for idx in top_indices[:3]]

    pred_idx = int(top_indices[0])
    predicted_name = str(_labels[pred_idx])
    confidence = float(prob[pred_idx])

    fam_info = get_family_info(predicted_name)
    family_suffix = ""
    if fam_info:
        family_suffix = f"\n• Dòng họ: {fam_info['family_name']} (Giai đoạn {fam_info['stage']}/{fam_info['total_stages']})"

    if _centroids and predicted_name in _centroids:
        c, s, t = analyze_features(components, predicted_name, _centroids)
        explanation = generate_explanation(predicted_name, c, s, t) + family_suffix
    else:
        types_str = ", ".join(fam_info["types"]) if fam_info else "Chưa rõ"
        explanation = (
            f"LÝ DO NHẬN DIỆN ({predicted_name}):\n\n"
            f"• Phân bố đặc trưng biên (HOG), tần số (Fourier/Gabor) và màu sắc (HSV) khớp mạnh nhất với nguyên mẫu.\n"
            f"• Hệ nguyên tố: {types_str}.{family_suffix}"
        )

    return predicted_name, confidence, explanation, top3


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = input("Nhập đường dẫn ảnh: ").strip()

    name, conf, expl, top3 = predict(path)
    print(f"\n==== KẾT QUẢ DỰ ĐOÁN ====")
    print(f"Top 1: {name} (Độ tin cậy: {conf*100:.1f}%)")
    print("\nTop 3 Ứng viên:")
    for rank, (cand, p) in enumerate(top3, 1):
        print(f"  {rank}. {cand:15s}: {p*100:.1f}%")
    print(f"\n{expl}")