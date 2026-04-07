# src/preprocess.py

import cv2
import numpy as np


def bilateral_denoise(img):
    """
    Khử nhiễu giữ nguyên cạnh.
    KHÔNG trộn edge vào ảnh → bảo toàn màu sắc cho color feature.
    """
    return cv2.bilateralFilter(img, d=9, sigmaColor=60, sigmaSpace=60)


def auto_canny(gray, sigma=0.33):
    """
    Canny tự động theo median pixel.
    Chỉ dùng trong features.py để extract shape — KHÔNG trộn vào ảnh.
    """
    median = np.median(gray)
    lower = int(max(0,   (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    return cv2.Canny(gray, lower, upper)


def clean_edges(edges):
    """Morphology nối viền đứt — dùng trong features.py."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


def segment_foreground(img):
    """
    Tách foreground bằng Otsu — có 3 lớp fallback:
    1. edge_ratio > 0.25 → nền phức tạp → dùng ảnh gốc
    2. mask_ratio < 0.1  → foreground quá nhỏ → dùng ảnh gốc
    3. mask_ratio > 0.9  → Otsu thất bại → dùng ảnh gốc
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ===== EDGE DENSITY CHECK =====
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])

    # Nền phức tạp → skip segmentation
    if edge_ratio > 0.25:
        return img

    # ===== OTSU =====
    _, mask_otsu = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # ===== CHECK MASK QUALITY =====
    mask_ratio = np.sum(mask_otsu > 0) / (gray.shape[0] * gray.shape[1])

    if mask_ratio < 0.1 or mask_ratio > 0.9:
        return img

    mask = cv2.medianBlur(mask_otsu, 5)
    return cv2.bitwise_and(img, img, mask=mask)


# ==============================
# DÙNG CHO MODEL (train + predict)
# ==============================

def extract_pokemon(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    img = cv2.resize(img, (128, 128))
    img = bilateral_denoise(img)
    img = segment_foreground(img)

    return cv2.resize(img, (64, 64))


# ==============================
# DÙNG CHO GUI (hiển thị từng bước)
# ==============================

def extract_pokemon_debug(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    original  = cv2.resize(img, (128, 128))
    denoised  = bilateral_denoise(original)
    segmented = segment_foreground(denoised)

    # Edge chỉ để visualize trong GUI
    gray      = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    edges     = auto_canny(gray)
    edges     = clean_edges(edges)
    edges_col = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    return {
        "original": original,
        "blur":     denoised,
        "edges":    edges_col,
        "combine":  segmented,
        "final":    cv2.resize(segmented, (64, 64))
    }