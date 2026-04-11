# src/preprocess.py

import cv2
import numpy as np


def bilateral_denoise(img):
    # Bước 1: Khử nhiễu muối tiêu (salt & pepper / speckle) cực tốt bằng Median Blur
    img = cv2.medianBlur(img, 3)
    # Bước 2: Khử nhiễu Gaussian và làm mịn nhưng vẫn giữ cạnh (Tăng sigma)
    return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)


def auto_canny(gray, sigma=0.33):
    median = np.median(gray)
    lower  = int(max(0,   (1.0 - sigma) * median))
    upper  = int(min(255, (1.0 + sigma) * median))
    return cv2.Canny(gray, lower, upper)


def clean_edges(edges):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


# ==============================
# DÙNG CHO MODEL (train + predict)
# ==============================

def extract_pokemon(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    img = cv2.resize(img, (128, 128))
    img = bilateral_denoise(img)

    return cv2.resize(img, (64, 64))


# ==============================
# DÙNG CHO GUI
# ==============================

def extract_pokemon_debug(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    original = cv2.resize(img, (128, 128))
    denoised = bilateral_denoise(original)

    gray      = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    edges     = auto_canny(gray)
    edges     = clean_edges(edges)
    edges_col = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    return {
        "original": original,
        "blur":     denoised,
        "edges":    edges_col,
        "combine":  denoised,
        "final":    cv2.resize(denoised, (64, 64))
    }