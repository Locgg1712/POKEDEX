# src/features.py

import cv2
import numpy as np
from skimage.feature import hog
from src.preprocess import auto_canny, clean_edges


# ==============================
# FOURIER DESCRIPTORS
# ==============================

def get_fourier_descriptors(img, num_descriptors=32):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Tăng cường khử nhiễu muối tiêu trước khi viền cạnh
    gray = cv2.medianBlur(gray, 3)
    # Làm mịn thêm với Gaussian
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    edges = auto_canny(blur)
    edges = clean_edges(edges)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        return np.zeros(num_descriptors)

    # ===== Lọc contour theo diện tích =====
    areas = [cv2.contourArea(c) for c in contours]
    max_area = max(areas)

    # Chỉ lấy contour đủ lớn (> 20% contour lớn nhất)
    valid_contours = [
        c for c, a in zip(contours, areas)
        if a > 0.2 * max_area
    ]

    fd_all  = []
    weights = []

    for contour in valid_contours:
        area    = cv2.contourArea(contour)   # tính area trước khi squeeze
        contour = contour.squeeze()

        if len(contour.shape) != 2 or len(contour) < 32:
            continue

        # Resample đều 128 điểm
        idx     = np.linspace(0, len(contour) - 1, 128).astype(int)
        contour = contour[idx]

        # FFT → magnitude
        contour_complex = contour[:, 0] + 1j * contour[:, 1]
        fourier_result  = np.fft.fft(contour_complex)
        fd_mag          = np.abs(fourier_result)[1:]

        # Scale invariant
        if fd_mag[0] != 0:
            fd_mag = fd_mag / fd_mag[0]

        fd = fd_mag[:num_descriptors] if len(fd_mag) >= num_descriptors \
            else np.pad(fd_mag, (0, num_descriptors - len(fd_mag)))

        fd_all.append(fd)
        weights.append(area)

    if len(fd_all) == 0:
        return np.zeros(num_descriptors)

    # ===== Weighted average theo diện tích =====
    fd_features = np.average(np.array(fd_all), axis=0,
                             weights=np.array(weights, dtype=np.float64))

    if fd_features.max() > 0:
        fd_features = fd_features / fd_features.max()

    return fd_features


# ==============================
# MAIN FEATURE EXTRACTION
# ==============================

def extract_feature_components(img):
    # ===== 1. HOG — shape/edge =====
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False
    )

    # ===== 2. HSV COLOR HISTOGRAM =====
    # H: 16 bins (màu sắc quan trọng nhất)
    # S: 8 bins, V: 8 bins
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None,
        [16, 8, 8],
        [0, 180, 0, 256, 0, 256]
    )
    hist = cv2.normalize(hist, hist, norm_type=cv2.NORM_L2).flatten()

    # ===== 3. FOURIER — DSP shape descriptor =====
    fd_feat = get_fourier_descriptors(img, num_descriptors=32)

    return hog_feat, hist, fd_feat


def extract_features(img):
    hog_feat, hist, fd_feat = extract_feature_components(img)
    # StandardScaler trong train.py cân bằng scale giữa 3 nhóm
    return np.hstack([hog_feat, hist, fd_feat])