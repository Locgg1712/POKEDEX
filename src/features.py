# src/features.py

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from src.preprocess import auto_canny, clean_edges
from src.preprocess import IMAGE_SIZE

HOG_DIM = 1764
HOG_PIXELS_PER_CELL = (16, 16) if IMAGE_SIZE == 128 else (8, 8)
HIST_DIM = 1024
FD_DIM = 32
SPATIAL_HIST_DIM = 288
MOMENTS_DIM = 9
LBP_DIM = 40
FEATURE_DIM = HOG_DIM + HIST_DIM + FD_DIM + SPATIAL_HIST_DIM + MOMENTS_DIM + LBP_DIM
FEATURE_GROUPS = {
    "hog": (0, HOG_DIM),
    "hist": (HOG_DIM, HOG_DIM + HIST_DIM),
    "fd": (HOG_DIM + HIST_DIM, HOG_DIM + HIST_DIM + FD_DIM),
    "sp_hist": (2820, 3108),
    "moments": (3108, 3117),
    "lbp": (3117, FEATURE_DIM),
}


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
        pixels_per_cell=HOG_PIXELS_PER_CELL,
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


def spatial_color_hist(img, grid=3, bins=(8, 4)):
    """Local H-S histograms, normalized independently for each grid cell."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    height, width = hsv.shape[:2]
    features = []
    for row in range(grid):
        for col in range(grid):
            y0, y1 = row * height // grid, (row + 1) * height // grid
            x0, x1 = col * width // grid, (col + 1) * width // grid
            cell = hsv[y0:y1, x0:x1]
            hist = cv2.calcHist(
                [cell], [0, 1], None, list(bins),
                [0, 180, 0, 256]
            )
            hist = cv2.normalize(hist, hist, norm_type=cv2.NORM_L2).flatten()
            features.append(hist)
    return np.concatenate(features).astype(np.float32)


def color_moments(img):
    """Mean, standard deviation and skew for each Lab channel."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    values = []
    for channel in range(3):
        flattened = lab[:, :, channel].reshape(-1)
        values.extend((
            float(np.mean(flattened)),
            float(np.std(flattened)),
            float(np.mean(((flattened - np.mean(flattened)) /
                           (np.std(flattened) + 1e-8)) ** 3)),
        ))
    return np.nan_to_num(np.asarray(values, dtype=np.float32))


def lbp_hist(img, grid=2, P=8, R=1):
    """Uniform LBP histograms over a 2x2 spatial grid."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    height, width = gray.shape[:2]
    features = []
    for row in range(grid):
        for col in range(grid):
            y0, y1 = row * height // grid, (row + 1) * height // grid
            x0, x1 = col * width // grid, (col + 1) * width // grid
            hist, _ = np.histogram(lbp[y0:y1, x0:x1], bins=np.arange(P + 3),
                                   range=(0, P + 2))
            hist = hist.astype(np.float32)
            norm = np.linalg.norm(hist)
            features.append(hist / norm if norm > 0 else hist)
    return np.concatenate(features)


def extra_features(img):
    return np.hstack((
        spatial_color_hist(img),
        color_moments(img),
        lbp_hist(img),
    )).astype(np.float32)


def extract_features(img):
    hog_feat, hist, fd_feat = extract_feature_components(img)
    features = np.hstack([hog_feat, hist, fd_feat, extra_features(img)])
    if features.shape[0] != FEATURE_DIM:
        raise RuntimeError(f"Unexpected feature dimension: {features.shape[0]}")
    return features.astype(np.float32)