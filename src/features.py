# src/features.py

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from src.preprocess import auto_canny, clean_edges
from src.preprocess import IMAGE_SIZE

# ==============================================================================
# FEATURE DIMENSIONS CONFIGURATION
# ==============================================================================

HOG_PIXELS_PER_CELL = (16, 16) if IMAGE_SIZE == 128 else (8, 8)
HOG_DIM = (
    (IMAGE_SIZE // HOG_PIXELS_PER_CELL[0] - 1)
    * (IMAGE_SIZE // HOG_PIXELS_PER_CELL[1] - 1) * 36
)
HOG_FINE_PIXELS_PER_CELL = (8, 8) if IMAGE_SIZE == 128 else (16, 16)
HOG_FINE_DIM = (
    (IMAGE_SIZE // HOG_FINE_PIXELS_PER_CELL[0] - 1)
    * (IMAGE_SIZE // HOG_FINE_PIXELS_PER_CELL[1] - 1) * 36
)

HIST_DIM = 1024
FD_DIM = 32
SPATIAL_HIST_DIM = 288
MOMENTS_DIM = 9
LBP_DIM = 40
GEOMETRY_DIM = 10
HU_DIM = 7

# New DSP Modules
GLCM_DIM = 40          # 5 props * 2 distances * 4 angles
GABOR_DIM = 36         # 3 frequencies * 6 orientations * 2 stats
FOURIER_2D_DIM = 16    # 8 radial frequency bands + 8 directional wedges

FEATURE_DIM = (
    HOG_DIM + HOG_FINE_DIM + HIST_DIM + FD_DIM + SPATIAL_HIST_DIM
    + MOMENTS_DIM + LBP_DIM + GEOMETRY_DIM + HU_DIM
    + GLCM_DIM + GABOR_DIM + FOURIER_2D_DIM
)

_offset = 0
FEATURE_GROUPS = {}
for _name, _size in (
    ("hog", HOG_DIM),
    ("hog_fine", HOG_FINE_DIM),
    ("hist", HIST_DIM),
    ("fd", FD_DIM),
    ("sp_hist", SPATIAL_HIST_DIM),
    ("moments", MOMENTS_DIM),
    ("lbp", LBP_DIM),
    ("geometry", GEOMETRY_DIM),
    ("hu", HU_DIM),
    ("glcm", GLCM_DIM),
    ("gabor", GABOR_DIM),
    ("fourier_2d", FOURIER_2D_DIM),
):
    FEATURE_GROUPS[_name] = (_offset, _offset + _size)
    _offset += _size


# ==============================================================================
# 1. FOURIER CONTOUR DESCRIPTORS (1D DSP)
# ==============================================================================

def get_fourier_descriptors(img, num_descriptors=32):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    edges = auto_canny(blur)
    edges = clean_edges(edges)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        return np.zeros(num_descriptors, dtype=np.float32)

    areas = [cv2.contourArea(c) for c in contours]
    max_area = max(areas)

    valid_contours = [
        c for c, a in zip(contours, areas)
        if a > 0.2 * max_area
    ]

    fd_all = []
    weights = []

    for contour in valid_contours:
        area = cv2.contourArea(contour)
        contour = contour.squeeze()

        if len(contour.shape) != 2 or len(contour) < 32:
            continue

        idx = np.linspace(0, len(contour) - 1, 128).astype(int)
        contour = contour[idx]

        contour_complex = contour[:, 0] + 1j * contour[:, 1]
        fourier_result = np.fft.fft(contour_complex)
        fd_mag = np.abs(fourier_result)[1:]

        if fd_mag[0] != 0:
            fd_mag = fd_mag / fd_mag[0]

        fd = fd_mag[:num_descriptors] if len(fd_mag) >= num_descriptors \
            else np.pad(fd_mag, (0, num_descriptors - len(fd_mag)))

        fd_all.append(fd)
        weights.append(area)

    if len(fd_all) == 0:
        return np.zeros(num_descriptors, dtype=np.float32)

    fd_features = np.average(np.array(fd_all), axis=0,
                             weights=np.array(weights, dtype=np.float64))

    if fd_features.max() > 0:
        fd_features = fd_features / fd_features.max()

    return fd_features.astype(np.float32)


# ==============================================================================
# 2. CORE FEATURE COMPONENTS (HOG, HSV, FD)
# ==============================================================================

def extract_feature_components(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=(2, 2),
        visualize=False
    )
    fine_hog = hog(
        gray,
        orientations=9,
        pixels_per_cell=HOG_FINE_PIXELS_PER_CELL,
        cells_per_block=(2, 2),
        visualize=False
    )

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None,
        [16, 8, 8],
        [0, 180, 0, 256, 0, 256]
    )
    hist = cv2.normalize(hist, hist, norm_type=cv2.NORM_L2).flatten()

    fd_feat = get_fourier_descriptors(img, num_descriptors=FD_DIM)

    return hog_feat, fine_hog, hist, fd_feat


# ==============================================================================
# 3. SPATIAL, COLOR, AND LBP FEATURES
# ==============================================================================

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
        mean_val = float(np.mean(flattened))
        std_val = float(np.std(flattened))
        skew_val = float(np.mean(((flattened - mean_val) / (std_val + 1e-8)) ** 3))
        values.extend((mean_val, std_val, skew_val))
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
    return np.concatenate(features).astype(np.float32)


# ==============================================================================
# 4. SHAPE GEOMETRY AND HU MOMENTS
# ==============================================================================

def _largest_contour_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = clean_edges(auto_canny(cv2.GaussianBlur(gray, (5, 5), 0)))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(gray.shape, dtype=np.uint8), None
    contour = max(contours, key=cv2.contourArea)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    return mask, contour


def shape_geometry(img):
    mask, contour = _largest_contour_mask(img)
    if contour is None:
        return np.zeros(GEOMETRY_DIM, dtype=np.float32)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    x, y, width, height = cv2.boundingRect(contour)
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    image_area = float(mask.shape[0] * mask.shape[1])
    values = np.array([
        area / image_area,
        perimeter / (2.0 * (mask.shape[0] + mask.shape[1])),
        width / mask.shape[1],
        height / mask.shape[0],
        width / max(height, 1),
        area / max(width * height, 1),
        area / max(hull_area, 1.0),
        (4.0 * np.pi * area) / max(perimeter * perimeter, 1e-8),
        float(np.count_nonzero(mask)) / image_area,
        float(cv2.arcLength(cv2.convexHull(contour), True)) /
        (2.0 * (mask.shape[0] + mask.shape[1])),
    ], dtype=np.float32)
    return np.nan_to_num(values)


def hu_moments(img):
    mask, _ = _largest_contour_mask(img)
    moments = cv2.moments(mask)
    hu = cv2.HuMoments(moments).flatten()
    return (np.sign(hu) * np.log10(np.abs(hu) + 1e-12)).astype(np.float32)


# ==============================================================================
# 5. NEW DSP TEXTURE & FREQUENCY MODULES (GLCM, GABOR, FOURIER 2D)
# ==============================================================================

def glcm_features(img, distances=(1, 3), angles=(0, np.pi/4, np.pi/2, 3*np.pi/4)):
    """
    GLCM texture features: contrast, dissimilarity, homogeneity, energy, correlation.
    Cơ sở DSP: Ma trận đồng xuất mức xám thống kê bậc 2 trên tín hiệu ảnh.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_quantized = (gray // 8).astype(np.uint8)  # 32 mức xám
    glcm = graycomatrix(gray_quantized, distances=list(distances), angles=list(angles),
                        levels=32, symmetric=True, normed=True)
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    feats = []
    for prop in props:
        feats.extend(graycoprops(glcm, prop).flatten())
    feats = np.array(feats, dtype=np.float32)
    return np.nan_to_num(feats)


def gabor_features(img, frequencies=(0.1, 0.2, 0.4), orientations=6):
    """
    Gabor filter bank: phân tích tần số và hướng đa thang đo.
    Cơ sở DSP: Tích bao Gaussian và sóng điều hòa sin.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    feats = []
    for freq in frequencies:
        for theta_idx in range(orientations):
            theta = theta_idx * np.pi / orientations
            kernel = cv2.getGaborKernel(
                (21, 21), sigma=3.0, theta=theta,
                lambd=1.0 / freq, gamma=0.5, psi=0
            )
            response = cv2.filter2D(gray, cv2.CV_64F, kernel)
            feats.extend([float(response.mean()), float(response.var())])
    feats = np.array(feats, dtype=np.float32)
    norm = np.linalg.norm(feats)
    if norm > 0:
        feats = feats / norm
    return np.nan_to_num(feats)


def fourier_2d_features(img, n_rings=8, n_wedges=8):
    """
    Phổ biên độ DFT 2D: Năng lượng theo dải tần số bán kính và hướng.
    Cơ sở DSP: Biến đổi Fourier 2 chiều rời rạc trên ảnh mức xám.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.log1p(np.abs(f_shift))

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    max_r = min(cy, cx)

    # 1. Radial rings
    radial = []
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    for i in range(n_rings):
        r_in = int(i * max_r / n_rings)
        r_out = int((i + 1) * max_r / n_rings)
        mask = (dist >= r_in) & (dist < r_out)
        radial.append(float(magnitude[mask].mean()) if mask.any() else 0.0)

    # 2. Angular wedges
    angular = []
    angles = np.arctan2(Y - cy, X - cx)
    for i in range(n_wedges):
        a_min = -np.pi + i * 2 * np.pi / n_wedges
        a_max = a_min + 2 * np.pi / n_wedges
        mask = (angles >= a_min) & (angles < a_max)
        angular.append(float(magnitude[mask].mean()) if mask.any() else 0.0)

    feats = np.array(radial + angular, dtype=np.float32)
    norm = np.linalg.norm(feats)
    if norm > 0:
        feats = feats / norm
    return np.nan_to_num(feats)


# ==============================================================================
# 6. FEATURE AGGREGATION
# ==============================================================================

def extra_features(img):
    """Tập hợp các đặc trưng mở rộng (color, texture, shape, dsp)."""
    return np.hstack((
        spatial_color_hist(img),
        color_moments(img),
        lbp_hist(img),
        shape_geometry(img),
        hu_moments(img),
        glcm_features(img),
        gabor_features(img),
        fourier_2d_features(img),
    )).astype(np.float32)


def extract_features(img):
    """Trích xuất toàn bộ vector đặc trưng DSP cho ảnh đầu vào."""
    hog_feat, fine_hog, hist, fd_feat = extract_feature_components(img)
    features = np.hstack([hog_feat, fine_hog, hist, fd_feat, extra_features(img)])
    if features.shape[0] != FEATURE_DIM:
        raise RuntimeError(f"Unexpected feature dimension: {features.shape[0]} (expected {FEATURE_DIM})")
    return features.astype(np.float32)