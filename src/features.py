import cv2
import numpy as np
from skimage.feature import hog


# ==============================
# FOURIER DESCRIPTORS (FIX DSP)
# ==============================
def get_fourier_descriptors(img, num_descriptors=32):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #  lọc nhiễu trước khi tìm biên
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        return np.zeros(num_descriptors)

    contour = max(contours, key=cv2.contourArea)

    # ==============================
    #   RESAMPLE contour (cực quan trọng)
    # ==============================
    contour = contour.squeeze()

    if len(contour.shape) != 2:
        return np.zeros(num_descriptors)

    # lấy đều 128 điểm
    idx = np.linspace(0, len(contour) - 1, 128).astype(int)
    contour = contour[idx]

    # chuyển sang số phức
    contour_complex = contour[:, 0] + 1j * contour[:, 1]

    # FFT
    fourier_result = np.fft.fft(contour_complex)

    # magnitude
    fd_mag = np.abs(fourier_result)

    # ==============================
    # ✅ FIX 3: chuẩn hóa DSP
    # ==============================
    fd_mag = fd_mag[1:]  # bỏ DC (tịnh tiến)

    if fd_mag[0] != 0:
        fd_mag = fd_mag / fd_mag[0]  # scale invariant

    # lấy tần số thấp (shape chính)
    if len(fd_mag) >= num_descriptors:
        fd_features = fd_mag[:num_descriptors]
    else:
        fd_features = np.pad(fd_mag, (0, num_descriptors - len(fd_mag)), 'constant')

    return fd_features


# ==============================
# MAIN FEATURE
# ==============================
def extract_features(img):

    # ==============================
    # 1. HOG (SHAPE)
    # ==============================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hog_feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False
    )

    # ==============================
    # 2. COLOR (FIX OVERFIT)
    # ==============================
    # ✅ giảm từ 32 xuống 16
    hist = cv2.calcHist([img], [0, 1, 2], None, [16, 16, 16], [0, 256]*3)
    hist = cv2.normalize(hist, hist).flatten()

    weighted_hist = hist * 8.0

    # ==============================
    # 3. FOURIER (DSP)
    # ==============================
    fd_feat = get_fourier_descriptors(img, num_descriptors=32)
    weighted_fd = fd_feat * 2.5

    # ==============================
    # CONCAT
    # ==============================
    return np.hstack([hog_feat, weighted_hist, weighted_fd])