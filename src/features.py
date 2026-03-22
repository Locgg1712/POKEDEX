# src/features.py

import cv2
import numpy as np
from skimage.feature import hog

def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hog_feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        visualize=False
    )

    hist = cv2.calcHist([img], [0,1,2], None, [16,16,16], [0,256]*3)
    hist = cv2.normalize(hist, hist).flatten()

    return np.hstack([hog_feat, hist])