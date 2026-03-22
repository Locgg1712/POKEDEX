# src/preprocess.py

import cv2

def extract_pokemon(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return None

    img = cv2.resize(img, (128, 128))

    img = cv2.GaussianBlur(img, (3,3), 0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    img = cv2.addWeighted(img, 0.8, edges, 0.2, 0)

    img = cv2.resize(img, (64, 64))

    return img