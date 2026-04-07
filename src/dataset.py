# src/dataset.py

import os
import cv2
import numpy as np
from src.preprocess import extract_pokemon
from src.features import extract_features


# ==============================
# AUGMENTATION
# ==============================

def augment(img):
    """
    11 biến thể: hình học đa dạng + màu sắc + nhiễu.
    """
    results = [img]
    h, w    = img.shape[:2]
    center  = (w // 2, h // 2)

    # ===== Hình học =====

    # 1. Flip ngang
    results.append(cv2.flip(img, 1))

    # 2. Xoay +20°
    M = cv2.getRotationMatrix2D(center, 20, 1.0)
    results.append(cv2.warpAffine(img, M, (w, h)))

    # 3. Xoay -20°
    M = cv2.getRotationMatrix2D(center, -20, 1.0)
    results.append(cv2.warpAffine(img, M, (w, h)))

    # 4. Scale nhỏ lại 80% (zoom out)
    M = cv2.getRotationMatrix2D(center, 0, 0.8)
    results.append(cv2.warpAffine(img, M, (w, h)))

    # 5. Shear ngang nhẹ
    shear = np.float32([[1, 0.15, 0], [0, 1, 0]])
    results.append(cv2.warpAffine(img, shear, (w, h)))

    # 6. Translate phải-xuống
    M = np.float32([[1, 0, 8], [0, 1, 8]])
    results.append(cv2.warpAffine(img, M, (w, h)))

    # 7. Translate trái-lên
    M = np.float32([[1, 0, -8], [0, 1, -8]])
    results.append(cv2.warpAffine(img, M, (w, h)))

    # ===== Màu sắc =====

    # 8. Sáng hơn
    results.append(cv2.convertScaleAbs(img, alpha=1.3, beta=20))

    # 9. Tối hơn
    results.append(cv2.convertScaleAbs(img, alpha=0.7, beta=-20))

    # 10. Nhiễu Gaussian nhẹ
    noise = np.random.normal(0, 8, img.shape).astype(np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    results.append(noisy)

    return results  # 11 ảnh


# ==============================
# LOAD DATASET
# ==============================

def load_dataset(data_dir):
    X, y     = [], []
    labels   = {}
    label_id = 0

    class_names = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    for class_name in class_names:
        class_path     = os.path.join(data_dir, class_name)
        labels[label_id] = class_name

        count = 0
        for file in os.listdir(class_path):
            path = os.path.join(class_path, file)
            img  = extract_pokemon(path)
            if img is None:
                continue

            for aug in augment(img):
                feat = extract_features(aug)
                X.append(feat)
                y.append(label_id)
                count += 1

        print(f"  {class_name}: {count} ảnh (sau augment)")
        label_id += 1

    print(f"\nTổng: {len(X)} samples, {len(labels)} classes")
    return np.array(X), np.array(y), labels