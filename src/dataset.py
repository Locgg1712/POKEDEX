# src/dataset.py

import os
import cv2
import numpy as np
from typing import overload, Literal
from src.preprocess import extract_pokemon, IMAGE_SIZE
from src.features import extract_features


# ==============================
# AUGMENTATION (FIXED BORDER ARTIFACTS)
# ==============================

def augment(img):
    """
    16 mẫu (1 ảnh gốc + 15 biến thể):
    Sử dụng BORDER_REFLECT_101 cho mọi phép biến đổi hình học
    để ngăn chặn viền đen nhân tạo làm sai lệch Canny Edge và Fourier Descriptors.
    """
    results = [img]  # index 0 luôn là ảnh gốc (Original)
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    border = cv2.BORDER_REFLECT_101

    # ===== Hình học =====

    # 1. Flip ngang
    results.append(cv2.flip(img, 1))

    # 2. Xoay +20°
    M = cv2.getRotationMatrix2D(center, 20, 1.0)
    results.append(cv2.warpAffine(img, M, (w, h), borderMode=border))

    # 3. Xoay -20°
    M = cv2.getRotationMatrix2D(center, -20, 1.0)
    results.append(cv2.warpAffine(img, M, (w, h), borderMode=border))

    # 4. Scale nhỏ lại 85% (zoom out nhẹ, giữ cấu trúc biên)
    M = cv2.getRotationMatrix2D(center, 0, 0.85)
    results.append(cv2.warpAffine(img, M, (w, h), borderMode=border))

    # 5. Shear ngang nhẹ
    shear = np.float32([[1, 0.12, 0], [0, 1, 0]])
    results.append(cv2.warpAffine(img, shear, (w, h), borderMode=border))

    # 6. Translate phải-xuống
    shift = IMAGE_SIZE // 10
    M = np.float32([[1, 0, shift], [0, 1, shift]])
    results.append(cv2.warpAffine(img, M, (w, h), borderMode=border))

    # 7. Translate trái-lên
    M = np.float32([[1, 0, -shift], [0, 1, -shift]])
    results.append(cv2.warpAffine(img, M, (w, h), borderMode=border))

    # ===== Màu sắc =====

    # 8. Sáng hơn
    results.append(cv2.convertScaleAbs(img, alpha=1.25, beta=15))

    # 9. Tối hơn
    results.append(cv2.convertScaleAbs(img, alpha=0.75, beta=-15))

    # 10. Nhiễu Gaussian nhẹ
    noise = np.random.normal(0, 6, img.shape).astype(np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    results.append(noisy)

    # ===== Biến thể bổ sung =====

    # 11. Xoay nhẹ +10°
    M = cv2.getRotationMatrix2D(center, 10, 1.0)
    results.append(cv2.warpAffine(img, M, (w, h), borderMode=border))

    # 12. Xoay nhẹ -10°
    M = cv2.getRotationMatrix2D(center, -10, 1.0)
    results.append(cv2.warpAffine(img, M, (w, h), borderMode=border))

    # 13. Zoom nhẹ vào trung tâm 1.12x
    M = cv2.getRotationMatrix2D(center, 0, 1.12)
    results.append(cv2.warpAffine(img, M, (w, h), borderMode=border))

    # 14. Thay đổi saturation và brightness trong HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.18, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.08 + 5, 0, 255)
    results.append(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR))

    # 15. Blur nhẹ mô phỏng ảnh chụp thiếu nét
    results.append(cv2.GaussianBlur(img, (3, 3), 0))

    return results


# ==============================
# LOAD DATASET
# ==============================

@overload
def load_dataset(data_dir, return_groups: Literal[True]) -> tuple[
    np.ndarray, np.ndarray, dict[int, str], np.ndarray
]:
    ...


@overload
def load_dataset(data_dir, return_groups: Literal[False] = False) -> tuple[
    np.ndarray, np.ndarray, dict[int, str]
]:
    ...


def load_dataset(data_dir, return_groups=False):
    X, y, groups = [], [], []
    labels = {}
    label_id = 0

    class_names = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        labels[label_id] = class_name

        count = 0
        for file in sorted(os.listdir(class_path)):
            path = os.path.join(class_path, file)
            img = extract_pokemon(path)
            if img is None:
                continue

            group_id = f"{class_name}/{file}"
            for aug in augment(img):
                feat = extract_features(aug)
                X.append(feat)
                y.append(label_id)
                groups.append(group_id)
                count += 1

        print(f"  {class_name}: {count} ảnh (sau augment)")
        label_id += 1

    print(f"\nTổng: {len(X)} samples, {len(labels)} classes")
    if return_groups:
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), labels, np.array(groups)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), labels