# src/dataset.py

import os
import numpy as np
import cv2
from src.preprocess import extract_pokemon
from src.features import extract_features

def augment(img):
    return [img, cv2.flip(img, 1)]

def load_dataset(data_dir):
    X, y = [], []
    labels = {}
    label_id = 0

    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        labels[label_id] = class_name

        for file in os.listdir(class_path):
            path = os.path.join(class_path, file)

            img = extract_pokemon(path)
            if img is None:
                continue

            for aug in augment(img):
                feat = extract_features(aug)
                X.append(feat)
                y.append(label_id)

        label_id += 1

    return np.array(X), np.array(y), labels