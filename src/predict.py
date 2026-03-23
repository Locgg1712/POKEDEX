# src/predict.py

import joblib
import numpy as np

from src.preprocess import extract_pokemon
from src.features import extract_features


def predict(image_path):
    model = joblib.load("model.pkl")
    labels = joblib.load("labels.pkl")
    scaler = joblib.load("scaler.pkl")

    img = extract_pokemon(image_path)
    feat = extract_features(img)

    feat = scaler.transform([feat])

    pred = model.predict(feat)[0]
    prob = model.predict_proba(feat)[0]

    return labels[pred], max(prob)


if __name__ == "__main__":
    path = input("Ảnh: ")
    name, conf = predict(path)

    print(name, f"{conf:.2f}")