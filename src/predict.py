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

    probs = model.predict_proba(feat)[0]
    top3 = np.argsort(probs)[-3:][::-1]

    return [(labels[i], probs[i]) for i in top3]

if __name__ == "__main__":
    path = input("Ảnh: ")
    results = predict(path)

    for name, p in results:
        print(name, f"{p:.2f}")