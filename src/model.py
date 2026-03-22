# src/model.py

from sklearn.svm import SVC

def create_base_model():
    return SVC(probability=True, class_weight='balanced')