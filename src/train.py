# src/train.py

from src.dataset import load_dataset
from src.model import create_base_model
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

def train(data_dir="data"):
    print("Loading data...")
    X, y, labels = load_dataset(data_dir)

    #  SCALE TRƯỚC
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # GRID SEARCH
    model = create_base_model()

    param_grid = {
        'C': [1, 10, 20, 50],
        'gamma': [0.001, 0.01, 0.1],
        'kernel': ['rbf']
    }

    grid = GridSearchCV(model, param_grid, cv=5, verbose=2, n_jobs=-1)

    print("Training with GridSearch...")
    grid.fit(X, y)

    best_model = grid.best_estimator_

    print("Best params:", grid.best_params_)

    # đánh giá
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    y_pred = best_model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    print(f" Accuracy: {acc*100:.2f}%")

    # confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.show()

    # save
    joblib.dump(best_model, "model.pkl")
    joblib.dump(labels, "labels.pkl")
    joblib.dump(scaler, "scaler.pkl")

    print("Saved model!")

if __name__ == "__main__":
    train()