import os
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml

MODEL_FILE = "digit_rf_mnist.pkl"


def load_or_train_model():
    if not os.path.exists(MODEL_FILE):
        print("Downloading MNIST...")

        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X = mnist.data.astype(np.float32) / 255.0
        y = mnist.target.astype(np.int64)

        print("Training model...")

        pca = PCA(n_components=80, random_state=42)
        X_pca = pca.fit_transform(X)

        rf = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_pca, y)

        joblib.dump((pca, rf), MODEL_FILE)
        print("Model saved.")

    return joblib.load(MODEL_FILE)
