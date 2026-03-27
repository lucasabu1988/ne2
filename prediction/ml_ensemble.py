import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

class MLEnsemble:
    def __init__(self):
        self.models = {
            "xgboost": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric="logloss"),
            "random_forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            "logistic": LogisticRegression(max_iter=1000, random_state=42),
        }
        self.weights = {"xgboost": 0.45, "random_forest": 0.35, "logistic": 0.20}
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray):
        for name, model in self.models.items():
            model.fit(X, y)
        self.is_trained = True

    def predict(self, features: np.ndarray) -> tuple[float, float]:
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call train() first.")
        X = features.reshape(1, -1)
        probabilities = {}
        for name, model in self.models.items():
            proba = model.predict_proba(X)[0]
            probabilities[name] = float(proba[1]) if len(proba) > 1 else float(proba[0])
        weighted_prob = sum(probabilities[name] * self.weights[name] for name in self.models)
        probs = list(probabilities.values())
        std = float(np.std(probs))
        confidence = max(0.0, 1.0 - (std * 4))
        return weighted_prob, confidence

    def save(self, path: str):
        joblib.dump({"models": self.models, "weights": self.weights}, path)

    def load(self, path: str):
        data = joblib.load(path)
        self.models = data["models"]
        self.weights = data["weights"]
        self.is_trained = True
