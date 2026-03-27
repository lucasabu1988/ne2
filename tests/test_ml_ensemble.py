import numpy as np
import pytest
from prediction.ml_ensemble import MLEnsemble

class TestMLEnsemble:
    def _make_training_data(self, n=200):
        np.random.seed(42)
        X = np.random.rand(n, 14)
        y = (X[:, 0] > 0.5).astype(float)
        return X, y

    def test_train_and_predict(self):
        X, y = self._make_training_data()
        ensemble = MLEnsemble()
        ensemble.train(X, y)
        features = np.random.rand(14)
        prob, confidence = ensemble.predict(features)
        assert 0.0 <= prob <= 1.0
        assert 0.0 <= confidence <= 1.0

    def test_predict_without_training_raises(self):
        ensemble = MLEnsemble()
        with pytest.raises(RuntimeError, match="not trained"):
            ensemble.predict(np.random.rand(14))

    def test_confidence_measures_agreement(self):
        X, y = self._make_training_data(500)
        ensemble = MLEnsemble()
        ensemble.train(X, y)
        clear_case = np.ones(14) * 0.9
        _, confidence_high = ensemble.predict(clear_case)
        ambiguous = np.ones(14) * 0.5
        _, confidence_mid = ensemble.predict(ambiguous)
        assert 0.0 <= confidence_high <= 1.0
        assert 0.0 <= confidence_mid <= 1.0

    def test_save_and_load(self, tmp_path):
        X, y = self._make_training_data()
        ensemble = MLEnsemble()
        ensemble.train(X, y)
        path = str(tmp_path / "model.joblib")
        ensemble.save(path)
        loaded = MLEnsemble()
        loaded.load(path)
        features = np.random.rand(14)
        prob1, _ = ensemble.predict(features)
        prob2, _ = loaded.predict(features)
        assert prob1 == pytest.approx(prob2, abs=1e-6)
