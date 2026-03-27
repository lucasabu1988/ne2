from unittest.mock import MagicMock
from datetime import datetime, timezone
import numpy as np
from data.models import MarketSnapshot, PredictionResult
from prediction.engine import PredictionEngine

class TestPredictionEngine:
    def _make_snapshot(self):
        return MarketSnapshot(
            market_id="0x123", question="Will BTC hit 100k?", category="crypto",
            polymarket_price=0.65, volume_24h=200000,
            timestamp=datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc),
        )

    def test_predict_returns_result(self):
        mock_features = MagicMock()
        mock_features.extract.return_value = np.random.rand(14)
        mock_ml = MagicMock()
        mock_ml.predict.return_value = (0.75, 0.90)
        mock_llm = MagicMock()
        mock_llm.analyze.return_value = (0.80, "Bullish analysis")
        mock_combiner = MagicMock()
        mock_combiner.combine.return_value = PredictionResult(
            market_id="0x123", ml_probability=0.75, ml_confidence=0.90,
            llm_probability=0.80, llm_reasoning="Bullish",
            final_probability=0.77, confidence=0.86, mispricing=0.12,
            polymarket_price=0.65,
        )
        mock_db = MagicMock()
        mock_db.get_price_history.return_value = []
        engine = PredictionEngine(
            feature_engineer=mock_features, ml_ensemble=mock_ml,
            llm_analyzer=mock_llm, combiner=mock_combiner, db=mock_db,
        )
        result = engine.predict(self._make_snapshot())
        assert isinstance(result, PredictionResult)
        assert result.market_id == "0x123"
        mock_db.save_prediction.assert_called_once()

    def test_predict_skips_llm_if_ml_low_confidence(self):
        mock_features = MagicMock()
        mock_features.extract.return_value = np.random.rand(14)
        mock_ml = MagicMock()
        mock_ml.predict.return_value = (0.55, 0.30)
        mock_llm = MagicMock()
        mock_combiner = MagicMock()
        mock_combiner.combine.return_value = PredictionResult(
            market_id="0x123", ml_probability=0.55, ml_confidence=0.30,
            llm_probability=0.5, llm_reasoning="Skipped - low ML confidence",
            final_probability=0.53, confidence=0.15, mispricing=-0.12,
            polymarket_price=0.65,
        )
        mock_db = MagicMock()
        mock_db.get_price_history.return_value = []
        engine = PredictionEngine(
            feature_engineer=mock_features, ml_ensemble=mock_ml,
            llm_analyzer=mock_llm, combiner=mock_combiner, db=mock_db,
            ml_confidence_threshold=0.50,
        )
        result = engine.predict(self._make_snapshot())
        mock_llm.analyze.assert_not_called()
