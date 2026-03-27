import pytest
from datetime import datetime, timezone
from data.models import PredictionResult
from prediction.combiner import SignalCombiner

class TestSignalCombiner:
    def test_combine_agreeing_signals(self):
        combiner = SignalCombiner(ml_weight=0.6, llm_weight=0.4)
        result = combiner.combine(
            market_id="0x123", polymarket_price=0.50,
            ml_probability=0.75, ml_confidence=0.90,
            llm_probability=0.80, llm_reasoning="Strong bullish signal",
        )
        assert isinstance(result, PredictionResult)
        assert result.final_probability == pytest.approx(0.77, abs=0.01)
        assert result.mispricing == pytest.approx(0.27, abs=0.01)
        assert result.confidence > 0.8

    def test_combine_disagreeing_signals(self):
        combiner = SignalCombiner(ml_weight=0.6, llm_weight=0.4)
        result = combiner.combine(
            market_id="0x123", polymarket_price=0.50,
            ml_probability=0.80, ml_confidence=0.90,
            llm_probability=0.30, llm_reasoning="Bearish outlook",
        )
        assert result.confidence < 0.8

    def test_combine_no_mispricing(self):
        combiner = SignalCombiner()
        result = combiner.combine(
            market_id="0x123", polymarket_price=0.70,
            ml_probability=0.70, ml_confidence=0.85,
            llm_probability=0.70, llm_reasoning="Fair price",
        )
        assert abs(result.mispricing) < 0.01

    def test_has_signal_filters_correctly(self):
        combiner = SignalCombiner()
        strong = combiner.combine("0x1", 0.50, 0.75, 0.90, 0.80, "Strong")
        weak = combiner.combine("0x2", 0.50, 0.52, 0.60, 0.51, "Weak")
        assert strong.has_signal(min_mispricing=0.10, min_confidence=0.80) is True
        assert weak.has_signal(min_mispricing=0.10, min_confidence=0.80) is False
