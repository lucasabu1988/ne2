import numpy as np
from datetime import datetime, timezone
from data.models import MarketSnapshot
from prediction.features import FeatureEngineer

class TestFeatureEngineer:
    def _make_snapshot(self, price=0.65, volume=100000, sentiment=0.3, news=0.5):
        return MarketSnapshot(
            market_id="0x123", question="Test?", category="crypto",
            polymarket_price=price, volume_24h=volume,
            order_book_depth={"total_bid_size": 5000, "total_ask_size": 4000, "bid_ask_imbalance": 0.56},
            news_score=news, news_count=5, latest_headlines=["Headline 1"],
            sentiment_score=sentiment, sentiment_velocity=0.05,
            economic_indicators={"fed_funds_rate": 4.25, "crypto": {"bitcoin": {"usd": 95000}}},
            timestamp=datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc),
        )

    def test_extract_features_returns_array(self):
        snap = self._make_snapshot()
        history = [self._make_snapshot(price=0.60 + i * 0.01) for i in range(10)]
        engineer = FeatureEngineer()
        features = engineer.extract(snap, history)
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        assert not np.any(np.isnan(features))

    def test_feature_names_match_length(self):
        snap = self._make_snapshot()
        history = [self._make_snapshot(price=0.60 + i * 0.01) for i in range(10)]
        engineer = FeatureEngineer()
        features = engineer.extract(snap, history)
        names = engineer.feature_names()
        assert len(features) == len(names)

    def test_handles_empty_history(self):
        snap = self._make_snapshot()
        engineer = FeatureEngineer()
        features = engineer.extract(snap, [])
        assert isinstance(features, np.ndarray)
        assert not np.any(np.isnan(features))
