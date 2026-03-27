import numpy as np
from data.models import MarketSnapshot

class FeatureEngineer:
    FEATURE_NAMES = [
        "price", "volume_24h_log", "bid_ask_imbalance", "news_score",
        "news_count_log", "sentiment_score", "sentiment_velocity",
        "momentum_short", "momentum_medium", "momentum_long",
        "volatility", "volume_change", "fed_funds_rate", "btc_price_log",
    ]

    def feature_names(self) -> list[str]:
        return self.FEATURE_NAMES.copy()

    def extract(self, snapshot: MarketSnapshot, history: list[MarketSnapshot]) -> np.ndarray:
        price = snapshot.polymarket_price
        volume_log = np.log1p(snapshot.volume_24h)
        depth = snapshot.order_book_depth
        imbalance = depth.get("bid_ask_imbalance", 0.5) if depth else 0.5
        news_score = snapshot.news_score
        news_count_log = np.log1p(snapshot.news_count)
        sentiment = snapshot.sentiment_score
        sentiment_vel = snapshot.sentiment_velocity
        prices = [s.polymarket_price for s in history]
        momentum_short = self._momentum(price, prices, 3)
        momentum_medium = self._momentum(price, prices, 6)
        momentum_long = self._momentum(price, prices, 10)
        volatility = float(np.std(prices[-10:])) if len(prices) >= 2 else 0.0
        volumes = [s.volume_24h for s in history]
        volume_change = 0.0
        if volumes:
            prev_vol = volumes[0] if volumes else snapshot.volume_24h
            volume_change = (snapshot.volume_24h - prev_vol) / max(prev_vol, 1.0)
        eco = snapshot.economic_indicators
        fed_rate = eco.get("fed_funds_rate", 0.0)
        btc_price = 0.0
        crypto = eco.get("crypto", {})
        if isinstance(crypto, dict) and "bitcoin" in crypto:
            btc_price = crypto["bitcoin"].get("usd", 0.0)
        btc_log = np.log1p(btc_price)
        return np.array([
            price, volume_log, imbalance, news_score, news_count_log,
            sentiment, sentiment_vel, momentum_short, momentum_medium,
            momentum_long, volatility, volume_change, fed_rate, btc_log,
        ], dtype=np.float64)

    def _momentum(self, current: float, prices: list[float], lookback: int) -> float:
        if len(prices) < lookback:
            return 0.0
        past = prices[-lookback]
        if past == 0:
            return 0.0
        return (current - past) / past
