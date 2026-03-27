"""ML training pipeline — fetches resolved Polymarket markets and trains the ensemble."""

import logging
import json
import numpy as np
import httpx
from datetime import datetime, timezone
from pathlib import Path

from prediction.ml_ensemble import MLEnsemble

logger = logging.getLogger(__name__)

GAMMA_URL = "https://gamma-api.polymarket.com"
MODEL_PATH = "models/ensemble.joblib"


def fetch_resolved_markets(limit_per_page: int = 100, max_markets: int = 3000) -> list[dict]:
    """Fetch resolved markets from Polymarket Gamma API."""
    client = httpx.Client(timeout=30.0)
    all_markets = []
    offset = 0

    while len(all_markets) < max_markets:
        try:
            response = client.get(
                f"{GAMMA_URL}/markets",
                params={
                    "closed": "true",
                    "limit": limit_per_page,
                    "offset": offset,
                    "order": "volume",
                    "ascending": "false",
                },
            )
            response.raise_for_status()
            batch = response.json()
            if not batch:
                break
            all_markets.extend(batch)
            offset += limit_per_page
            logger.info(f"Fetched {len(all_markets)} resolved markets...")
        except Exception as e:
            logger.error(f"Error fetching markets at offset {offset}: {e}")
            break

    client.close()
    logger.info(f"Total resolved markets fetched: {len(all_markets)}")
    return all_markets


def extract_label(market: dict) -> int | None:
    """Extract binary label: 1 if resolved YES, 0 if resolved NO."""
    prices_str = market.get("outcomePrices", "")
    if isinstance(prices_str, str):
        try:
            prices = json.loads(prices_str)
        except (json.JSONDecodeError, ValueError):
            return None
    elif isinstance(prices_str, list):
        prices = prices_str
    else:
        return None

    if len(prices) < 2:
        return None

    yes_price = float(prices[0])
    # Resolved markets have prices at 0 or 1
    if yes_price >= 0.99:
        return 1
    elif yes_price <= 0.01:
        return 0
    else:
        return None  # Not clearly resolved


def extract_features(market: dict) -> np.ndarray | None:
    """Extract feature vector from a resolved market snapshot."""
    try:
        volume = float(market.get("volumeNum", 0) or 0)
        if volume < 100:  # Skip markets with negligible volume
            return None

        last_price = float(market.get("lastTradePrice", 0) or 0)
        best_bid = float(market.get("bestBid", 0) or 0)
        best_ask = float(market.get("bestAsk", 0) or 0)
        spread = float(market.get("spread", 0) or 0)
        one_day_change = float(market.get("oneDayPriceChange", 0) or 0)

        # Volume metrics
        vol_1wk = float(market.get("volume1wk", 0) or 0)
        vol_1mo = float(market.get("volume1mo", 0) or 0)
        volume_log = np.log1p(volume)
        vol_1wk_ratio = vol_1wk / max(volume, 1)
        vol_1mo_ratio = vol_1mo / max(volume, 1)

        # Time features
        start_str = market.get("startDate") or market.get("startDateIso", "")
        end_str = market.get("endDate") or market.get("endDateIso", "")
        duration_hours = 0.0
        if start_str and end_str:
            try:
                start = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                end = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                duration_hours = max((end - start).total_seconds() / 3600, 0)
            except (ValueError, TypeError):
                pass
        duration_log = np.log1p(duration_hours)

        # Bid-ask imbalance
        bid_ask_sum = best_bid + best_ask
        imbalance = best_bid / bid_ask_sum if bid_ask_sum > 0 else 0.5

        # Price extremity — how far from 0.5 (uncertain) the price is
        price_extremity = abs(last_price - 0.5) * 2  # 0 = uncertain, 1 = certain

        # Feature vector (14 features to match our FeatureEngineer)
        features = np.array([
            last_price,           # price
            volume_log,           # volume_24h_log
            imbalance,            # bid_ask_imbalance
            0.0,                  # news_score (not available historically)
            0.0,                  # news_count_log
            0.0,                  # sentiment_score
            0.0,                  # sentiment_velocity
            one_day_change,       # momentum_short
            one_day_change * 0.5, # momentum_medium (approximation)
            one_day_change * 0.3, # momentum_long (approximation)
            spread,               # volatility proxy
            vol_1wk_ratio,        # volume_change proxy
            duration_log,         # fed_funds_rate slot → reused as duration
            price_extremity,      # btc_price_log slot → reused as price extremity
        ], dtype=np.float64)

        if np.any(np.isnan(features)):
            return None

        return features

    except (ValueError, TypeError) as e:
        return None


def build_training_data(markets: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Convert resolved markets into feature matrix X and label vector y."""
    X_list = []
    y_list = []

    for market in markets:
        label = extract_label(market)
        if label is None:
            continue

        features = extract_features(market)
        if features is None:
            continue

        X_list.append(features)
        y_list.append(label)

    X = np.array(X_list)
    y = np.array(y_list)
    logger.info(f"Training data: {len(X)} samples ({sum(y == 1)} YES, {sum(y == 0)} NO)")
    return X, y


def train_model(max_markets: int = 3000) -> MLEnsemble:
    """Full training pipeline: fetch data → extract features → train → save."""
    logger.info("=== Starting ML training pipeline ===")

    # Step 1: Fetch resolved markets
    markets = fetch_resolved_markets(max_markets=max_markets)

    # Step 2: Build training data
    X, y = build_training_data(markets)

    if len(X) < 50:
        raise ValueError(f"Not enough training data: {len(X)} samples (need at least 50)")

    logger.info(f"Training on {len(X)} samples with {X.shape[1]} features")

    # Step 3: Train ensemble
    ensemble = MLEnsemble()
    ensemble.train(X, y)

    # Step 4: Quick evaluation
    from sklearn.model_selection import cross_val_score
    for name, model in ensemble.models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        logger.info(f"  {name}: accuracy = {scores.mean():.3f} (+/- {scores.std():.3f})")

    # Step 5: Save model
    Path("models").mkdir(exist_ok=True)
    ensemble.save(MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")

    return ensemble


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    train_model()
