import logging
from data.models import MarketSnapshot, PredictionResult

logger = logging.getLogger(__name__)

class PredictionEngine:
    def __init__(self, feature_engineer, ml_ensemble, llm_analyzer, combiner, db,
                 ml_confidence_threshold: float = 0.10):
        self.feature_engineer = feature_engineer
        self.ml_ensemble = ml_ensemble
        self.llm_analyzer = llm_analyzer
        self.combiner = combiner
        self.db = db
        self.ml_confidence_threshold = ml_confidence_threshold

    def predict(self, snapshot: MarketSnapshot) -> PredictionResult:
        history = self.db.get_price_history(snapshot.market_id, limit=50)
        features = self.feature_engineer.extract(snapshot, history)
        ml_prob, ml_conf = self.ml_ensemble.predict(features)
        logger.info(f"[{snapshot.market_id}] ML: prob={ml_prob:.3f}, conf={ml_conf:.3f}")
        if ml_conf >= self.ml_confidence_threshold:
            llm_prob, llm_reasoning = self.llm_analyzer.analyze(snapshot)
            logger.info(f"[{snapshot.market_id}] LLM: prob={llm_prob:.3f}")
        else:
            llm_prob = 0.5
            llm_reasoning = "Skipped - low ML confidence"
            logger.info(f"[{snapshot.market_id}] LLM skipped (ML conf={ml_conf:.3f})")
        result = self.combiner.combine(
            market_id=snapshot.market_id, polymarket_price=snapshot.polymarket_price,
            ml_probability=ml_prob, ml_confidence=ml_conf,
            llm_probability=llm_prob, llm_reasoning=llm_reasoning,
        )
        self.db.save_prediction(result)
        return result

    def predict_batch(self, snapshots: list[MarketSnapshot]) -> list[PredictionResult]:
        results = []
        for snap in snapshots:
            try:
                result = self.predict(snap)
                results.append(result)
            except Exception as e:
                logger.error(f"Prediction failed for {snap.market_id}: {e}")
        return results
