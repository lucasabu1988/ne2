from datetime import datetime, timezone
from data.models import PredictionResult


class SignalCombiner:
    def __init__(self, ml_weight: float = 0.6, llm_weight: float = 0.4):
        self.ml_weight = ml_weight
        self.llm_weight = llm_weight

    def combine(self, market_id: str, polymarket_price: float,
                ml_probability: float, ml_confidence: float,
                llm_probability: float, llm_reasoning: str) -> PredictionResult:

        llm_available = not (
            llm_probability == 0.5 and (
                "failed" in llm_reasoning.lower()
                or "skipped" in llm_reasoning.lower()
                or "error" in llm_reasoning.lower()
            )
        )

        if llm_available:
            # Both ML and LLM produced real predictions — weighted blend
            final_prob = (self.ml_weight * ml_probability) + (self.llm_weight * llm_probability)
            disagreement = abs(ml_probability - llm_probability)
            agreement_factor = 1.0 - disagreement
            confidence = ml_confidence * agreement_factor
        else:
            # LLM unavailable — use ML only at full weight
            final_prob = ml_probability
            confidence = ml_confidence

        mispricing = final_prob - polymarket_price

        return PredictionResult(
            market_id=market_id, ml_probability=ml_probability,
            ml_confidence=ml_confidence, llm_probability=llm_probability,
            llm_reasoning=llm_reasoning, final_probability=final_prob,
            confidence=confidence, mispricing=mispricing,
            polymarket_price=polymarket_price,
            timestamp=datetime.now(timezone.utc),
        )
