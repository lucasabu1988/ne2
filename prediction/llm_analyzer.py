"""Market analyzer using local Hugging Face zero-shot classification model.
No API key needed — runs entirely on CPU."""

import logging
from data.models import MarketSnapshot

logger = logging.getLogger(__name__)

# Lazy-loaded model to avoid slow import at startup
_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        logger.info("Loading zero-shot classification model (first time only)...")
        from transformers import pipeline
        _classifier = pipeline(
            "zero-shot-classification",
            model="typeform/distilbert-base-uncased-mnli",
            device=-1,  # CPU
        )
        logger.info("Model loaded.")
    return _classifier


class LLMAnalyzer:
    def __init__(self, client=None, **kwargs):
        # client param kept for backwards compatibility but ignored
        pass

    def analyze(self, snapshot: MarketSnapshot) -> tuple[float, str]:
        try:
            return self._analyze_with_nli(snapshot)
        except Exception as e:
            logger.error(f"Analysis failed for {snapshot.market_id}: {e}")
            return 0.5, f"Analysis failed: {e}"

    def _analyze_with_nli(self, snap: MarketSnapshot) -> tuple[float, str]:
        classifier = _get_classifier()
        question = snap.question
        headlines = snap.latest_headlines[:5]

        if not headlines:
            return self._fallback_analysis(snap)

        # Classify each headline: does it support YES or NO for this market?
        yes_label = f"Yes, {question}"
        no_label = f"No, not {question}"
        labels = [yes_label, no_label, "unrelated"]

        yes_scores = []
        reasoning_parts = []

        for headline in headlines:
            result = classifier(headline, candidate_labels=labels)
            top_label = result["labels"][0]
            top_score = result["scores"][0]

            if top_label == yes_label:
                yes_scores.append(top_score)
                reasoning_parts.append(f"'{headline[:60]}' → supports YES ({top_score:.0%})")
            elif top_label == no_label:
                yes_scores.append(1.0 - top_score)
                reasoning_parts.append(f"'{headline[:60]}' → supports NO ({top_score:.0%})")
            else:
                yes_scores.append(0.5)
                reasoning_parts.append(f"'{headline[:60]}' → unrelated")

        # Aggregate headline signals
        if yes_scores:
            news_probability = sum(yes_scores) / len(yes_scores)
        else:
            news_probability = 0.5

        # Blend with market signals
        sentiment_adjustment = snap.sentiment_score * 0.1  # +-10% max
        news_weight = min(snap.news_score, 1.0) * 0.3  # news_score determines how much news matters

        # Final probability: market price as anchor, adjusted by news + sentiment
        anchor = snap.polymarket_price
        news_pull = (news_probability - anchor) * news_weight
        final_prob = anchor + news_pull + sentiment_adjustment
        final_prob = max(0.01, min(0.99, final_prob))

        reasoning = f"Analyzed {len(headlines)} headlines. " + " | ".join(reasoning_parts[:3])
        if snap.sentiment_score != 0:
            reasoning += f" | Sentiment: {snap.sentiment_score:+.2f}"

        return final_prob, reasoning

    def _fallback_analysis(self, snap: MarketSnapshot) -> tuple[float, str]:
        """When no news is available, use market signals only."""
        price = snap.polymarket_price
        sentiment_adj = snap.sentiment_score * 0.05
        prob = max(0.01, min(0.99, price + sentiment_adj))
        return prob, "No recent news found. Using market price + sentiment only."
