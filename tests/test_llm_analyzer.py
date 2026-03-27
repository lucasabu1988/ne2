from datetime import datetime, timezone
from data.models import MarketSnapshot
from prediction.llm_analyzer import LLMAnalyzer


class TestLLMAnalyzer:
    def _make_snapshot(self, headlines=None):
        return MarketSnapshot(
            market_id="0x123", question="Will Bitcoin reach $100k by June 2026?",
            category="crypto", polymarket_price=0.65, volume_24h=200000,
            news_score=0.7, news_count=8,
            latest_headlines=headlines or ["Bitcoin surges past $95k", "Institutional buying increases"],
            sentiment_score=0.45, sentiment_velocity=0.1,
            economic_indicators={"fed_funds_rate": 4.25},
            timestamp=datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc),
        )

    def test_analyze_returns_probability_and_reasoning(self):
        analyzer = LLMAnalyzer()
        prob, reasoning = analyzer.analyze(self._make_snapshot())
        assert 0.0 <= prob <= 1.0
        assert len(reasoning) > 0

    def test_fallback_when_no_headlines(self):
        analyzer = LLMAnalyzer()
        prob, reasoning = analyzer.analyze(self._make_snapshot(headlines=[]))
        assert 0.0 <= prob <= 1.0
        assert "no recent news" in reasoning.lower()

    def test_probability_in_valid_range(self):
        analyzer = LLMAnalyzer()
        snap = self._make_snapshot(headlines=[
            "Bitcoin crashes to $50k",
            "Major sell-off in crypto markets",
            "Investors flee digital assets",
        ])
        prob, reasoning = analyzer.analyze(snap)
        assert 0.01 <= prob <= 0.99
