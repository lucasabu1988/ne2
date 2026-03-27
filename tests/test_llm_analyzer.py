from unittest.mock import MagicMock
from datetime import datetime, timezone
from data.models import MarketSnapshot
from prediction.llm_analyzer import LLMAnalyzer

class TestLLMAnalyzer:
    def _make_snapshot(self):
        return MarketSnapshot(
            market_id="0x123", question="Will Bitcoin reach $100k by June 2026?",
            category="crypto", polymarket_price=0.65, volume_24h=200000,
            news_score=0.7, news_count=8,
            latest_headlines=["Bitcoin surges past $95k", "Institutional buying increases"],
            sentiment_score=0.45, sentiment_velocity=0.1,
            economic_indicators={"fed_funds_rate": 4.25},
            timestamp=datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc),
        )

    def test_analyze_returns_probability_and_reasoning(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"probability": 0.75, "reasoning": "Strong bullish momentum with institutional backing."}')]
        mock_client.messages.create.return_value = mock_response
        analyzer = LLMAnalyzer(client=mock_client)
        prob, reasoning = analyzer.analyze(self._make_snapshot())
        assert 0.0 <= prob <= 1.0
        assert len(reasoning) > 0

    def test_analyze_handles_malformed_response(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="I think the probability is about 70%")]
        mock_client.messages.create.return_value = mock_response
        analyzer = LLMAnalyzer(client=mock_client)
        prob, reasoning = analyzer.analyze(self._make_snapshot())
        assert 0.0 <= prob <= 1.0

    def test_analyze_handles_api_error(self):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API rate limit")
        analyzer = LLMAnalyzer(client=mock_client)
        prob, reasoning = analyzer.analyze(self._make_snapshot())
        assert prob == 0.5
        assert "error" in reasoning.lower() or "fail" in reasoning.lower()
