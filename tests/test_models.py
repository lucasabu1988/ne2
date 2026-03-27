# tests/test_models.py
from datetime import datetime, timezone

from data.models import MarketSnapshot, PredictionResult, TradeRecord, TradeAction, TradeStatus


class TestMarketSnapshot:
    def test_create_snapshot(self):
        snap = MarketSnapshot(
            market_id="0x123abc",
            question="Will Bitcoin reach $100k by June 2026?",
            category="crypto",
            polymarket_price=0.65,
            volume_24h=150000.0,
            order_book_depth={"bids": 50000, "asks": 45000},
            news_score=0.7,
            news_count=12,
            latest_headlines=["Bitcoin surges past $90k"],
            sentiment_score=0.45,
            sentiment_velocity=0.12,
            economic_indicators={"btc_price": 91000},
            timestamp=datetime(2026, 3, 26, tzinfo=timezone.utc),
        )
        assert snap.market_id == "0x123abc"
        assert snap.polymarket_price == 0.65
        assert snap.sentiment_score == 0.45

    def test_snapshot_defaults(self):
        snap = MarketSnapshot(
            market_id="0xabc",
            question="Test?",
            category="test",
            polymarket_price=0.5,
            volume_24h=1000.0,
        )
        assert snap.news_score == 0.0
        assert snap.news_count == 0
        assert snap.latest_headlines == []
        assert snap.sentiment_score == 0.0
        assert snap.sentiment_velocity == 0.0
        assert snap.economic_indicators == {}
        assert snap.order_book_depth == {}


class TestPredictionResult:
    def test_create_prediction(self):
        pred = PredictionResult(
            market_id="0x123abc",
            ml_probability=0.75,
            ml_confidence=0.85,
            llm_probability=0.80,
            llm_reasoning="Strong momentum and positive news cycle.",
            final_probability=0.77,
            confidence=0.83,
            mispricing=0.12,
            polymarket_price=0.65,
            timestamp=datetime(2026, 3, 26, tzinfo=timezone.utc),
        )
        assert pred.final_probability == 0.77
        assert pred.mispricing == 0.12

    def test_has_signal_true(self):
        pred = PredictionResult(
            market_id="0x1",
            ml_probability=0.8,
            ml_confidence=0.9,
            llm_probability=0.8,
            llm_reasoning="Strong",
            final_probability=0.8,
            confidence=0.85,
            mispricing=0.15,
            polymarket_price=0.65,
            timestamp=datetime(2026, 3, 26, tzinfo=timezone.utc),
        )
        assert pred.has_signal(min_mispricing=0.10, min_confidence=0.80) is True

    def test_has_signal_false_low_confidence(self):
        pred = PredictionResult(
            market_id="0x1",
            ml_probability=0.8,
            ml_confidence=0.5,
            llm_probability=0.8,
            llm_reasoning="Weak",
            final_probability=0.8,
            confidence=0.50,
            mispricing=0.15,
            polymarket_price=0.65,
            timestamp=datetime(2026, 3, 26, tzinfo=timezone.utc),
        )
        assert pred.has_signal(min_mispricing=0.10, min_confidence=0.80) is False


class TestTradeRecord:
    def test_create_trade(self):
        trade = TradeRecord(
            trade_id="t001",
            market_id="0x123abc",
            action=TradeAction.BUY_YES,
            amount=50.0,
            price=0.65,
            confidence=0.85,
            mispricing=0.12,
            status=TradeStatus.EXECUTED,
            timestamp=datetime(2026, 3, 26, tzinfo=timezone.utc),
        )
        assert trade.action == TradeAction.BUY_YES
        assert trade.status == TradeStatus.EXECUTED

    def test_pnl_calculation(self):
        trade = TradeRecord(
            trade_id="t001",
            market_id="0x123abc",
            action=TradeAction.BUY_YES,
            amount=100.0,
            price=0.65,
            confidence=0.85,
            mispricing=0.12,
            status=TradeStatus.EXECUTED,
            timestamp=datetime(2026, 3, 26, tzinfo=timezone.utc),
        )
        assert trade.unrealized_pnl(current_price=0.75) > 0
        assert trade.unrealized_pnl(current_price=0.50) < 0
