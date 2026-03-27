from datetime import datetime, timezone
from data.models import MarketSnapshot, PredictionResult, TradeRecord, TradeAction, TradeStatus


class TestDatabase:
    def test_save_and_get_snapshot(self, db):
        snap = MarketSnapshot(
            market_id="0x123", question="Test market?", category="test",
            polymarket_price=0.65, volume_24h=100000.0, news_score=0.5,
            sentiment_score=0.3, timestamp=datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc),
        )
        db.save_snapshot(snap)
        result = db.get_latest_snapshot("0x123")
        assert result is not None
        assert result.market_id == "0x123"
        assert result.polymarket_price == 0.65

    def test_save_and_get_prediction(self, db):
        pred = PredictionResult(
            market_id="0x123", ml_probability=0.75, ml_confidence=0.85,
            llm_probability=0.80, llm_reasoning="Test reasoning",
            final_probability=0.77, confidence=0.83, mispricing=0.12,
            polymarket_price=0.65, timestamp=datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc),
        )
        db.save_prediction(pred)
        result = db.get_latest_prediction("0x123")
        assert result is not None
        assert result.final_probability == 0.77

    def test_save_and_get_trade(self, db):
        trade = TradeRecord(
            trade_id="t001", market_id="0x123", action=TradeAction.BUY_YES,
            amount=50.0, price=0.65, confidence=0.85, mispricing=0.12,
            status=TradeStatus.EXECUTED, timestamp=datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc),
        )
        db.save_trade(trade)
        result = db.get_open_trades()
        assert len(result) == 1
        assert result[0].trade_id == "t001"

    def test_get_price_history(self, db):
        for i in range(5):
            snap = MarketSnapshot(
                market_id="0x123", question="Test?", category="test",
                polymarket_price=0.60 + i * 0.02, volume_24h=100000.0,
                timestamp=datetime(2026, 3, 26, i, 0, tzinfo=timezone.utc),
            )
            db.save_snapshot(snap)
        history = db.get_price_history("0x123", limit=3)
        assert len(history) == 3

    def test_get_daily_trade_total(self, db):
        trade = TradeRecord(
            trade_id="t001", market_id="0x123", action=TradeAction.BUY_YES,
            amount=50.0, price=0.65, confidence=0.85, mispricing=0.12,
            status=TradeStatus.EXECUTED, timestamp=datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc),
        )
        db.save_trade(trade)
        total = db.get_daily_trade_total(datetime(2026, 3, 26, tzinfo=timezone.utc))
        assert total == 50.0

    def test_update_trade_status(self, db):
        trade = TradeRecord(
            trade_id="t001", market_id="0x123", action=TradeAction.BUY_YES,
            amount=50.0, price=0.65, confidence=0.85, mispricing=0.12,
            status=TradeStatus.EXECUTED, timestamp=datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc),
        )
        db.save_trade(trade)
        db.update_trade_status("t001", TradeStatus.CLOSED, close_price=0.80)
        trades = db.get_open_trades()
        assert len(trades) == 0
