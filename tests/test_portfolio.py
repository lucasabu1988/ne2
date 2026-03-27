from unittest.mock import MagicMock
from datetime import datetime, timezone
from data.models import TradeRecord, TradeAction, TradeStatus
from trading.portfolio import Portfolio

def _make_trade(trade_id, amount, price, action=TradeAction.BUY_YES, status=TradeStatus.EXECUTED):
    return TradeRecord(
        trade_id=trade_id, market_id="0x123", action=action,
        amount=amount, price=price, confidence=0.85, mispricing=0.12,
        status=status, timestamp=datetime(2026, 3, 26, tzinfo=timezone.utc),
    )

class TestPortfolio:
    def test_get_open_positions(self):
        mock_db = MagicMock()
        mock_db.get_open_trades.return_value = [_make_trade("t1", 50, 0.65), _make_trade("t2", 30, 0.70)]
        portfolio = Portfolio(db=mock_db, bankroll=1000.0)
        positions = portfolio.get_open_positions()
        assert len(positions) == 2

    def test_total_invested(self):
        mock_db = MagicMock()
        mock_db.get_open_trades.return_value = [_make_trade("t1", 50, 0.65), _make_trade("t2", 30, 0.70)]
        portfolio = Portfolio(db=mock_db, bankroll=1000.0)
        assert portfolio.total_invested() == 80.0

    def test_metrics_with_closed_trades(self):
        mock_db = MagicMock()
        closed1 = _make_trade("t1", 50, 0.65, status=TradeStatus.CLOSED)
        closed1.close_price = 0.80
        closed2 = _make_trade("t2", 50, 0.60, status=TradeStatus.CLOSED)
        closed2.close_price = 0.40
        mock_db.get_all_trades.return_value = [closed1, closed2]
        mock_db.get_open_trades.return_value = []
        portfolio = Portfolio(db=mock_db, bankroll=1000.0)
        metrics = portfolio.compute_metrics()
        assert "win_rate" in metrics
        assert "total_pnl" in metrics
        assert metrics["total_trades"] == 2
