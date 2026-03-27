from unittest.mock import MagicMock
from datetime import datetime, timezone
from data.models import PredictionResult
from trading.risk_manager import RiskManager, RiskDecision

class TestRiskManager:
    def _make_prediction(self, confidence=0.85, mispricing=0.15):
        return PredictionResult(
            market_id="0x123", ml_probability=0.80, ml_confidence=0.90,
            llm_probability=0.80, llm_reasoning="Strong",
            final_probability=0.80, confidence=confidence, mispricing=mispricing,
            polymarket_price=0.65,
        )

    def test_approve_valid_trade(self):
        mock_db = MagicMock()
        mock_db.get_open_trades.return_value = []
        mock_db.get_daily_trade_total.return_value = 0.0
        rm = RiskManager(db=mock_db, bankroll=1000.0, max_trade_pct=0.02, max_daily_pct=0.10,
                         min_confidence=0.80, min_mispricing=0.10, max_open_positions=5, cooldown_minutes=60)
        decision = rm.evaluate(self._make_prediction())
        assert decision.approved is True
        assert decision.position_size == 20.0
        assert decision.rejection_reason is None

    def test_reject_low_confidence(self):
        mock_db = MagicMock()
        mock_db.get_open_trades.return_value = []
        mock_db.get_daily_trade_total.return_value = 0.0
        rm = RiskManager(db=mock_db, bankroll=1000.0, min_confidence=0.80)
        decision = rm.evaluate(self._make_prediction(confidence=0.60))
        assert decision.approved is False
        assert "confidence" in decision.rejection_reason.lower()

    def test_reject_low_mispricing(self):
        mock_db = MagicMock()
        mock_db.get_open_trades.return_value = []
        mock_db.get_daily_trade_total.return_value = 0.0
        rm = RiskManager(db=mock_db, bankroll=1000.0, min_mispricing=0.10)
        decision = rm.evaluate(self._make_prediction(mispricing=0.05))
        assert decision.approved is False
        assert "mispricing" in decision.rejection_reason.lower()

    def test_reject_max_positions(self):
        mock_db = MagicMock()
        mock_db.get_open_trades.return_value = [MagicMock()] * 5
        mock_db.get_daily_trade_total.return_value = 0.0
        rm = RiskManager(db=mock_db, bankroll=1000.0, max_open_positions=5)
        decision = rm.evaluate(self._make_prediction())
        assert decision.approved is False
        assert "position" in decision.rejection_reason.lower()

    def test_reject_daily_limit(self):
        mock_db = MagicMock()
        mock_db.get_open_trades.return_value = []
        mock_db.get_daily_trade_total.return_value = 95.0
        rm = RiskManager(db=mock_db, bankroll=1000.0, max_daily_pct=0.10)
        decision = rm.evaluate(self._make_prediction())
        assert decision.approved is False
        assert "daily" in decision.rejection_reason.lower()

    def test_kill_switch(self):
        mock_db = MagicMock()
        mock_db.get_open_trades.return_value = []
        mock_db.get_daily_trade_total.return_value = 0.0
        rm = RiskManager(db=mock_db, bankroll=1000.0)
        rm.kill_switch = True
        decision = rm.evaluate(self._make_prediction())
        assert decision.approved is False
        assert "kill" in decision.rejection_reason.lower()
