from unittest.mock import MagicMock
from data.models import PredictionResult, TradeRecord, TradeAction, TradeStatus
from trading.risk_manager import RiskDecision
from trading.engine import TradingEngine

class TestTradingEngine:
    def _make_prediction(self, confidence=0.85, mispricing=0.15):
        return PredictionResult(
            market_id="0x123", ml_probability=0.80, ml_confidence=0.90,
            llm_probability=0.80, llm_reasoning="Strong",
            final_probability=0.80, confidence=confidence, mispricing=mispricing,
            polymarket_price=0.65,
        )

    def test_process_approved_signal(self):
        mock_risk = MagicMock()
        mock_risk.evaluate.return_value = RiskDecision(approved=True, position_size=20.0)
        mock_executor = MagicMock()
        mock_executor.execute.return_value = TradeRecord(
            trade_id="t1", market_id="0x123", action=TradeAction.BUY_YES,
            amount=20.0, price=0.65, confidence=0.85, mispricing=0.15,
            status=TradeStatus.EXECUTED,
        )
        mock_db = MagicMock()
        engine = TradingEngine(risk_manager=mock_risk, executor=mock_executor, db=mock_db)
        trade = engine.process_signal(self._make_prediction())
        assert trade is not None
        assert trade.status == TradeStatus.EXECUTED

    def test_process_rejected_signal(self):
        mock_risk = MagicMock()
        mock_risk.evaluate.return_value = RiskDecision(approved=False, rejection_reason="Low confidence")
        mock_executor = MagicMock()
        mock_db = MagicMock()
        engine = TradingEngine(risk_manager=mock_risk, executor=mock_executor, db=mock_db)
        trade = engine.process_signal(self._make_prediction(confidence=0.5))
        assert trade is None
        mock_executor.execute.assert_not_called()

    def test_process_batch(self):
        mock_risk = MagicMock()
        mock_risk.evaluate.side_effect = [
            RiskDecision(approved=True, position_size=20.0),
            RiskDecision(approved=False, rejection_reason="Low conf"),
        ]
        mock_executor = MagicMock()
        mock_executor.execute.return_value = TradeRecord(
            trade_id="t1", market_id="0x1", action=TradeAction.BUY_YES,
            amount=20.0, price=0.65, confidence=0.85, mispricing=0.15,
            status=TradeStatus.EXECUTED,
        )
        mock_db = MagicMock()
        engine = TradingEngine(risk_manager=mock_risk, executor=mock_executor, db=mock_db)
        preds = [self._make_prediction(), self._make_prediction(confidence=0.5)]
        trades = engine.process_batch(preds)
        assert len(trades) == 1
