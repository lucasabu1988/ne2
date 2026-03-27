from unittest.mock import MagicMock
from data.models import PredictionResult, TradeAction, TradeStatus
from trading.executor import TradeExecutor

class TestTradeExecutor:
    def _make_prediction(self, mispricing=0.15):
        return PredictionResult(
            market_id="0x123", ml_probability=0.80, ml_confidence=0.90,
            llm_probability=0.80, llm_reasoning="Strong",
            final_probability=0.80, confidence=0.86, mispricing=mispricing,
            polymarket_price=0.65,
        )

    def test_execute_buy_yes(self):
        mock_db = MagicMock()
        executor = TradeExecutor(db=mock_db, dry_run=True)
        trade = executor.execute(self._make_prediction(mispricing=0.15), position_size=20.0)
        assert trade.action == TradeAction.BUY_YES
        assert trade.amount == 20.0
        assert trade.status == TradeStatus.EXECUTED
        mock_db.save_trade.assert_called_once()

    def test_execute_buy_no(self):
        mock_db = MagicMock()
        executor = TradeExecutor(db=mock_db, dry_run=True)
        trade = executor.execute(self._make_prediction(mispricing=-0.15), position_size=20.0)
        assert trade.action == TradeAction.BUY_NO

    def test_dry_run_does_not_call_api(self):
        mock_db = MagicMock()
        executor = TradeExecutor(db=mock_db, dry_run=True)
        trade = executor.execute(self._make_prediction(), position_size=20.0)
        assert trade.status == TradeStatus.EXECUTED
