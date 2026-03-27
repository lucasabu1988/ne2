from unittest.mock import MagicMock
from scheduler.jobs import run_analysis_cycle

class TestScheduler:
    def test_run_cycle_calls_pipeline(self):
        mock_ingestion = MagicMock()
        mock_ingestion.run.return_value = [MagicMock()]
        mock_prediction = MagicMock()
        mock_prediction.predict_batch.return_value = [MagicMock(has_signal=MagicMock(return_value=True))]
        mock_trading = MagicMock()
        mock_db = MagicMock()
        run_analysis_cycle(
            ingestion=mock_ingestion, prediction_engine=mock_prediction,
            trading_engine=mock_trading, db=mock_db,
            min_mispricing=0.10, min_confidence=0.80,
        )
        mock_ingestion.run.assert_called_once()
        mock_prediction.predict_batch.assert_called_once()
        mock_trading.process_batch.assert_called_once()

    def test_run_cycle_filters_signals(self):
        mock_ingestion = MagicMock()
        mock_ingestion.run.return_value = [MagicMock()]
        pred_with_signal = MagicMock()
        pred_with_signal.has_signal.return_value = True
        pred_without_signal = MagicMock()
        pred_without_signal.has_signal.return_value = False
        mock_prediction = MagicMock()
        mock_prediction.predict_batch.return_value = [pred_with_signal, pred_without_signal]
        mock_trading = MagicMock()
        mock_db = MagicMock()
        run_analysis_cycle(
            ingestion=mock_ingestion, prediction_engine=mock_prediction,
            trading_engine=mock_trading, db=mock_db,
            min_mispricing=0.10, min_confidence=0.80,
        )
        args = mock_trading.process_batch.call_args[0][0]
        assert len(args) == 1
