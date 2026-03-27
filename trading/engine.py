import logging
from data.models import PredictionResult, TradeRecord, TradeStatus

logger = logging.getLogger(__name__)

class TradingEngine:
    def __init__(self, risk_manager, executor, db):
        self.risk_manager = risk_manager
        self.executor = executor
        self.db = db

    def process_signal(self, prediction: PredictionResult) -> TradeRecord | None:
        decision = self.risk_manager.evaluate(prediction)
        if not decision.approved:
            logger.info(f"Trade rejected for {prediction.market_id}: {decision.rejection_reason}")
            self.db.save_log("INFO", "trading", f"Rejected: {prediction.market_id} - {decision.rejection_reason}")
            return None
        trade = self.executor.execute(prediction, decision.position_size)
        if trade.status == TradeStatus.EXECUTED:
            self.db.save_log("INFO", "trading", f"Executed: {trade.action.value} {trade.market_id} ${trade.amount:.2f}")
        else:
            self.db.save_log("WARNING", "trading", f"Failed: {trade.market_id} - {trade.rejection_reason}")
        return trade

    def process_batch(self, predictions: list[PredictionResult]) -> list[TradeRecord]:
        trades = []
        for pred in predictions:
            trade = self.process_signal(pred)
            if trade:
                trades.append(trade)
        return trades
