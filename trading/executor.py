import logging
import uuid
from datetime import datetime, timezone
from data.models import PredictionResult, TradeAction, TradeRecord, TradeStatus

logger = logging.getLogger(__name__)

class TradeExecutor:
    def __init__(self, db, api_client=None, dry_run: bool = True):
        self.db = db
        self.api_client = api_client
        self.dry_run = dry_run

    def execute(self, prediction: PredictionResult, position_size: float) -> TradeRecord:
        action = TradeAction.BUY_YES if prediction.mispricing > 0 else TradeAction.BUY_NO
        price = prediction.polymarket_price
        trade = TradeRecord(
            trade_id=str(uuid.uuid4())[:8], market_id=prediction.market_id,
            action=action, amount=position_size, price=price,
            confidence=prediction.confidence, mispricing=prediction.mispricing,
            status=TradeStatus.PENDING, timestamp=datetime.now(timezone.utc),
        )
        if self.dry_run:
            logger.info(f"[DRY RUN] Would {action.value} on {prediction.market_id} for ${position_size:.2f} at {price:.3f}")
            trade.status = TradeStatus.EXECUTED
        else:
            try:
                self._place_order(trade)
                trade.status = TradeStatus.EXECUTED
            except Exception as e:
                trade.status = TradeStatus.REJECTED
                trade.rejection_reason = str(e)
                logger.error(f"Trade failed for {prediction.market_id}: {e}")
        self.db.save_trade(trade)
        return trade

    def _place_order(self, trade: TradeRecord):
        if not self.api_client:
            raise RuntimeError("No API client configured for live trading")
        raise NotImplementedError("Live trading not yet implemented")
