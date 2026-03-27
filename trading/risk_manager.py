import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from data.models import PredictionResult

logger = logging.getLogger(__name__)

@dataclass
class RiskDecision:
    approved: bool
    position_size: float = 0.0
    rejection_reason: str | None = None

class RiskManager:
    def __init__(self, db, bankroll: float = 1000.0, max_trade_pct: float = 0.02,
                 max_daily_pct: float = 0.10, stop_loss_pct: float = 0.25,
                 min_confidence: float = 0.80, min_mispricing: float = 0.10,
                 max_open_positions: int = 5, cooldown_minutes: int = 60):
        self.db = db
        self.bankroll = bankroll
        self.max_trade_pct = max_trade_pct
        self.max_daily_pct = max_daily_pct
        self.stop_loss_pct = stop_loss_pct
        self.min_confidence = min_confidence
        self.min_mispricing = min_mispricing
        self.max_open_positions = max_open_positions
        self.cooldown_minutes = cooldown_minutes
        self.kill_switch = False
        self.last_loss_time: datetime | None = None

    def evaluate(self, prediction: PredictionResult) -> RiskDecision:
        if self.kill_switch:
            return RiskDecision(approved=False, rejection_reason="Kill switch is active")
        if prediction.confidence < self.min_confidence:
            return RiskDecision(approved=False, rejection_reason=f"Confidence {prediction.confidence:.2f} below minimum {self.min_confidence}")
        if abs(prediction.mispricing) < self.min_mispricing:
            return RiskDecision(approved=False, rejection_reason=f"Mispricing {abs(prediction.mispricing):.2f} below minimum {self.min_mispricing}")
        open_trades = self.db.get_open_trades()
        if len(open_trades) >= self.max_open_positions:
            return RiskDecision(approved=False, rejection_reason=f"Max open positions reached ({self.max_open_positions})")
        now = datetime.now(timezone.utc)
        daily_total = self.db.get_daily_trade_total(now)
        max_daily = self.bankroll * self.max_daily_pct
        position_size = self.bankroll * self.max_trade_pct
        if daily_total + position_size > max_daily:
            return RiskDecision(approved=False, rejection_reason=f"Daily limit would be exceeded (${daily_total:.0f} + ${position_size:.0f} > ${max_daily:.0f})")
        if self.last_loss_time:
            cooldown_end = self.last_loss_time + timedelta(minutes=self.cooldown_minutes)
            if now < cooldown_end:
                return RiskDecision(approved=False, rejection_reason=f"Post-loss cooldown active until {cooldown_end.isoformat()}")
        logger.info(f"Trade approved for {prediction.market_id}: ${position_size:.2f}")
        return RiskDecision(approved=True, position_size=position_size)

    def record_loss(self):
        self.last_loss_time = datetime.now(timezone.utc)

    def check_stop_loss(self, entry_price: float, current_price: float, action: str) -> bool:
        if action == "buy_yes":
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / (1 - entry_price)
        return pnl_pct <= -self.stop_loss_pct
