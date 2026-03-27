# data/models.py
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


@dataclass
class MarketSnapshot:
    market_id: str
    question: str
    category: str
    polymarket_price: float
    volume_24h: float
    order_book_depth: dict = field(default_factory=dict)
    news_score: float = 0.0
    news_count: int = 0
    latest_headlines: list[str] = field(default_factory=list)
    sentiment_score: float = 0.0
    sentiment_velocity: float = 0.0
    economic_indicators: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class TradeAction(Enum):
    BUY_YES = "buy_yes"
    BUY_NO = "buy_no"


class TradeStatus(Enum):
    PENDING = "pending"
    EXECUTED = "executed"
    REJECTED = "rejected"
    CLOSED = "closed"
    STOPPED = "stopped"


@dataclass
class PredictionResult:
    market_id: str
    ml_probability: float
    ml_confidence: float
    llm_probability: float
    llm_reasoning: str
    final_probability: float
    confidence: float
    mispricing: float
    polymarket_price: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def has_signal(self, min_mispricing: float, min_confidence: float) -> bool:
        return abs(self.mispricing) >= min_mispricing and self.confidence >= min_confidence


@dataclass
class TradeRecord:
    trade_id: str
    market_id: str
    action: TradeAction
    amount: float
    price: float
    confidence: float
    mispricing: float
    status: TradeStatus
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    close_price: float | None = None
    close_timestamp: datetime | None = None
    rejection_reason: str | None = None

    def unrealized_pnl(self, current_price: float) -> float:
        if self.action == TradeAction.BUY_YES:
            return self.amount * (current_price - self.price) / self.price
        else:
            return self.amount * (self.price - current_price) / (1 - self.price)
