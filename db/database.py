import json
import sqlite3
from datetime import datetime, timezone

from data.models import (
    MarketSnapshot, PredictionResult, TradeAction, TradeRecord, TradeStatus,
)
from db.migrations import SCHEMA_SQL


class Database:
    def __init__(self, db_path: str = "ne2.db"):
        self.db_path = db_path
        self.conn: sqlite3.Connection | None = None

    def initialize(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def close(self):
        if self.conn:
            self.conn.close()

    def save_snapshot(self, snap: MarketSnapshot):
        self.conn.execute(
            """INSERT INTO snapshots
            (market_id, question, category, polymarket_price, volume_24h,
             order_book_depth, news_score, news_count, latest_headlines,
             sentiment_score, sentiment_velocity, economic_indicators, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (snap.market_id, snap.question, snap.category,
             snap.polymarket_price, snap.volume_24h,
             json.dumps(snap.order_book_depth), snap.news_score, snap.news_count,
             json.dumps(snap.latest_headlines), snap.sentiment_score,
             snap.sentiment_velocity, json.dumps(snap.economic_indicators),
             snap.timestamp.isoformat()),
        )
        self.conn.commit()

    def get_latest_snapshot(self, market_id: str) -> MarketSnapshot | None:
        row = self.conn.execute(
            "SELECT * FROM snapshots WHERE market_id = ? ORDER BY timestamp DESC LIMIT 1",
            (market_id,),
        ).fetchone()
        if not row:
            return None
        return self._row_to_snapshot(row)

    def get_price_history(self, market_id: str, limit: int = 100) -> list[MarketSnapshot]:
        rows = self.conn.execute(
            "SELECT * FROM snapshots WHERE market_id = ? ORDER BY timestamp DESC LIMIT ?",
            (market_id, limit),
        ).fetchall()
        return [self._row_to_snapshot(r) for r in rows]

    def save_prediction(self, pred: PredictionResult):
        self.conn.execute(
            """INSERT INTO predictions
            (market_id, ml_probability, ml_confidence, llm_probability, llm_reasoning,
             final_probability, confidence, mispricing, polymarket_price, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (pred.market_id, pred.ml_probability, pred.ml_confidence,
             pred.llm_probability, pred.llm_reasoning, pred.final_probability,
             pred.confidence, pred.mispricing, pred.polymarket_price,
             pred.timestamp.isoformat()),
        )
        self.conn.commit()

    def get_latest_prediction(self, market_id: str) -> PredictionResult | None:
        row = self.conn.execute(
            "SELECT * FROM predictions WHERE market_id = ? ORDER BY timestamp DESC LIMIT 1",
            (market_id,),
        ).fetchone()
        if not row:
            return None
        return PredictionResult(
            market_id=row["market_id"], ml_probability=row["ml_probability"],
            ml_confidence=row["ml_confidence"], llm_probability=row["llm_probability"],
            llm_reasoning=row["llm_reasoning"], final_probability=row["final_probability"],
            confidence=row["confidence"], mispricing=row["mispricing"],
            polymarket_price=row["polymarket_price"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
        )

    def save_trade(self, trade: TradeRecord):
        self.conn.execute(
            """INSERT INTO trades
            (trade_id, market_id, action, amount, price, confidence, mispricing,
             status, timestamp, close_price, close_timestamp, rejection_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (trade.trade_id, trade.market_id, trade.action.value,
             trade.amount, trade.price, trade.confidence, trade.mispricing,
             trade.status.value, trade.timestamp.isoformat(),
             trade.close_price,
             trade.close_timestamp.isoformat() if trade.close_timestamp else None,
             trade.rejection_reason),
        )
        self.conn.commit()

    def get_open_trades(self) -> list[TradeRecord]:
        rows = self.conn.execute(
            "SELECT * FROM trades WHERE status = ?",
            (TradeStatus.EXECUTED.value,),
        ).fetchall()
        return [self._row_to_trade(r) for r in rows]

    def get_all_trades(self, limit: int = 100) -> list[TradeRecord]:
        rows = self.conn.execute(
            "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_trade(r) for r in rows]

    def get_daily_trade_total(self, date: datetime) -> float:
        day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = date.replace(hour=23, minute=59, second=59, microsecond=999999)
        row = self.conn.execute(
            "SELECT COALESCE(SUM(amount), 0) as total FROM trades WHERE status = ? AND timestamp BETWEEN ? AND ?",
            (TradeStatus.EXECUTED.value, day_start.isoformat(), day_end.isoformat()),
        ).fetchone()
        return row["total"]

    def update_trade_status(self, trade_id: str, status: TradeStatus, close_price: float | None = None):
        close_ts = datetime.now(timezone.utc).isoformat() if close_price else None
        self.conn.execute(
            "UPDATE trades SET status = ?, close_price = ?, close_timestamp = ? WHERE trade_id = ?",
            (status.value, close_price, close_ts, trade_id),
        )
        self.conn.commit()

    def save_log(self, level: str, module: str, message: str):
        self.conn.execute(
            "INSERT INTO system_logs (level, module, message, timestamp) VALUES (?, ?, ?, ?)",
            (level, module, message, datetime.now(timezone.utc).isoformat()),
        )
        self.conn.commit()

    def get_recent_logs(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM system_logs ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def _row_to_snapshot(self, row) -> MarketSnapshot:
        return MarketSnapshot(
            market_id=row["market_id"], question=row["question"],
            category=row["category"], polymarket_price=row["polymarket_price"],
            volume_24h=row["volume_24h"],
            order_book_depth=json.loads(row["order_book_depth"]),
            news_score=row["news_score"], news_count=row["news_count"],
            latest_headlines=json.loads(row["latest_headlines"]),
            sentiment_score=row["sentiment_score"],
            sentiment_velocity=row["sentiment_velocity"],
            economic_indicators=json.loads(row["economic_indicators"]),
            timestamp=datetime.fromisoformat(row["timestamp"]),
        )

    def _row_to_trade(self, row) -> TradeRecord:
        return TradeRecord(
            trade_id=row["trade_id"], market_id=row["market_id"],
            action=TradeAction(row["action"]), amount=row["amount"],
            price=row["price"], confidence=row["confidence"],
            mispricing=row["mispricing"], status=TradeStatus(row["status"]),
            timestamp=datetime.fromisoformat(row["timestamp"]),
            close_price=row["close_price"],
            close_timestamp=datetime.fromisoformat(row["close_timestamp"]) if row["close_timestamp"] else None,
            rejection_reason=row["rejection_reason"],
        )
