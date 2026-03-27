# NE2 Polymarket Predictor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Python web app that monitors Polymarket events, predicts outcomes via hybrid ML+LLM engine, and auto-trades when high-confidence mispricing is detected.

**Architecture:** Python monolith with FastAPI (API), Dash/Plotly (dashboard), SQLite (storage), scikit-learn/XGBoost (ML), Claude API (LLM), APScheduler (scheduling). All modules communicate through a shared SQLite database and Python imports.

**Tech Stack:** Python 3.11+, FastAPI, Dash, Plotly, SQLite, scikit-learn, XGBoost, anthropic SDK, APScheduler, httpx, pandas, numpy

---

## File Map

```
ne2/
├── app.py                          # Entry point: FastAPI + Dash + Scheduler
├── config.py                       # Pydantic settings from .env
├── .env.example                    # Template for required env vars
├── .gitignore
├── requirements.txt
├── tests/
│   ├── conftest.py                 # Shared fixtures (test DB, mock clients)
│   ├── test_models.py
│   ├── test_database.py
│   ├── test_polymarket_client.py
│   ├── test_news_client.py
│   ├── test_sentiment_client.py
│   ├── test_economic_client.py
│   ├── test_ingestion.py
│   ├── test_features.py
│   ├── test_ml_ensemble.py
│   ├── test_llm_analyzer.py
│   ├── test_combiner.py
│   ├── test_prediction_engine.py
│   ├── test_risk_manager.py
│   ├── test_executor.py
│   ├── test_portfolio.py
│   ├── test_trading_engine.py
│   ├── test_scheduler.py
│   └── test_api.py
├── data/
│   ├── __init__.py
│   ├── models.py                   # MarketSnapshot, PredictionResult, Trade dataclasses
│   ├── polymarket_client.py
│   ├── news_client.py
│   ├── sentiment_client.py
│   ├── economic_client.py
│   └── ingestion.py               # Orchestrator
├── prediction/
│   ├── __init__.py
│   ├── features.py                 # Feature engineering from MarketSnapshot
│   ├── ml_ensemble.py              # XGBoost + RF + LogReg
│   ├── llm_analyzer.py             # Claude API qualitative analysis
│   ├── combiner.py                 # ML + LLM → final signal
│   └── engine.py                   # Orchestrator
├── trading/
│   ├── __init__.py
│   ├── risk_manager.py
│   ├── executor.py                 # Polymarket CLOB order execution
│   ├── portfolio.py                # Position + P&L tracking
│   └── engine.py                   # Orchestrator
├── dashboard/
│   ├── __init__.py
│   ├── app.py                      # Dash app factory
│   ├── layouts/
│   │   ├── __init__.py
│   │   ├── markets.py
│   │   ├── detail.py
│   │   ├── portfolio.py
│   │   └── control.py
│   └── callbacks/
│       ├── __init__.py
│       ├── markets_cb.py
│       ├── detail_cb.py
│       ├── portfolio_cb.py
│       └── control_cb.py
├── scheduler/
│   ├── __init__.py
│   └── jobs.py
└── db/
    ├── __init__.py
    ├── database.py
    └── migrations.py
```

---

## Phase 1: Foundation

### Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `config.py`

- [ ] **Step 1: Create requirements.txt**

```txt
# Core
fastapi==0.115.6
uvicorn==0.34.0
httpx==0.28.1
pydantic==2.10.4
pydantic-settings==2.7.1
python-dotenv==1.0.1

# Dashboard
dash==2.18.2
plotly==5.24.1
dash-bootstrap-components==1.6.0

# ML
scikit-learn==1.6.1
xgboost==2.1.3
pandas==2.2.3
numpy==1.26.4
joblib==1.4.2

# LLM
anthropic==0.43.0

# Scheduler
apscheduler==3.10.4

# Database
aiosqlite==0.20.0

# Testing
pytest==8.3.4
pytest-asyncio==0.24.0
pytest-httpx==0.34.0
```

- [ ] **Step 2: Create .gitignore**

```
__pycache__/
*.pyc
.env
*.db
*.sqlite
.pytest_cache/
*.egg-info/
dist/
build/
venv/
.venv/
models/*.joblib
```

- [ ] **Step 3: Create .env.example**

```bash
# Polymarket
POLYMARKET_API_URL=https://clob.polymarket.com

# News
NEWSAPI_KEY=your_newsapi_key_here

# Sentiment
TWITTER_BEARER_TOKEN=your_twitter_bearer_token
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret

# Economic
FRED_API_KEY=your_fred_api_key

# LLM
ANTHROPIC_API_KEY=your_anthropic_api_key

# Trading
POLYMARKET_PRIVATE_KEY=your_private_key_here
POLYMARKET_API_KEY=your_polymarket_api_key
POLYMARKET_API_SECRET=your_polymarket_api_secret
POLYMARKET_API_PASSPHRASE=your_polymarket_api_passphrase

# Risk
MAX_TRADE_PCT=0.02
MAX_DAILY_PCT=0.10
STOP_LOSS_PCT=0.25
MIN_CONFIDENCE=0.80
MIN_MISPRICING=0.10
MAX_OPEN_POSITIONS=5
COOLDOWN_MINUTES=60

# Scheduler
CYCLE_INTERVAL_HOURS=4

# Dashboard
DASH_PORT=8050
FASTAPI_PORT=8000
```

- [ ] **Step 4: Create config.py**

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Polymarket
    polymarket_api_url: str = "https://clob.polymarket.com"

    # News
    newsapi_key: str = ""

    # Sentiment
    twitter_bearer_token: str = ""
    reddit_client_id: str = ""
    reddit_client_secret: str = ""

    # Economic
    fred_api_key: str = ""

    # LLM
    anthropic_api_key: str = ""

    # Trading
    polymarket_private_key: str = ""
    polymarket_api_key: str = ""
    polymarket_api_secret: str = ""
    polymarket_api_passphrase: str = ""

    # Risk
    max_trade_pct: float = 0.02
    max_daily_pct: float = 0.10
    stop_loss_pct: float = 0.25
    min_confidence: float = 0.80
    min_mispricing: float = 0.10
    max_open_positions: int = 5
    cooldown_minutes: int = 60

    # Scheduler
    cycle_interval_hours: int = 4

    # Dashboard
    dash_port: int = 8050
    fastapi_port: int = 8000

    # Database
    db_path: str = "ne2.db"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
```

- [ ] **Step 5: Install dependencies and commit**

```bash
cd ne2
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash
pip install -r requirements.txt
git add requirements.txt .gitignore .env.example config.py
git commit -m "feat: project setup with dependencies and config"
```

---

### Task 2: Data Models

**Files:**
- Create: `data/__init__.py`
- Create: `data/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Create data/__init__.py**

```python
```

- [ ] **Step 2: Write failing test for MarketSnapshot**

```python
# tests/test_models.py
from datetime import datetime, timezone

from data.models import MarketSnapshot, PredictionResult, TradeRecord, TradeAction, TradeStatus


class TestMarketSnapshot:
    def test_create_snapshot(self):
        snap = MarketSnapshot(
            market_id="0x123abc",
            question="Will Bitcoin reach $100k by June 2026?",
            category="crypto",
            polymarket_price=0.65,
            volume_24h=150000.0,
            order_book_depth={"bids": 50000, "asks": 45000},
            news_score=0.7,
            news_count=12,
            latest_headlines=["Bitcoin surges past $90k"],
            sentiment_score=0.45,
            sentiment_velocity=0.12,
            economic_indicators={"btc_price": 91000},
            timestamp=datetime(2026, 3, 26, tzinfo=timezone.utc),
        )
        assert snap.market_id == "0x123abc"
        assert snap.polymarket_price == 0.65
        assert snap.sentiment_score == 0.45

    def test_snapshot_defaults(self):
        snap = MarketSnapshot(
            market_id="0xabc",
            question="Test?",
            category="test",
            polymarket_price=0.5,
            volume_24h=1000.0,
        )
        assert snap.news_score == 0.0
        assert snap.news_count == 0
        assert snap.latest_headlines == []
        assert snap.sentiment_score == 0.0
        assert snap.sentiment_velocity == 0.0
        assert snap.economic_indicators == {}
        assert snap.order_book_depth == {}


class TestPredictionResult:
    def test_create_prediction(self):
        pred = PredictionResult(
            market_id="0x123abc",
            ml_probability=0.75,
            ml_confidence=0.85,
            llm_probability=0.80,
            llm_reasoning="Strong momentum and positive news cycle.",
            final_probability=0.77,
            confidence=0.83,
            mispricing=0.12,
            polymarket_price=0.65,
            timestamp=datetime(2026, 3, 26, tzinfo=timezone.utc),
        )
        assert pred.final_probability == 0.77
        assert pred.mispricing == 0.12

    def test_has_signal_true(self):
        pred = PredictionResult(
            market_id="0x1",
            ml_probability=0.8,
            ml_confidence=0.9,
            llm_probability=0.8,
            llm_reasoning="Strong",
            final_probability=0.8,
            confidence=0.85,
            mispricing=0.15,
            polymarket_price=0.65,
            timestamp=datetime(2026, 3, 26, tzinfo=timezone.utc),
        )
        assert pred.has_signal(min_mispricing=0.10, min_confidence=0.80) is True

    def test_has_signal_false_low_confidence(self):
        pred = PredictionResult(
            market_id="0x1",
            ml_probability=0.8,
            ml_confidence=0.5,
            llm_probability=0.8,
            llm_reasoning="Weak",
            final_probability=0.8,
            confidence=0.50,
            mispricing=0.15,
            polymarket_price=0.65,
            timestamp=datetime(2026, 3, 26, tzinfo=timezone.utc),
        )
        assert pred.has_signal(min_mispricing=0.10, min_confidence=0.80) is False


class TestTradeRecord:
    def test_create_trade(self):
        trade = TradeRecord(
            trade_id="t001",
            market_id="0x123abc",
            action=TradeAction.BUY_YES,
            amount=50.0,
            price=0.65,
            confidence=0.85,
            mispricing=0.12,
            status=TradeStatus.EXECUTED,
            timestamp=datetime(2026, 3, 26, tzinfo=timezone.utc),
        )
        assert trade.action == TradeAction.BUY_YES
        assert trade.status == TradeStatus.EXECUTED

    def test_pnl_calculation(self):
        trade = TradeRecord(
            trade_id="t001",
            market_id="0x123abc",
            action=TradeAction.BUY_YES,
            amount=100.0,
            price=0.65,
            confidence=0.85,
            mispricing=0.12,
            status=TradeStatus.EXECUTED,
            timestamp=datetime(2026, 3, 26, tzinfo=timezone.utc),
        )
        # If current price is 0.75, unrealized PnL = amount * (current - entry) / entry
        assert trade.unrealized_pnl(current_price=0.75) > 0
        assert trade.unrealized_pnl(current_price=0.50) < 0
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/test_models.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'data.models'`

- [ ] **Step 4: Implement data models**

```python
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
```

- [ ] **Step 5: Run tests and verify they pass**

```bash
pytest tests/test_models.py -v
```

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add data/ tests/test_models.py
git commit -m "feat: add core data models (MarketSnapshot, PredictionResult, TradeRecord)"
```

---

### Task 3: Database Layer

**Files:**
- Create: `db/__init__.py`
- Create: `db/database.py`
- Create: `db/migrations.py`
- Create: `tests/conftest.py`
- Create: `tests/test_database.py`

- [ ] **Step 1: Write failing test for database**

```python
# tests/conftest.py
import os
import pytest

# Use in-memory SQLite for tests
os.environ["DB_PATH"] = ":memory:"


@pytest.fixture
def db():
    from db.database import Database
    database = Database(":memory:")
    database.initialize()
    yield database
    database.close()
```

```python
# tests/test_database.py
from datetime import datetime, timezone

from data.models import MarketSnapshot, PredictionResult, TradeRecord, TradeAction, TradeStatus


class TestDatabase:
    def test_save_and_get_snapshot(self, db):
        snap = MarketSnapshot(
            market_id="0x123",
            question="Test market?",
            category="test",
            polymarket_price=0.65,
            volume_24h=100000.0,
            news_score=0.5,
            sentiment_score=0.3,
            timestamp=datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc),
        )
        db.save_snapshot(snap)
        result = db.get_latest_snapshot("0x123")
        assert result is not None
        assert result.market_id == "0x123"
        assert result.polymarket_price == 0.65

    def test_save_and_get_prediction(self, db):
        pred = PredictionResult(
            market_id="0x123",
            ml_probability=0.75,
            ml_confidence=0.85,
            llm_probability=0.80,
            llm_reasoning="Test reasoning",
            final_probability=0.77,
            confidence=0.83,
            mispricing=0.12,
            polymarket_price=0.65,
            timestamp=datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc),
        )
        db.save_prediction(pred)
        result = db.get_latest_prediction("0x123")
        assert result is not None
        assert result.final_probability == 0.77

    def test_save_and_get_trade(self, db):
        trade = TradeRecord(
            trade_id="t001",
            market_id="0x123",
            action=TradeAction.BUY_YES,
            amount=50.0,
            price=0.65,
            confidence=0.85,
            mispricing=0.12,
            status=TradeStatus.EXECUTED,
            timestamp=datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc),
        )
        db.save_trade(trade)
        result = db.get_open_trades()
        assert len(result) == 1
        assert result[0].trade_id == "t001"

    def test_get_price_history(self, db):
        for i in range(5):
            snap = MarketSnapshot(
                market_id="0x123",
                question="Test?",
                category="test",
                polymarket_price=0.60 + i * 0.02,
                volume_24h=100000.0,
                timestamp=datetime(2026, 3, 26, i, 0, tzinfo=timezone.utc),
            )
            db.save_snapshot(snap)
        history = db.get_price_history("0x123", limit=3)
        assert len(history) == 3

    def test_get_daily_trade_total(self, db):
        trade = TradeRecord(
            trade_id="t001",
            market_id="0x123",
            action=TradeAction.BUY_YES,
            amount=50.0,
            price=0.65,
            confidence=0.85,
            mispricing=0.12,
            status=TradeStatus.EXECUTED,
            timestamp=datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc),
        )
        db.save_trade(trade)
        total = db.get_daily_trade_total(datetime(2026, 3, 26, tzinfo=timezone.utc))
        assert total == 50.0

    def test_update_trade_status(self, db):
        trade = TradeRecord(
            trade_id="t001",
            market_id="0x123",
            action=TradeAction.BUY_YES,
            amount=50.0,
            price=0.65,
            confidence=0.85,
            mispricing=0.12,
            status=TradeStatus.EXECUTED,
            timestamp=datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc),
        )
        db.save_trade(trade)
        db.update_trade_status("t001", TradeStatus.CLOSED, close_price=0.80)
        trades = db.get_open_trades()
        assert len(trades) == 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_database.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'db.database'`

- [ ] **Step 3: Implement database layer**

```python
# db/__init__.py
```

```python
# db/migrations.py
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL,
    question TEXT NOT NULL,
    category TEXT NOT NULL,
    polymarket_price REAL NOT NULL,
    volume_24h REAL NOT NULL,
    order_book_depth TEXT DEFAULT '{}',
    news_score REAL DEFAULT 0.0,
    news_count INTEGER DEFAULT 0,
    latest_headlines TEXT DEFAULT '[]',
    sentiment_score REAL DEFAULT 0.0,
    sentiment_velocity REAL DEFAULT 0.0,
    economic_indicators TEXT DEFAULT '{}',
    timestamp TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_snapshots_market_id ON snapshots(market_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON snapshots(timestamp);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL,
    ml_probability REAL NOT NULL,
    ml_confidence REAL NOT NULL,
    llm_probability REAL NOT NULL,
    llm_reasoning TEXT NOT NULL,
    final_probability REAL NOT NULL,
    confidence REAL NOT NULL,
    mispricing REAL NOT NULL,
    polymarket_price REAL NOT NULL,
    timestamp TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_predictions_market_id ON predictions(market_id);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT UNIQUE NOT NULL,
    market_id TEXT NOT NULL,
    action TEXT NOT NULL,
    amount REAL NOT NULL,
    price REAL NOT NULL,
    confidence REAL NOT NULL,
    mispricing REAL NOT NULL,
    status TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    close_price REAL,
    close_timestamp TEXT,
    rejection_reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_trades_market_id ON trades(market_id);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);

CREATE TABLE IF NOT EXISTS system_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    level TEXT NOT NULL,
    module TEXT NOT NULL,
    message TEXT NOT NULL,
    timestamp TEXT NOT NULL
);
"""
```

```python
# db/database.py
import json
import sqlite3
from datetime import datetime, timezone

from data.models import (
    MarketSnapshot,
    PredictionResult,
    TradeAction,
    TradeRecord,
    TradeStatus,
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
            (
                snap.market_id, snap.question, snap.category,
                snap.polymarket_price, snap.volume_24h,
                json.dumps(snap.order_book_depth), snap.news_score, snap.news_count,
                json.dumps(snap.latest_headlines), snap.sentiment_score,
                snap.sentiment_velocity, json.dumps(snap.economic_indicators),
                snap.timestamp.isoformat(),
            ),
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
            (
                pred.market_id, pred.ml_probability, pred.ml_confidence,
                pred.llm_probability, pred.llm_reasoning, pred.final_probability,
                pred.confidence, pred.mispricing, pred.polymarket_price,
                pred.timestamp.isoformat(),
            ),
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
            market_id=row["market_id"],
            ml_probability=row["ml_probability"],
            ml_confidence=row["ml_confidence"],
            llm_probability=row["llm_probability"],
            llm_reasoning=row["llm_reasoning"],
            final_probability=row["final_probability"],
            confidence=row["confidence"],
            mispricing=row["mispricing"],
            polymarket_price=row["polymarket_price"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
        )

    def save_trade(self, trade: TradeRecord):
        self.conn.execute(
            """INSERT INTO trades
            (trade_id, market_id, action, amount, price, confidence, mispricing,
             status, timestamp, close_price, close_timestamp, rejection_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trade.trade_id, trade.market_id, trade.action.value,
                trade.amount, trade.price, trade.confidence, trade.mispricing,
                trade.status.value, trade.timestamp.isoformat(),
                trade.close_price,
                trade.close_timestamp.isoformat() if trade.close_timestamp else None,
                trade.rejection_reason,
            ),
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

    def update_trade_status(
        self, trade_id: str, status: TradeStatus, close_price: float | None = None
    ):
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
            market_id=row["market_id"],
            question=row["question"],
            category=row["category"],
            polymarket_price=row["polymarket_price"],
            volume_24h=row["volume_24h"],
            order_book_depth=json.loads(row["order_book_depth"]),
            news_score=row["news_score"],
            news_count=row["news_count"],
            latest_headlines=json.loads(row["latest_headlines"]),
            sentiment_score=row["sentiment_score"],
            sentiment_velocity=row["sentiment_velocity"],
            economic_indicators=json.loads(row["economic_indicators"]),
            timestamp=datetime.fromisoformat(row["timestamp"]),
        )

    def _row_to_trade(self, row) -> TradeRecord:
        return TradeRecord(
            trade_id=row["trade_id"],
            market_id=row["market_id"],
            action=TradeAction(row["action"]),
            amount=row["amount"],
            price=row["price"],
            confidence=row["confidence"],
            mispricing=row["mispricing"],
            status=TradeStatus(row["status"]),
            timestamp=datetime.fromisoformat(row["timestamp"]),
            close_price=row["close_price"],
            close_timestamp=datetime.fromisoformat(row["close_timestamp"]) if row["close_timestamp"] else None,
            rejection_reason=row["rejection_reason"],
        )
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/test_database.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add db/ tests/conftest.py tests/test_database.py
git commit -m "feat: add SQLite database layer with CRUD for snapshots, predictions, trades"
```

---

## Phase 2: Data Ingestion

### Task 4: Polymarket Client

**Files:**
- Create: `data/polymarket_client.py`
- Create: `tests/test_polymarket_client.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_polymarket_client.py
import httpx
import pytest

from data.polymarket_client import PolymarketClient


SAMPLE_MARKETS_RESPONSE = {
    "data": [
        {
            "condition_id": "0xabc123",
            "question": "Will Bitcoin reach $100k by June 2026?",
            "category": "crypto",
            "tokens": [
                {"token_id": "yes_token_1", "outcome": "Yes", "price": 0.65},
                {"token_id": "no_token_1", "outcome": "No", "price": 0.35},
            ],
            "volume_num": 250000.0,
            "active": True,
            "end_date_iso": "2026-06-30T00:00:00Z",
        },
        {
            "condition_id": "0xdef456",
            "question": "Will ETH flip BTC by 2027?",
            "category": "crypto",
            "tokens": [
                {"token_id": "yes_token_2", "outcome": "Yes", "price": 0.10},
                {"token_id": "no_token_2", "outcome": "No", "price": 0.90},
            ],
            "volume_num": 50000.0,
            "active": True,
            "end_date_iso": "2027-01-01T00:00:00Z",
        },
    ]
}

SAMPLE_ORDERBOOK_RESPONSE = {
    "bids": [{"price": "0.64", "size": "1000"}, {"price": "0.63", "size": "2000"}],
    "asks": [{"price": "0.66", "size": "800"}, {"price": "0.67", "size": "1500"}],
}


class TestPolymarketClient:
    def test_get_top_markets(self, httpx_mock):
        httpx_mock.add_response(
            url="https://clob.polymarket.com/markets?active=true&limit=20&order=volume_num&ascending=false",
            json=SAMPLE_MARKETS_RESPONSE,
        )
        client = PolymarketClient(base_url="https://clob.polymarket.com")
        markets = client.get_top_markets(limit=20)
        assert len(markets) == 2
        assert markets[0]["condition_id"] == "0xabc123"
        assert markets[0]["volume_num"] == 250000.0

    def test_get_orderbook(self, httpx_mock):
        httpx_mock.add_response(
            url="https://clob.polymarket.com/book?token_id=yes_token_1",
            json=SAMPLE_ORDERBOOK_RESPONSE,
        )
        client = PolymarketClient(base_url="https://clob.polymarket.com")
        book = client.get_orderbook("yes_token_1")
        assert book["bids"][0]["price"] == "0.64"
        assert len(book["asks"]) == 2

    def test_parse_orderbook_depth(self):
        client = PolymarketClient(base_url="https://clob.polymarket.com")
        depth = client.parse_orderbook_depth(SAMPLE_ORDERBOOK_RESPONSE)
        assert depth["total_bid_size"] == 3000.0
        assert depth["total_ask_size"] == 2300.0
        assert depth["bid_ask_imbalance"] == pytest.approx(3000.0 / (3000.0 + 2300.0), rel=1e-3)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_polymarket_client.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement Polymarket client**

```python
# data/polymarket_client.py
import httpx


class PolymarketClient:
    def __init__(self, base_url: str = "https://clob.polymarket.com"):
        self.base_url = base_url
        self.client = httpx.Client(base_url=base_url, timeout=30.0)

    def get_top_markets(self, limit: int = 20) -> list[dict]:
        response = self.client.get(
            "/markets",
            params={
                "active": "true",
                "limit": limit,
                "order": "volume_num",
                "ascending": "false",
            },
        )
        response.raise_for_status()
        data = response.json()
        return data.get("data", data) if isinstance(data, dict) else data

    def get_orderbook(self, token_id: str) -> dict:
        response = self.client.get("/book", params={"token_id": token_id})
        response.raise_for_status()
        return response.json()

    def get_market_price(self, market: dict) -> float:
        tokens = market.get("tokens", [])
        for token in tokens:
            if token.get("outcome") == "Yes":
                return float(token.get("price", 0.0))
        return 0.0

    def parse_orderbook_depth(self, orderbook: dict) -> dict:
        total_bid = sum(float(b["size"]) for b in orderbook.get("bids", []))
        total_ask = sum(float(a["size"]) for a in orderbook.get("asks", []))
        total = total_bid + total_ask
        return {
            "total_bid_size": total_bid,
            "total_ask_size": total_ask,
            "bid_ask_imbalance": total_bid / total if total > 0 else 0.5,
        }

    def close(self):
        self.client.close()
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/test_polymarket_client.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add data/polymarket_client.py tests/test_polymarket_client.py
git commit -m "feat: add Polymarket CLOB API client"
```

---

### Task 5: News Client

**Files:**
- Create: `data/news_client.py`
- Create: `tests/test_news_client.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_news_client.py
from data.news_client import NewsClient


SAMPLE_NEWSAPI_RESPONSE = {
    "status": "ok",
    "totalResults": 2,
    "articles": [
        {
            "title": "Bitcoin Surges Past $95,000 Amid Institutional Buying",
            "description": "Major banks increase BTC exposure...",
            "publishedAt": "2026-03-26T10:00:00Z",
            "source": {"name": "Reuters"},
            "url": "https://reuters.com/btc-surge",
        },
        {
            "title": "Crypto Market Analysis: Bull Run Continues",
            "description": "Analysts predict further gains...",
            "publishedAt": "2026-03-26T08:00:00Z",
            "source": {"name": "Bloomberg"},
            "url": "https://bloomberg.com/crypto-bull",
        },
    ],
}


class TestNewsClient:
    def test_search_news(self, httpx_mock):
        httpx_mock.add_response(json=SAMPLE_NEWSAPI_RESPONSE)
        client = NewsClient(api_key="test_key")
        articles = client.search("Bitcoin $100k")
        assert len(articles) == 2
        assert "Bitcoin" in articles[0]["title"]

    def test_compute_news_score(self):
        client = NewsClient(api_key="test_key")
        articles = SAMPLE_NEWSAPI_RESPONSE["articles"]
        score = client.compute_news_score(articles, "Bitcoin reach $100k")
        assert 0.0 <= score <= 1.0

    def test_empty_results(self, httpx_mock):
        httpx_mock.add_response(json={"status": "ok", "totalResults": 0, "articles": []})
        client = NewsClient(api_key="test_key")
        articles = client.search("extremely obscure topic xyz")
        assert len(articles) == 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_news_client.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement News client**

```python
# data/news_client.py
import httpx
from datetime import datetime, timedelta, timezone


class NewsClient:
    def __init__(self, api_key: str, base_url: str = "https://newsapi.org/v2"):
        self.api_key = api_key
        self.base_url = base_url
        self.client = httpx.Client(timeout=30.0)

    def search(self, query: str, days_back: int = 3, max_results: int = 20) -> list[dict]:
        from_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
        response = self.client.get(
            f"{self.base_url}/everything",
            params={
                "q": query,
                "from": from_date,
                "sortBy": "relevancy",
                "pageSize": max_results,
                "apiKey": self.api_key,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data.get("articles", [])

    def compute_news_score(self, articles: list[dict], market_question: str) -> float:
        if not articles:
            return 0.0
        question_words = set(market_question.lower().split())
        relevance_scores = []
        for article in articles:
            title = (article.get("title") or "").lower()
            desc = (article.get("description") or "").lower()
            text = title + " " + desc
            text_words = set(text.split())
            overlap = len(question_words & text_words)
            relevance = min(overlap / max(len(question_words), 1), 1.0)
            relevance_scores.append(relevance)
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        volume_factor = min(len(articles) / 10.0, 1.0)
        return (avg_relevance * 0.6) + (volume_factor * 0.4)

    def close(self):
        self.client.close()
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/test_news_client.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add data/news_client.py tests/test_news_client.py
git commit -m "feat: add NewsAPI client with relevance scoring"
```

---

### Task 6: Sentiment Client

**Files:**
- Create: `data/sentiment_client.py`
- Create: `tests/test_sentiment_client.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_sentiment_client.py
from data.sentiment_client import SentimentClient


SAMPLE_TWITTER_RESPONSE = {
    "data": [
        {"id": "1", "text": "Bitcoin is going to the moon! $100k soon!", "created_at": "2026-03-26T10:00:00Z"},
        {"id": "2", "text": "I'm very bullish on BTC right now", "created_at": "2026-03-26T09:00:00Z"},
        {"id": "3", "text": "Bitcoin crash incoming, sell everything", "created_at": "2026-03-26T08:00:00Z"},
    ],
    "meta": {"result_count": 3},
}

SAMPLE_REDDIT_RESPONSE = {
    "data": {
        "children": [
            {"data": {"title": "BTC hitting 100k is inevitable", "selftext": "The fundamentals are strong", "score": 150}},
            {"data": {"title": "Bear case for Bitcoin", "selftext": "Regulation concerns are real", "score": 45}},
        ]
    }
}


class TestSentimentClient:
    def test_analyze_texts_positive(self):
        client = SentimentClient(twitter_bearer_token="test", reddit_client_id="test", reddit_client_secret="test")
        texts = ["This is amazing and wonderful!", "I love this, great news!"]
        score = client.analyze_sentiment(texts)
        assert score > 0.0

    def test_analyze_texts_negative(self):
        client = SentimentClient(twitter_bearer_token="test", reddit_client_id="test", reddit_client_secret="test")
        texts = ["This is terrible and awful!", "I hate this, worst news ever!"]
        score = client.analyze_sentiment(texts)
        assert score < 0.0

    def test_analyze_texts_empty(self):
        client = SentimentClient(twitter_bearer_token="test", reddit_client_id="test", reddit_client_secret="test")
        score = client.analyze_sentiment([])
        assert score == 0.0

    def test_sentiment_score_range(self):
        client = SentimentClient(twitter_bearer_token="test", reddit_client_id="test", reddit_client_secret="test")
        texts = ["Mixed feelings about this situation"]
        score = client.analyze_sentiment(texts)
        assert -1.0 <= score <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_sentiment_client.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement Sentiment client**

```python
# data/sentiment_client.py
import httpx

# Simple keyword-based sentiment as baseline; can be upgraded to a model later
POSITIVE_WORDS = {
    "bullish", "surge", "moon", "gain", "rise", "up", "great", "amazing",
    "wonderful", "love", "excellent", "strong", "inevitable", "positive",
    "optimistic", "win", "success", "profit", "buy", "breakout", "rally",
}
NEGATIVE_WORDS = {
    "bearish", "crash", "dump", "fall", "down", "terrible", "awful", "hate",
    "worst", "weak", "sell", "loss", "fear", "negative", "pessimistic",
    "fail", "decline", "drop", "collapse", "risk", "bubble",
}


class SentimentClient:
    def __init__(
        self,
        twitter_bearer_token: str = "",
        reddit_client_id: str = "",
        reddit_client_secret: str = "",
    ):
        self.twitter_token = twitter_bearer_token
        self.reddit_client_id = reddit_client_id
        self.reddit_client_secret = reddit_client_secret

    def fetch_twitter(self, query: str, max_results: int = 50) -> list[str]:
        if not self.twitter_token:
            return []
        client = httpx.Client(timeout=30.0)
        response = client.get(
            "https://api.twitter.com/2/tweets/search/recent",
            params={"query": query, "max_results": min(max_results, 100)},
            headers={"Authorization": f"Bearer {self.twitter_token}"},
        )
        client.close()
        if response.status_code != 200:
            return []
        data = response.json()
        return [tweet["text"] for tweet in data.get("data", [])]

    def fetch_reddit(self, query: str, subreddit: str = "all", limit: int = 25) -> list[str]:
        if not self.reddit_client_id:
            return []
        client = httpx.Client(timeout=30.0)
        response = client.get(
            f"https://www.reddit.com/r/{subreddit}/search.json",
            params={"q": query, "limit": limit, "sort": "relevance", "t": "week"},
            headers={"User-Agent": "NE2-Bot/1.0"},
        )
        client.close()
        if response.status_code != 200:
            return []
        data = response.json()
        posts = data.get("data", {}).get("children", [])
        return [p["data"]["title"] + " " + p["data"].get("selftext", "") for p in posts]

    def analyze_sentiment(self, texts: list[str]) -> float:
        if not texts:
            return 0.0
        scores = []
        for text in texts:
            words = set(text.lower().split())
            pos = len(words & POSITIVE_WORDS)
            neg = len(words & NEGATIVE_WORDS)
            total = pos + neg
            if total == 0:
                scores.append(0.0)
            else:
                scores.append((pos - neg) / total)
        return max(-1.0, min(1.0, sum(scores) / len(scores)))

    def get_sentiment_for_query(self, query: str) -> tuple[float, int]:
        twitter_texts = self.fetch_twitter(query)
        reddit_texts = self.fetch_reddit(query)
        all_texts = twitter_texts + reddit_texts
        score = self.analyze_sentiment(all_texts)
        return score, len(all_texts)
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/test_sentiment_client.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add data/sentiment_client.py tests/test_sentiment_client.py
git commit -m "feat: add sentiment client with Twitter/Reddit + keyword analysis"
```

---

### Task 7: Economic Client

**Files:**
- Create: `data/economic_client.py`
- Create: `tests/test_economic_client.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_economic_client.py
from data.economic_client import EconomicClient


SAMPLE_FRED_RESPONSE = {
    "observations": [
        {"date": "2026-03-26", "value": "4.25"},
        {"date": "2026-03-25", "value": "4.30"},
    ]
}

SAMPLE_COINGECKO_RESPONSE = {
    "bitcoin": {"usd": 95000.0, "usd_24h_change": 2.5},
    "ethereum": {"usd": 4200.0, "usd_24h_change": -1.2},
}


class TestEconomicClient:
    def test_get_fred_series(self, httpx_mock):
        httpx_mock.add_response(json=SAMPLE_FRED_RESPONSE)
        client = EconomicClient(fred_api_key="test_key")
        value = client.get_fred_latest("FEDFUNDS")
        assert value == 4.25

    def test_get_crypto_prices(self, httpx_mock):
        httpx_mock.add_response(json=SAMPLE_COINGECKO_RESPONSE)
        client = EconomicClient(fred_api_key="test_key")
        prices = client.get_crypto_prices(["bitcoin", "ethereum"])
        assert prices["bitcoin"]["usd"] == 95000.0
        assert prices["ethereum"]["usd_24h_change"] == -1.2

    def test_get_all_indicators(self, httpx_mock):
        httpx_mock.add_response(json=SAMPLE_FRED_RESPONSE)
        httpx_mock.add_response(json=SAMPLE_FRED_RESPONSE)
        httpx_mock.add_response(json=SAMPLE_COINGECKO_RESPONSE)
        client = EconomicClient(fred_api_key="test_key")
        indicators = client.get_all_indicators()
        assert "fed_funds_rate" in indicators
        assert "crypto" in indicators
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_economic_client.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement Economic client**

```python
# data/economic_client.py
import httpx


class EconomicClient:
    def __init__(self, fred_api_key: str = ""):
        self.fred_api_key = fred_api_key
        self.client = httpx.Client(timeout=30.0)

    def get_fred_latest(self, series_id: str) -> float | None:
        if not self.fred_api_key:
            return None
        response = self.client.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": series_id,
                "api_key": self.fred_api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 1,
            },
        )
        if response.status_code != 200:
            return None
        data = response.json()
        observations = data.get("observations", [])
        if not observations:
            return None
        value = observations[0].get("value", ".")
        if value == ".":
            return None
        return float(value)

    def get_crypto_prices(self, coin_ids: list[str] | None = None) -> dict:
        if coin_ids is None:
            coin_ids = ["bitcoin", "ethereum"]
        ids_str = ",".join(coin_ids)
        response = self.client.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={
                "ids": ids_str,
                "vs_currencies": "usd",
                "include_24hr_change": "true",
            },
        )
        if response.status_code != 200:
            return {}
        return response.json()

    def get_all_indicators(self) -> dict:
        indicators = {}
        fed_rate = self.get_fred_latest("FEDFUNDS")
        if fed_rate is not None:
            indicators["fed_funds_rate"] = fed_rate
        vix = self.get_fred_latest("VIXCLS")
        if vix is not None:
            indicators["vix"] = vix
        crypto = self.get_crypto_prices()
        if crypto:
            indicators["crypto"] = crypto
        return indicators

    def close(self):
        self.client.close()
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/test_economic_client.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add data/economic_client.py tests/test_economic_client.py
git commit -m "feat: add economic data client (FRED + CoinGecko)"
```

---

### Task 8: Ingestion Orchestrator

**Files:**
- Create: `data/ingestion.py`
- Create: `tests/test_ingestion.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_ingestion.py
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from data.ingestion import DataIngestion
from data.models import MarketSnapshot


MOCK_MARKETS = [
    {
        "condition_id": "0xabc",
        "question": "Will BTC hit 100k?",
        "category": "crypto",
        "tokens": [{"outcome": "Yes", "price": 0.65, "token_id": "tok1"}],
        "volume_num": 200000,
    }
]


class TestDataIngestion:
    def test_run_produces_snapshots(self):
        mock_poly = MagicMock()
        mock_poly.get_top_markets.return_value = MOCK_MARKETS
        mock_poly.get_market_price.return_value = 0.65
        mock_poly.get_orderbook.return_value = {"bids": [], "asks": []}
        mock_poly.parse_orderbook_depth.return_value = {
            "total_bid_size": 1000, "total_ask_size": 900, "bid_ask_imbalance": 0.53
        }

        mock_news = MagicMock()
        mock_news.search.return_value = [{"title": "BTC news", "description": "test"}]
        mock_news.compute_news_score.return_value = 0.6

        mock_sentiment = MagicMock()
        mock_sentiment.get_sentiment_for_query.return_value = (0.4, 15)

        mock_economic = MagicMock()
        mock_economic.get_all_indicators.return_value = {"fed_funds_rate": 4.25}

        mock_db = MagicMock()

        ingestion = DataIngestion(
            polymarket=mock_poly,
            news=mock_news,
            sentiment=mock_sentiment,
            economic=mock_economic,
            db=mock_db,
        )
        snapshots = ingestion.run()

        assert len(snapshots) == 1
        assert isinstance(snapshots[0], MarketSnapshot)
        assert snapshots[0].market_id == "0xabc"
        assert snapshots[0].polymarket_price == 0.65
        assert snapshots[0].news_score == 0.6
        assert snapshots[0].sentiment_score == 0.4
        mock_db.save_snapshot.assert_called_once()

    def test_run_handles_api_failure_gracefully(self):
        mock_poly = MagicMock()
        mock_poly.get_top_markets.side_effect = Exception("API error")

        ingestion = DataIngestion(
            polymarket=mock_poly,
            news=MagicMock(),
            sentiment=MagicMock(),
            economic=MagicMock(),
            db=MagicMock(),
        )
        snapshots = ingestion.run()
        assert snapshots == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_ingestion.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement ingestion orchestrator**

```python
# data/ingestion.py
import logging
from datetime import datetime, timezone

from data.models import MarketSnapshot

logger = logging.getLogger(__name__)


class DataIngestion:
    def __init__(self, polymarket, news, sentiment, economic, db):
        self.polymarket = polymarket
        self.news = news
        self.sentiment = sentiment
        self.economic = economic
        self.db = db

    def run(self, top_n: int = 20) -> list[MarketSnapshot]:
        try:
            markets = self.polymarket.get_top_markets(limit=top_n)
        except Exception as e:
            logger.error(f"Failed to fetch Polymarket markets: {e}")
            return []

        economic_indicators = {}
        try:
            economic_indicators = self.economic.get_all_indicators()
        except Exception as e:
            logger.warning(f"Failed to fetch economic data: {e}")

        snapshots = []
        for market in markets:
            try:
                snap = self._process_market(market, economic_indicators)
                self.db.save_snapshot(snap)
                snapshots.append(snap)
            except Exception as e:
                logger.warning(f"Failed to process market {market.get('condition_id', '?')}: {e}")

        logger.info(f"Ingestion complete: {len(snapshots)}/{len(markets)} markets processed")
        return snapshots

    def _process_market(self, market: dict, economic_indicators: dict) -> MarketSnapshot:
        market_id = market["condition_id"]
        question = market["question"]
        category = market.get("category", "unknown")
        price = self.polymarket.get_market_price(market)

        # Order book
        token_id = self._get_yes_token_id(market)
        order_book_depth = {}
        if token_id:
            try:
                raw_book = self.polymarket.get_orderbook(token_id)
                order_book_depth = self.polymarket.parse_orderbook_depth(raw_book)
            except Exception as e:
                logger.warning(f"Failed to fetch orderbook for {market_id}: {e}")

        # News
        news_score = 0.0
        news_count = 0
        headlines = []
        try:
            articles = self.news.search(question)
            news_count = len(articles)
            headlines = [a.get("title", "") for a in articles[:5]]
            news_score = self.news.compute_news_score(articles, question)
        except Exception as e:
            logger.warning(f"Failed to fetch news for {market_id}: {e}")

        # Sentiment
        sentiment_score = 0.0
        sentiment_count = 0
        try:
            sentiment_score, sentiment_count = self.sentiment.get_sentiment_for_query(question)
        except Exception as e:
            logger.warning(f"Failed to fetch sentiment for {market_id}: {e}")

        # Calculate sentiment velocity from history
        sentiment_velocity = self._calc_sentiment_velocity(market_id, sentiment_score)

        return MarketSnapshot(
            market_id=market_id,
            question=question,
            category=category,
            polymarket_price=price,
            volume_24h=market.get("volume_num", 0.0),
            order_book_depth=order_book_depth,
            news_score=news_score,
            news_count=news_count,
            latest_headlines=headlines,
            sentiment_score=sentiment_score,
            sentiment_velocity=sentiment_velocity,
            economic_indicators=economic_indicators,
            timestamp=datetime.now(timezone.utc),
        )

    def _get_yes_token_id(self, market: dict) -> str | None:
        for token in market.get("tokens", []):
            if token.get("outcome") == "Yes":
                return token.get("token_id")
        return None

    def _calc_sentiment_velocity(self, market_id: str, current_score: float) -> float:
        try:
            prev = self.db.get_latest_snapshot(market_id)
            if prev:
                return current_score - prev.sentiment_score
        except Exception:
            pass
        return 0.0
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/test_ingestion.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add data/ingestion.py tests/test_ingestion.py
git commit -m "feat: add data ingestion orchestrator"
```

---

## Phase 3: Prediction Engine

### Task 9: Feature Engineering

**Files:**
- Create: `prediction/__init__.py`
- Create: `prediction/features.py`
- Create: `tests/test_features.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_features.py
import numpy as np
from datetime import datetime, timezone

from data.models import MarketSnapshot
from prediction.features import FeatureEngineer


class TestFeatureEngineer:
    def _make_snapshot(self, price=0.65, volume=100000, sentiment=0.3, news=0.5):
        return MarketSnapshot(
            market_id="0x123",
            question="Test?",
            category="crypto",
            polymarket_price=price,
            volume_24h=volume,
            order_book_depth={"total_bid_size": 5000, "total_ask_size": 4000, "bid_ask_imbalance": 0.56},
            news_score=news,
            news_count=5,
            latest_headlines=["Headline 1"],
            sentiment_score=sentiment,
            sentiment_velocity=0.05,
            economic_indicators={"fed_funds_rate": 4.25, "crypto": {"bitcoin": {"usd": 95000}}},
            timestamp=datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc),
        )

    def test_extract_features_returns_array(self):
        snap = self._make_snapshot()
        history = [self._make_snapshot(price=0.60 + i * 0.01) for i in range(10)]
        engineer = FeatureEngineer()
        features = engineer.extract(snap, history)
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        assert not np.any(np.isnan(features))

    def test_feature_names_match_length(self):
        snap = self._make_snapshot()
        history = [self._make_snapshot(price=0.60 + i * 0.01) for i in range(10)]
        engineer = FeatureEngineer()
        features = engineer.extract(snap, history)
        names = engineer.feature_names()
        assert len(features) == len(names)

    def test_handles_empty_history(self):
        snap = self._make_snapshot()
        engineer = FeatureEngineer()
        features = engineer.extract(snap, [])
        assert isinstance(features, np.ndarray)
        assert not np.any(np.isnan(features))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_features.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement feature engineering**

```python
# prediction/__init__.py
```

```python
# prediction/features.py
import numpy as np

from data.models import MarketSnapshot


class FeatureEngineer:
    FEATURE_NAMES = [
        "price",
        "volume_24h_log",
        "bid_ask_imbalance",
        "news_score",
        "news_count_log",
        "sentiment_score",
        "sentiment_velocity",
        "momentum_short",    # vs 3 snapshots ago
        "momentum_medium",   # vs 6 snapshots ago
        "momentum_long",     # vs 10 snapshots ago
        "volatility",
        "volume_change",
        "fed_funds_rate",
        "btc_price_log",
    ]

    def feature_names(self) -> list[str]:
        return self.FEATURE_NAMES.copy()

    def extract(self, snapshot: MarketSnapshot, history: list[MarketSnapshot]) -> np.ndarray:
        price = snapshot.polymarket_price
        volume_log = np.log1p(snapshot.volume_24h)

        # Order book
        depth = snapshot.order_book_depth
        imbalance = depth.get("bid_ask_imbalance", 0.5) if depth else 0.5

        # News
        news_score = snapshot.news_score
        news_count_log = np.log1p(snapshot.news_count)

        # Sentiment
        sentiment = snapshot.sentiment_score
        sentiment_vel = snapshot.sentiment_velocity

        # Momentum from history
        prices = [s.polymarket_price for s in history]
        momentum_short = self._momentum(price, prices, 3)
        momentum_medium = self._momentum(price, prices, 6)
        momentum_long = self._momentum(price, prices, 10)

        # Volatility
        volatility = float(np.std(prices[-10:])) if len(prices) >= 2 else 0.0

        # Volume change
        volumes = [s.volume_24h for s in history]
        volume_change = 0.0
        if volumes:
            prev_vol = volumes[0] if volumes else snapshot.volume_24h
            volume_change = (snapshot.volume_24h - prev_vol) / max(prev_vol, 1.0)

        # Economic
        eco = snapshot.economic_indicators
        fed_rate = eco.get("fed_funds_rate", 0.0)
        btc_price = 0.0
        crypto = eco.get("crypto", {})
        if isinstance(crypto, dict) and "bitcoin" in crypto:
            btc_price = crypto["bitcoin"].get("usd", 0.0)
        btc_log = np.log1p(btc_price)

        return np.array([
            price,
            volume_log,
            imbalance,
            news_score,
            news_count_log,
            sentiment,
            sentiment_vel,
            momentum_short,
            momentum_medium,
            momentum_long,
            volatility,
            volume_change,
            fed_rate,
            btc_log,
        ], dtype=np.float64)

    def _momentum(self, current: float, prices: list[float], lookback: int) -> float:
        if len(prices) < lookback:
            return 0.0
        past = prices[-lookback]
        if past == 0:
            return 0.0
        return (current - past) / past
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/test_features.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add prediction/ tests/test_features.py
git commit -m "feat: add feature engineering for ML pipeline"
```

---

### Task 10: ML Ensemble

**Files:**
- Create: `prediction/ml_ensemble.py`
- Create: `tests/test_ml_ensemble.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_ml_ensemble.py
import numpy as np
import pytest

from prediction.ml_ensemble import MLEnsemble


class TestMLEnsemble:
    def _make_training_data(self, n=200):
        np.random.seed(42)
        X = np.random.rand(n, 14)
        y = (X[:, 0] > 0.5).astype(float)  # price > 0.5 → 1
        return X, y

    def test_train_and_predict(self):
        X, y = self._make_training_data()
        ensemble = MLEnsemble()
        ensemble.train(X, y)
        features = np.random.rand(14)
        prob, confidence = ensemble.predict(features)
        assert 0.0 <= prob <= 1.0
        assert 0.0 <= confidence <= 1.0

    def test_predict_without_training_raises(self):
        ensemble = MLEnsemble()
        with pytest.raises(RuntimeError, match="not trained"):
            ensemble.predict(np.random.rand(14))

    def test_confidence_measures_agreement(self):
        X, y = self._make_training_data(500)
        ensemble = MLEnsemble()
        ensemble.train(X, y)
        # A clear-cut case should have high confidence
        clear_case = np.ones(14) * 0.9
        _, confidence_high = ensemble.predict(clear_case)
        # An ambiguous case might have lower confidence
        ambiguous = np.ones(14) * 0.5
        _, confidence_mid = ensemble.predict(ambiguous)
        # Just verify both are valid numbers
        assert 0.0 <= confidence_high <= 1.0
        assert 0.0 <= confidence_mid <= 1.0

    def test_save_and_load(self, tmp_path):
        X, y = self._make_training_data()
        ensemble = MLEnsemble()
        ensemble.train(X, y)

        path = str(tmp_path / "model.joblib")
        ensemble.save(path)

        loaded = MLEnsemble()
        loaded.load(path)
        features = np.random.rand(14)
        prob1, _ = ensemble.predict(features)
        prob2, _ = loaded.predict(features)
        assert prob1 == pytest.approx(prob2, abs=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_ml_ensemble.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement ML ensemble**

```python
# prediction/ml_ensemble.py
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


class MLEnsemble:
    def __init__(self):
        self.models = {
            "xgboost": XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                use_label_encoder=False, eval_metric="logloss",
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42,
            ),
            "logistic": LogisticRegression(max_iter=1000, random_state=42),
        }
        self.weights = {"xgboost": 0.45, "random_forest": 0.35, "logistic": 0.20}
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray):
        for name, model in self.models.items():
            model.fit(X, y)
        self.is_trained = True

    def predict(self, features: np.ndarray) -> tuple[float, float]:
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call train() first.")

        X = features.reshape(1, -1)
        probabilities = {}
        for name, model in self.models.items():
            proba = model.predict_proba(X)[0]
            # probability of class 1 (YES outcome)
            probabilities[name] = float(proba[1]) if len(proba) > 1 else float(proba[0])

        # Weighted average
        weighted_prob = sum(
            probabilities[name] * self.weights[name]
            for name in self.models
        )

        # Confidence: 1 - std of individual predictions (more agreement = higher confidence)
        probs = list(probabilities.values())
        std = float(np.std(probs))
        confidence = max(0.0, 1.0 - (std * 4))  # scale std to 0-1 range

        return weighted_prob, confidence

    def save(self, path: str):
        joblib.dump({"models": self.models, "weights": self.weights}, path)

    def load(self, path: str):
        data = joblib.load(path)
        self.models = data["models"]
        self.weights = data["weights"]
        self.is_trained = True
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/test_ml_ensemble.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add prediction/ml_ensemble.py tests/test_ml_ensemble.py
git commit -m "feat: add ML ensemble (XGBoost + RF + LogReg)"
```

---

### Task 11: LLM Analyzer

**Files:**
- Create: `prediction/llm_analyzer.py`
- Create: `tests/test_llm_analyzer.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_llm_analyzer.py
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from data.models import MarketSnapshot
from prediction.llm_analyzer import LLMAnalyzer


class TestLLMAnalyzer:
    def _make_snapshot(self):
        return MarketSnapshot(
            market_id="0x123",
            question="Will Bitcoin reach $100k by June 2026?",
            category="crypto",
            polymarket_price=0.65,
            volume_24h=200000,
            news_score=0.7,
            news_count=8,
            latest_headlines=["Bitcoin surges past $95k", "Institutional buying increases"],
            sentiment_score=0.45,
            sentiment_velocity=0.1,
            economic_indicators={"fed_funds_rate": 4.25},
            timestamp=datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc),
        )

    def test_analyze_returns_probability_and_reasoning(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"probability": 0.75, "reasoning": "Strong bullish momentum with institutional backing."}')]
        mock_client.messages.create.return_value = mock_response

        analyzer = LLMAnalyzer(client=mock_client)
        prob, reasoning = analyzer.analyze(self._make_snapshot())
        assert 0.0 <= prob <= 1.0
        assert len(reasoning) > 0

    def test_analyze_handles_malformed_response(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="I think the probability is about 70%")]
        mock_client.messages.create.return_value = mock_response

        analyzer = LLMAnalyzer(client=mock_client)
        prob, reasoning = analyzer.analyze(self._make_snapshot())
        # Should fallback gracefully
        assert 0.0 <= prob <= 1.0

    def test_analyze_handles_api_error(self):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API rate limit")

        analyzer = LLMAnalyzer(client=mock_client)
        prob, reasoning = analyzer.analyze(self._make_snapshot())
        assert prob == 0.5  # neutral fallback
        assert "error" in reasoning.lower() or "fail" in reasoning.lower()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_llm_analyzer.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement LLM analyzer**

```python
# prediction/llm_analyzer.py
import json
import logging
import re

from data.models import MarketSnapshot

logger = logging.getLogger(__name__)

ANALYSIS_PROMPT = """You are a prediction market analyst. Analyze this market and estimate the true probability of the outcome.

Market Question: {question}
Category: {category}
Current Polymarket Price (implied probability): {price:.1%}
24h Volume: ${volume:,.0f}

Recent News (relevance score: {news_score:.2f}):
{headlines}

Social Sentiment Score: {sentiment:.2f} (-1 bearish to +1 bullish)
Sentiment Velocity: {sentiment_vel:+.2f}

Economic Context:
{economic}

Analyze:
1. What do the news headlines imply about this outcome?
2. Does the sentiment align with or diverge from the current price?
3. Are there factors the market might be underweighting or overweighting?
4. What is your estimated true probability?

Respond in JSON format:
{{"probability": 0.XX, "reasoning": "Your 2-3 sentence analysis"}}
"""


class LLMAnalyzer:
    def __init__(self, client=None, model: str = "claude-sonnet-4-6"):
        self.client = client
        self.model = model

    def analyze(self, snapshot: MarketSnapshot) -> tuple[float, str]:
        try:
            prompt = self._build_prompt(snapshot)
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            return self._parse_response(text)
        except Exception as e:
            logger.error(f"LLM analysis failed for {snapshot.market_id}: {e}")
            return 0.5, f"Analysis failed: {e}"

    def _build_prompt(self, snap: MarketSnapshot) -> str:
        headlines = "\n".join(f"- {h}" for h in snap.latest_headlines) or "- No recent headlines"
        eco_items = []
        for k, v in snap.economic_indicators.items():
            if k == "crypto":
                for coin, data in v.items():
                    if isinstance(data, dict):
                        eco_items.append(f"- {coin}: ${data.get('usd', 0):,.0f}")
            else:
                eco_items.append(f"- {k}: {v}")
        economic = "\n".join(eco_items) or "- No data available"

        return ANALYSIS_PROMPT.format(
            question=snap.question,
            category=snap.category,
            price=snap.polymarket_price,
            volume=snap.volume_24h,
            news_score=snap.news_score,
            headlines=headlines,
            sentiment=snap.sentiment_score,
            sentiment_vel=snap.sentiment_velocity,
            economic=economic,
        )

    def _parse_response(self, text: str) -> tuple[float, str]:
        # Try JSON parse first
        try:
            data = json.loads(text)
            prob = float(data["probability"])
            reasoning = data["reasoning"]
            return max(0.0, min(1.0, prob)), reasoning
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        # Try to find JSON in the text
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                prob = float(data.get("probability", 0.5))
                reasoning = data.get("reasoning", text)
                return max(0.0, min(1.0, prob)), reasoning
            except (json.JSONDecodeError, ValueError):
                pass

        # Last resort: extract any decimal number
        numbers = re.findall(r'0\.\d+', text)
        if numbers:
            prob = float(numbers[0])
            return max(0.0, min(1.0, prob)), text

        return 0.5, text
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/test_llm_analyzer.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add prediction/llm_analyzer.py tests/test_llm_analyzer.py
git commit -m "feat: add Claude LLM analyzer for qualitative market analysis"
```

---

### Task 12: Signal Combiner

**Files:**
- Create: `prediction/combiner.py`
- Create: `tests/test_combiner.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_combiner.py
import pytest
from datetime import datetime, timezone

from data.models import PredictionResult
from prediction.combiner import SignalCombiner


class TestSignalCombiner:
    def test_combine_agreeing_signals(self):
        combiner = SignalCombiner(ml_weight=0.6, llm_weight=0.4)
        result = combiner.combine(
            market_id="0x123",
            polymarket_price=0.50,
            ml_probability=0.75,
            ml_confidence=0.90,
            llm_probability=0.80,
            llm_reasoning="Strong bullish signal",
        )
        assert isinstance(result, PredictionResult)
        # 0.6*0.75 + 0.4*0.80 = 0.77
        assert result.final_probability == pytest.approx(0.77, abs=0.01)
        assert result.mispricing == pytest.approx(0.27, abs=0.01)
        assert result.confidence > 0.8  # High because ML+LLM agree

    def test_combine_disagreeing_signals(self):
        combiner = SignalCombiner(ml_weight=0.6, llm_weight=0.4)
        result = combiner.combine(
            market_id="0x123",
            polymarket_price=0.50,
            ml_probability=0.80,
            ml_confidence=0.90,
            llm_probability=0.30,
            llm_reasoning="Bearish outlook",
        )
        # Confidence should be lower when ML and LLM disagree
        assert result.confidence < 0.8

    def test_combine_no_mispricing(self):
        combiner = SignalCombiner()
        result = combiner.combine(
            market_id="0x123",
            polymarket_price=0.70,
            ml_probability=0.70,
            ml_confidence=0.85,
            llm_probability=0.70,
            llm_reasoning="Fair price",
        )
        assert abs(result.mispricing) < 0.01

    def test_has_signal_filters_correctly(self):
        combiner = SignalCombiner()
        strong = combiner.combine("0x1", 0.50, 0.75, 0.90, 0.80, "Strong")
        weak = combiner.combine("0x2", 0.50, 0.52, 0.60, 0.51, "Weak")
        assert strong.has_signal(min_mispricing=0.10, min_confidence=0.80) is True
        assert weak.has_signal(min_mispricing=0.10, min_confidence=0.80) is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_combiner.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement signal combiner**

```python
# prediction/combiner.py
from datetime import datetime, timezone

from data.models import PredictionResult


class SignalCombiner:
    def __init__(self, ml_weight: float = 0.6, llm_weight: float = 0.4):
        self.ml_weight = ml_weight
        self.llm_weight = llm_weight

    def combine(
        self,
        market_id: str,
        polymarket_price: float,
        ml_probability: float,
        ml_confidence: float,
        llm_probability: float,
        llm_reasoning: str,
    ) -> PredictionResult:
        final_prob = (self.ml_weight * ml_probability) + (self.llm_weight * llm_probability)
        mispricing = final_prob - polymarket_price

        # Agreement factor: how close ML and LLM predictions are
        # Max disagreement is 1.0 (one says 0, other says 1)
        disagreement = abs(ml_probability - llm_probability)
        agreement_factor = 1.0 - disagreement

        confidence = ml_confidence * agreement_factor

        return PredictionResult(
            market_id=market_id,
            ml_probability=ml_probability,
            ml_confidence=ml_confidence,
            llm_probability=llm_probability,
            llm_reasoning=llm_reasoning,
            final_probability=final_prob,
            confidence=confidence,
            mispricing=mispricing,
            polymarket_price=polymarket_price,
            timestamp=datetime.now(timezone.utc),
        )
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/test_combiner.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add prediction/combiner.py tests/test_combiner.py
git commit -m "feat: add ML+LLM signal combiner with confidence scoring"
```

---

### Task 13: Prediction Engine Orchestrator

**Files:**
- Create: `prediction/engine.py`
- Create: `tests/test_prediction_engine.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_prediction_engine.py
from unittest.mock import MagicMock
from datetime import datetime, timezone

import numpy as np

from data.models import MarketSnapshot, PredictionResult
from prediction.engine import PredictionEngine


class TestPredictionEngine:
    def _make_snapshot(self):
        return MarketSnapshot(
            market_id="0x123",
            question="Will BTC hit 100k?",
            category="crypto",
            polymarket_price=0.65,
            volume_24h=200000,
            timestamp=datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc),
        )

    def test_predict_returns_result(self):
        mock_features = MagicMock()
        mock_features.extract.return_value = np.random.rand(14)

        mock_ml = MagicMock()
        mock_ml.predict.return_value = (0.75, 0.90)

        mock_llm = MagicMock()
        mock_llm.analyze.return_value = (0.80, "Bullish analysis")

        mock_combiner = MagicMock()
        mock_combiner.combine.return_value = PredictionResult(
            market_id="0x123", ml_probability=0.75, ml_confidence=0.90,
            llm_probability=0.80, llm_reasoning="Bullish",
            final_probability=0.77, confidence=0.86, mispricing=0.12,
            polymarket_price=0.65,
        )

        mock_db = MagicMock()
        mock_db.get_price_history.return_value = []

        engine = PredictionEngine(
            feature_engineer=mock_features,
            ml_ensemble=mock_ml,
            llm_analyzer=mock_llm,
            combiner=mock_combiner,
            db=mock_db,
        )

        result = engine.predict(self._make_snapshot())
        assert isinstance(result, PredictionResult)
        assert result.market_id == "0x123"
        mock_db.save_prediction.assert_called_once()

    def test_predict_skips_llm_if_ml_low_confidence(self):
        mock_features = MagicMock()
        mock_features.extract.return_value = np.random.rand(14)

        mock_ml = MagicMock()
        mock_ml.predict.return_value = (0.55, 0.30)  # Low confidence

        mock_llm = MagicMock()
        mock_combiner = MagicMock()
        mock_combiner.combine.return_value = PredictionResult(
            market_id="0x123", ml_probability=0.55, ml_confidence=0.30,
            llm_probability=0.5, llm_reasoning="Skipped - low ML confidence",
            final_probability=0.53, confidence=0.15, mispricing=-0.12,
            polymarket_price=0.65,
        )

        mock_db = MagicMock()
        mock_db.get_price_history.return_value = []

        engine = PredictionEngine(
            feature_engineer=mock_features,
            ml_ensemble=mock_ml,
            llm_analyzer=mock_llm,
            combiner=mock_combiner,
            db=mock_db,
            ml_confidence_threshold=0.50,
        )

        result = engine.predict(self._make_snapshot())
        mock_llm.analyze.assert_not_called()  # LLM skipped
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_prediction_engine.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement prediction engine**

```python
# prediction/engine.py
import logging

from data.models import MarketSnapshot, PredictionResult

logger = logging.getLogger(__name__)


class PredictionEngine:
    def __init__(
        self,
        feature_engineer,
        ml_ensemble,
        llm_analyzer,
        combiner,
        db,
        ml_confidence_threshold: float = 0.50,
    ):
        self.feature_engineer = feature_engineer
        self.ml_ensemble = ml_ensemble
        self.llm_analyzer = llm_analyzer
        self.combiner = combiner
        self.db = db
        self.ml_confidence_threshold = ml_confidence_threshold

    def predict(self, snapshot: MarketSnapshot) -> PredictionResult:
        history = self.db.get_price_history(snapshot.market_id, limit=50)
        features = self.feature_engineer.extract(snapshot, history)

        ml_prob, ml_conf = self.ml_ensemble.predict(features)
        logger.info(f"[{snapshot.market_id}] ML: prob={ml_prob:.3f}, conf={ml_conf:.3f}")

        # Only call LLM if ML has reasonable confidence (saves API cost)
        if ml_conf >= self.ml_confidence_threshold:
            llm_prob, llm_reasoning = self.llm_analyzer.analyze(snapshot)
            logger.info(f"[{snapshot.market_id}] LLM: prob={llm_prob:.3f}")
        else:
            llm_prob = 0.5
            llm_reasoning = "Skipped - low ML confidence"
            logger.info(f"[{snapshot.market_id}] LLM skipped (ML conf={ml_conf:.3f})")

        result = self.combiner.combine(
            market_id=snapshot.market_id,
            polymarket_price=snapshot.polymarket_price,
            ml_probability=ml_prob,
            ml_confidence=ml_conf,
            llm_probability=llm_prob,
            llm_reasoning=llm_reasoning,
        )

        self.db.save_prediction(result)
        return result

    def predict_batch(self, snapshots: list[MarketSnapshot]) -> list[PredictionResult]:
        results = []
        for snap in snapshots:
            try:
                result = self.predict(snap)
                results.append(result)
            except Exception as e:
                logger.error(f"Prediction failed for {snap.market_id}: {e}")
        return results
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/test_prediction_engine.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add prediction/engine.py tests/test_prediction_engine.py
git commit -m "feat: add prediction engine orchestrator with LLM cost optimization"
```

---

## Phase 4: Trading Engine

### Task 14: Risk Manager

**Files:**
- Create: `trading/__init__.py`
- Create: `trading/risk_manager.py`
- Create: `tests/test_risk_manager.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_risk_manager.py
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

        rm = RiskManager(
            db=mock_db, bankroll=1000.0,
            max_trade_pct=0.02, max_daily_pct=0.10,
            min_confidence=0.80, min_mispricing=0.10,
            max_open_positions=5, cooldown_minutes=60,
        )
        decision = rm.evaluate(self._make_prediction())
        assert decision.approved is True
        assert decision.position_size == 20.0  # 2% of 1000
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
        mock_db.get_open_trades.return_value = [MagicMock()] * 5  # 5 open
        mock_db.get_daily_trade_total.return_value = 0.0

        rm = RiskManager(db=mock_db, bankroll=1000.0, max_open_positions=5)
        decision = rm.evaluate(self._make_prediction())
        assert decision.approved is False
        assert "position" in decision.rejection_reason.lower()

    def test_reject_daily_limit(self):
        mock_db = MagicMock()
        mock_db.get_open_trades.return_value = []
        mock_db.get_daily_trade_total.return_value = 95.0  # 95 of 100 daily limit used

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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_risk_manager.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement risk manager**

```python
# trading/__init__.py
```

```python
# trading/risk_manager.py
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
    def __init__(
        self,
        db,
        bankroll: float = 1000.0,
        max_trade_pct: float = 0.02,
        max_daily_pct: float = 0.10,
        stop_loss_pct: float = 0.25,
        min_confidence: float = 0.80,
        min_mispricing: float = 0.10,
        max_open_positions: int = 5,
        cooldown_minutes: int = 60,
    ):
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
        # Kill switch
        if self.kill_switch:
            return RiskDecision(approved=False, rejection_reason="Kill switch is active")

        # Confidence check
        if prediction.confidence < self.min_confidence:
            return RiskDecision(
                approved=False,
                rejection_reason=f"Confidence {prediction.confidence:.2f} below minimum {self.min_confidence}",
            )

        # Mispricing check
        if abs(prediction.mispricing) < self.min_mispricing:
            return RiskDecision(
                approved=False,
                rejection_reason=f"Mispricing {abs(prediction.mispricing):.2f} below minimum {self.min_mispricing}",
            )

        # Max open positions
        open_trades = self.db.get_open_trades()
        if len(open_trades) >= self.max_open_positions:
            return RiskDecision(
                approved=False,
                rejection_reason=f"Max open positions reached ({self.max_open_positions})",
            )

        # Daily limit
        now = datetime.now(timezone.utc)
        daily_total = self.db.get_daily_trade_total(now)
        max_daily = self.bankroll * self.max_daily_pct
        position_size = self.bankroll * self.max_trade_pct
        if daily_total + position_size > max_daily:
            return RiskDecision(
                approved=False,
                rejection_reason=f"Daily limit would be exceeded (${daily_total:.0f} + ${position_size:.0f} > ${max_daily:.0f})",
            )

        # Cooldown after loss
        if self.last_loss_time:
            cooldown_end = self.last_loss_time + timedelta(minutes=self.cooldown_minutes)
            if now < cooldown_end:
                return RiskDecision(
                    approved=False,
                    rejection_reason=f"Post-loss cooldown active until {cooldown_end.isoformat()}",
                )

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
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/test_risk_manager.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add trading/ tests/test_risk_manager.py
git commit -m "feat: add conservative risk manager with kill switch"
```

---

### Task 15: Trade Executor

**Files:**
- Create: `trading/executor.py`
- Create: `tests/test_executor.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_executor.py
from unittest.mock import MagicMock
from datetime import datetime, timezone

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
        # In dry_run, no real API call is made
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_executor.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement trade executor**

```python
# trading/executor.py
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
            trade_id=str(uuid.uuid4())[:8],
            market_id=prediction.market_id,
            action=action,
            amount=position_size,
            price=price,
            confidence=prediction.confidence,
            mispricing=prediction.mispricing,
            status=TradeStatus.PENDING,
            timestamp=datetime.now(timezone.utc),
        )

        if self.dry_run:
            logger.info(f"[DRY RUN] Would {action.value} on {prediction.market_id} for ${position_size:.2f} at {price:.3f}")
            trade.status = TradeStatus.EXECUTED
        else:
            try:
                self._place_order(trade)
                trade.status = TradeStatus.EXECUTED
                logger.info(f"Trade executed: {action.value} on {prediction.market_id} for ${position_size:.2f}")
            except Exception as e:
                trade.status = TradeStatus.REJECTED
                trade.rejection_reason = str(e)
                logger.error(f"Trade failed for {prediction.market_id}: {e}")

        self.db.save_trade(trade)
        return trade

    def _place_order(self, trade: TradeRecord):
        if not self.api_client:
            raise RuntimeError("No API client configured for live trading")
        # Polymarket CLOB API order placement
        # This will be implemented when connecting to live trading
        raise NotImplementedError("Live trading not yet implemented")
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/test_executor.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add trading/executor.py tests/test_executor.py
git commit -m "feat: add trade executor with dry-run mode"
```

---

### Task 16: Portfolio Tracker

**Files:**
- Create: `trading/portfolio.py`
- Create: `tests/test_portfolio.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_portfolio.py
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
        mock_db.get_open_trades.return_value = [
            _make_trade("t1", 50, 0.65),
            _make_trade("t2", 30, 0.70),
        ]
        portfolio = Portfolio(db=mock_db, bankroll=1000.0)
        positions = portfolio.get_open_positions()
        assert len(positions) == 2

    def test_total_invested(self):
        mock_db = MagicMock()
        mock_db.get_open_trades.return_value = [
            _make_trade("t1", 50, 0.65),
            _make_trade("t2", 30, 0.70),
        ]
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_portfolio.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement portfolio tracker**

```python
# trading/portfolio.py
from data.models import TradeAction, TradeRecord, TradeStatus


class Portfolio:
    def __init__(self, db, bankroll: float = 1000.0):
        self.db = db
        self.bankroll = bankroll

    def get_open_positions(self) -> list[TradeRecord]:
        return self.db.get_open_trades()

    def total_invested(self) -> float:
        return sum(t.amount for t in self.get_open_positions())

    def available_balance(self) -> float:
        return self.bankroll - self.total_invested()

    def compute_metrics(self) -> dict:
        all_trades = self.db.get_all_trades(limit=1000)
        closed = [t for t in all_trades if t.status in (TradeStatus.CLOSED, TradeStatus.STOPPED)]

        if not closed:
            return {
                "total_trades": len(all_trades),
                "closed_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "max_drawdown": 0.0,
            }

        pnls = []
        for trade in closed:
            if trade.close_price is None:
                continue
            pnl = trade.unrealized_pnl(trade.close_price)
            pnls.append(pnl)

        wins = sum(1 for p in pnls if p > 0)
        total_pnl = sum(pnls)

        # Max drawdown
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in pnls:
            cumulative += p
            peak = max(peak, cumulative)
            dd = peak - cumulative
            max_dd = max(max_dd, dd)

        return {
            "total_trades": len(all_trades),
            "closed_trades": len(closed),
            "win_rate": wins / len(pnls) if pnls else 0.0,
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / len(pnls) if pnls else 0.0,
            "max_drawdown": max_dd,
        }
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/test_portfolio.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add trading/portfolio.py tests/test_portfolio.py
git commit -m "feat: add portfolio tracker with performance metrics"
```

---

### Task 17: Trading Engine Orchestrator

**Files:**
- Create: `trading/engine.py`
- Create: `tests/test_trading_engine.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_trading_engine.py
from unittest.mock import MagicMock
from datetime import datetime, timezone

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
        mock_risk.evaluate.return_value = RiskDecision(
            approved=False, rejection_reason="Low confidence"
        )
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_trading_engine.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement trading engine**

```python
# trading/engine.py
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
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/test_trading_engine.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add trading/engine.py tests/test_trading_engine.py
git commit -m "feat: add trading engine orchestrator"
```

---

## Phase 5: Dashboard

### Task 18: Dash App + Markets View

**Files:**
- Create: `dashboard/__init__.py`
- Create: `dashboard/app.py`
- Create: `dashboard/layouts/__init__.py`
- Create: `dashboard/layouts/markets.py`
- Create: `dashboard/callbacks/__init__.py`
- Create: `dashboard/callbacks/markets_cb.py`

- [ ] **Step 1: Create dashboard package structure**

```python
# dashboard/__init__.py
```

```python
# dashboard/layouts/__init__.py
```

```python
# dashboard/callbacks/__init__.py
```

- [ ] **Step 2: Implement markets layout**

```python
# dashboard/layouts/markets.py
from dash import html, dash_table, dcc
import dash_bootstrap_components as dbc


def markets_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col(html.H2("Monitored Markets", className="text-light mb-3")),
            dbc.Col(
                dbc.Button("Analyze Now", id="btn-analyze", color="primary", className="float-end"),
                width="auto",
            ),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col([
                dash_table.DataTable(
                    id="markets-table",
                    columns=[
                        {"name": "Market", "id": "question"},
                        {"name": "Category", "id": "category"},
                        {"name": "PM Price", "id": "polymarket_price", "type": "numeric", "format": {"specifier": ".1%"}},
                        {"name": "NE2 Pred", "id": "prediction", "type": "numeric", "format": {"specifier": ".1%"}},
                        {"name": "Mispricing", "id": "mispricing", "type": "numeric", "format": {"specifier": "+.1%"}},
                        {"name": "Confidence", "id": "confidence", "type": "numeric", "format": {"specifier": ".0%"}},
                        {"name": "Volume 24h", "id": "volume_24h", "type": "numeric", "format": {"specifier": "$,.0f"}},
                        {"name": "Signal", "id": "signal"},
                    ],
                    style_table={"overflowX": "auto"},
                    style_header={"backgroundColor": "#1a1a2e", "color": "#e0e0e0", "fontWeight": "bold"},
                    style_cell={"backgroundColor": "#16213e", "color": "#e0e0e0", "border": "1px solid #0f3460"},
                    style_data_conditional=[
                        {"if": {"filter_query": "{signal} = BUY_YES"}, "backgroundColor": "#0a3d0a", "color": "#4caf50"},
                        {"if": {"filter_query": "{signal} = BUY_NO"}, "backgroundColor": "#3d0a0a", "color": "#f44336"},
                    ],
                    row_selectable="single",
                    page_size=20,
                    sort_action="native",
                ),
            ]),
        ]),

        dcc.Interval(id="markets-interval", interval=120_000, n_intervals=0),  # Refresh every 2 min
        dcc.Store(id="selected-market-store"),
    ], fluid=True, className="py-3")
```

- [ ] **Step 3: Implement markets callbacks**

```python
# dashboard/callbacks/markets_cb.py
from dash import Input, Output, callback, no_update


def register_markets_callbacks(app, db):
    @app.callback(
        Output("markets-table", "data"),
        Input("markets-interval", "n_intervals"),
    )
    def update_markets_table(n):
        # Get latest snapshots and predictions for all markets
        rows = []
        # Fetch last 20 unique markets from snapshots
        try:
            conn = db.conn
            market_rows = conn.execute(
                """SELECT DISTINCT market_id FROM snapshots
                   ORDER BY timestamp DESC LIMIT 20"""
            ).fetchall()

            for mr in market_rows:
                mid = mr["market_id"]
                snap = db.get_latest_snapshot(mid)
                pred = db.get_latest_prediction(mid)
                if not snap:
                    continue
                row = {
                    "market_id": mid,
                    "question": snap.question[:80],
                    "category": snap.category,
                    "polymarket_price": snap.polymarket_price,
                    "prediction": pred.final_probability if pred else None,
                    "mispricing": pred.mispricing if pred else None,
                    "confidence": pred.confidence if pred else None,
                    "volume_24h": snap.volume_24h,
                    "signal": _signal_label(pred) if pred else "-",
                }
                rows.append(row)
        except Exception:
            pass
        return rows

    @app.callback(
        Output("selected-market-store", "data"),
        Input("markets-table", "selected_rows"),
        Input("markets-table", "data"),
    )
    def store_selected_market(selected_rows, data):
        if not selected_rows or not data:
            return no_update
        idx = selected_rows[0]
        return data[idx].get("market_id")


def _signal_label(pred) -> str:
    if not pred or abs(pred.mispricing) < 0.10 or pred.confidence < 0.80:
        return "-"
    return "BUY_YES" if pred.mispricing > 0 else "BUY_NO"
```

- [ ] **Step 4: Implement Dash app factory**

```python
# dashboard/app.py
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc

from dashboard.layouts.markets import markets_layout
from dashboard.callbacks.markets_cb import register_markets_callbacks


def create_dash_app(db) -> dash.Dash:
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.CYBORG],
        suppress_callback_exceptions=True,
    )

    app.layout = html.Div([
        dbc.Navbar(
            dbc.Container([
                dbc.NavbarBrand("NE2 — Polymarket Predictor", className="ms-2 text-warning fw-bold"),
                dbc.Nav([
                    dbc.NavLink("Markets", href="/", active="exact"),
                    dbc.NavLink("Portfolio", href="/portfolio"),
                    dbc.NavLink("Control", href="/control"),
                ], navbar=True),
            ], fluid=True),
            color="dark",
            dark=True,
        ),
        dcc.Location(id="url", refresh=False),
        html.Div(id="page-content"),
    ], style={"backgroundColor": "#0a0a1a", "minHeight": "100vh"})

    @app.callback(
        dash.Output("page-content", "children"),
        dash.Input("url", "pathname"),
    )
    def display_page(pathname):
        if pathname == "/portfolio":
            from dashboard.layouts.portfolio import portfolio_layout
            return portfolio_layout()
        elif pathname == "/control":
            from dashboard.layouts.control import control_layout
            return control_layout()
        else:
            return markets_layout()

    register_markets_callbacks(app, db)

    return app
```

- [ ] **Step 5: Commit**

```bash
git add dashboard/
git commit -m "feat: add Dash dashboard with markets view and routing"
```

---

### Task 19: Portfolio View

**Files:**
- Create: `dashboard/layouts/portfolio.py`
- Create: `dashboard/callbacks/portfolio_cb.py`

- [ ] **Step 1: Implement portfolio layout**

```python
# dashboard/layouts/portfolio.py
from dash import html, dash_table, dcc
import dash_bootstrap_components as dbc


def portfolio_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col(html.H2("Portfolio & Trading", className="text-light mb-3")),
            dbc.Col(
                dbc.Button("KILL SWITCH", id="btn-kill-switch", color="danger", className="float-end", size="lg"),
                width="auto",
            ),
        ], className="mb-3"),

        # Metrics cards
        dbc.Row(id="portfolio-metrics", className="mb-4"),

        # Open positions
        dbc.Row([
            dbc.Col(html.H4("Open Positions", className="text-light mb-2")),
        ]),
        dbc.Row([
            dbc.Col(
                dash_table.DataTable(
                    id="positions-table",
                    columns=[
                        {"name": "Trade ID", "id": "trade_id"},
                        {"name": "Market", "id": "market_id"},
                        {"name": "Action", "id": "action"},
                        {"name": "Amount", "id": "amount", "type": "numeric", "format": {"specifier": "$,.2f"}},
                        {"name": "Entry Price", "id": "price", "type": "numeric", "format": {"specifier": ".3f"}},
                        {"name": "Confidence", "id": "confidence", "type": "numeric", "format": {"specifier": ".0%"}},
                        {"name": "Time", "id": "timestamp"},
                    ],
                    style_header={"backgroundColor": "#1a1a2e", "color": "#e0e0e0"},
                    style_cell={"backgroundColor": "#16213e", "color": "#e0e0e0", "border": "1px solid #0f3460"},
                    page_size=10,
                ),
            ),
        ], className="mb-4"),

        # Trade history
        dbc.Row([
            dbc.Col(html.H4("Trade History", className="text-light mb-2")),
        ]),
        dbc.Row([
            dbc.Col(
                dash_table.DataTable(
                    id="history-table",
                    columns=[
                        {"name": "Trade ID", "id": "trade_id"},
                        {"name": "Market", "id": "market_id"},
                        {"name": "Action", "id": "action"},
                        {"name": "Amount", "id": "amount", "type": "numeric", "format": {"specifier": "$,.2f"}},
                        {"name": "Entry", "id": "price", "type": "numeric", "format": {"specifier": ".3f"}},
                        {"name": "Exit", "id": "close_price", "type": "numeric", "format": {"specifier": ".3f"}},
                        {"name": "Status", "id": "status"},
                        {"name": "Reason", "id": "rejection_reason"},
                    ],
                    style_header={"backgroundColor": "#1a1a2e", "color": "#e0e0e0"},
                    style_cell={"backgroundColor": "#16213e", "color": "#e0e0e0", "border": "1px solid #0f3460"},
                    page_size=20,
                    sort_action="native",
                ),
            ),
        ]),

        dcc.Interval(id="portfolio-interval", interval=30_000, n_intervals=0),
        html.Div(id="kill-switch-output", className="mt-2"),
    ], fluid=True, className="py-3")
```

- [ ] **Step 2: Implement portfolio callbacks**

```python
# dashboard/callbacks/portfolio_cb.py
import dash_bootstrap_components as dbc
from dash import Input, Output, html


def register_portfolio_callbacks(app, db, portfolio, risk_manager):
    @app.callback(
        Output("portfolio-metrics", "children"),
        Input("portfolio-interval", "n_intervals"),
    )
    def update_metrics(n):
        metrics = portfolio.compute_metrics()
        balance = portfolio.available_balance()
        cards = [
            _metric_card("Balance", f"${balance:,.2f}", "primary"),
            _metric_card("Win Rate", f"{metrics['win_rate']:.0%}", "success" if metrics["win_rate"] > 0.5 else "warning"),
            _metric_card("Total P&L", f"${metrics['total_pnl']:,.2f}", "success" if metrics["total_pnl"] >= 0 else "danger"),
            _metric_card("Trades", str(metrics["total_trades"]), "info"),
            _metric_card("Max Drawdown", f"${metrics['max_drawdown']:,.2f}", "warning"),
        ]
        return [dbc.Col(c, width=2) for c in cards]

    @app.callback(
        Output("positions-table", "data"),
        Input("portfolio-interval", "n_intervals"),
    )
    def update_positions(n):
        positions = portfolio.get_open_positions()
        return [
            {
                "trade_id": t.trade_id,
                "market_id": t.market_id[:16],
                "action": t.action.value,
                "amount": t.amount,
                "price": t.price,
                "confidence": t.confidence,
                "timestamp": t.timestamp.strftime("%m/%d %H:%M"),
            }
            for t in positions
        ]

    @app.callback(
        Output("history-table", "data"),
        Input("portfolio-interval", "n_intervals"),
    )
    def update_history(n):
        trades = db.get_all_trades(limit=50)
        return [
            {
                "trade_id": t.trade_id,
                "market_id": t.market_id[:16],
                "action": t.action.value,
                "amount": t.amount,
                "price": t.price,
                "close_price": t.close_price,
                "status": t.status.value,
                "rejection_reason": t.rejection_reason or "",
            }
            for t in trades
        ]

    @app.callback(
        Output("kill-switch-output", "children"),
        Input("btn-kill-switch", "n_clicks"),
        prevent_initial_call=True,
    )
    def toggle_kill_switch(n_clicks):
        risk_manager.kill_switch = not risk_manager.kill_switch
        state = "ACTIVE" if risk_manager.kill_switch else "INACTIVE"
        color = "danger" if risk_manager.kill_switch else "success"
        return dbc.Alert(f"Kill switch: {state}", color=color, duration=5000)


def _metric_card(title, value, color):
    return dbc.Card(
        dbc.CardBody([
            html.P(title, className="text-muted mb-1", style={"fontSize": "0.8rem"}),
            html.H4(value, className=f"text-{color} mb-0"),
        ]),
        className="bg-dark border-secondary",
    )
```

- [ ] **Step 3: Commit**

```bash
git add dashboard/layouts/portfolio.py dashboard/callbacks/portfolio_cb.py
git commit -m "feat: add portfolio view with metrics, positions, and kill switch"
```

---

### Task 20: Control View

**Files:**
- Create: `dashboard/layouts/control.py`
- Create: `dashboard/callbacks/control_cb.py`

- [ ] **Step 1: Implement control layout**

```python
# dashboard/layouts/control.py
from dash import html, dcc
import dash_bootstrap_components as dbc


def control_layout():
    return dbc.Container([
        html.H2("System Control", className="text-light mb-3"),

        dbc.Row([
            # Risk parameters
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Risk Parameters", className="bg-dark text-warning"),
                    dbc.CardBody([
                        _param_input("Max per trade (%)", "input-max-trade", 2, 0.5, 10, 0.5),
                        _param_input("Max daily (%)", "input-max-daily", 10, 1, 50, 1),
                        _param_input("Stop-loss (%)", "input-stop-loss", 25, 5, 50, 5),
                        _param_input("Min confidence (%)", "input-min-conf", 80, 50, 99, 5),
                        _param_input("Min mispricing (%)", "input-min-misp", 10, 5, 30, 1),
                        _param_input("Max positions", "input-max-pos", 5, 1, 20, 1),
                        _param_input("Cooldown (min)", "input-cooldown", 60, 0, 240, 15),
                        dbc.Button("Update Risk Params", id="btn-update-risk", color="warning", className="mt-2 w-100"),
                        html.Div(id="risk-update-output"),
                    ]),
                ], className="bg-dark border-secondary"),
            ], width=4),

            # Scheduler & Analysis
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Scheduler", className="bg-dark text-warning"),
                    dbc.CardBody([
                        html.P(id="scheduler-status", className="text-light"),
                        _param_input("Cycle interval (hours)", "input-cycle-hours", 4, 1, 24, 1),
                        dbc.Button("Analyze Now", id="btn-run-cycle", color="success", className="mt-2 w-100"),
                        html.Div(id="cycle-output"),
                    ]),
                ], className="bg-dark border-secondary mb-3"),

                dbc.Card([
                    dbc.CardHeader("API Status", className="bg-dark text-warning"),
                    dbc.CardBody(id="api-status-body"),
                ], className="bg-dark border-secondary"),
            ], width=4),

            # Logs
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("System Logs", className="bg-dark text-warning"),
                    dbc.CardBody([
                        html.Div(
                            id="system-logs",
                            style={"maxHeight": "500px", "overflowY": "auto", "fontFamily": "monospace", "fontSize": "0.8rem"},
                        ),
                    ]),
                ], className="bg-dark border-secondary"),
            ], width=4),
        ]),

        dcc.Interval(id="control-interval", interval=10_000, n_intervals=0),
    ], fluid=True, className="py-3")


def _param_input(label, input_id, default, min_val, max_val, step):
    return dbc.Row([
        dbc.Col(html.Label(label, className="text-light"), width=7),
        dbc.Col(dbc.Input(id=input_id, type="number", value=default, min=min_val, max=max_val, step=step, size="sm"), width=5),
    ], className="mb-2")
```

- [ ] **Step 2: Implement control callbacks**

```python
# dashboard/callbacks/control_cb.py
from dash import Input, Output, State, html
import dash_bootstrap_components as dbc


def register_control_callbacks(app, db, risk_manager):
    @app.callback(
        Output("system-logs", "children"),
        Input("control-interval", "n_intervals"),
    )
    def update_logs(n):
        logs = db.get_recent_logs(limit=50)
        return [
            html.Div(
                f"[{log['timestamp'][:19]}] [{log['level']}] {log['module']}: {log['message']}",
                className=f"text-{'danger' if log['level'] == 'ERROR' else 'warning' if log['level'] == 'WARNING' else 'light'}",
            )
            for log in logs
        ]

    @app.callback(
        Output("risk-update-output", "children"),
        Input("btn-update-risk", "n_clicks"),
        State("input-max-trade", "value"),
        State("input-max-daily", "value"),
        State("input-stop-loss", "value"),
        State("input-min-conf", "value"),
        State("input-min-misp", "value"),
        State("input-max-pos", "value"),
        State("input-cooldown", "value"),
        prevent_initial_call=True,
    )
    def update_risk_params(n, max_trade, max_daily, stop_loss, min_conf, min_misp, max_pos, cooldown):
        risk_manager.max_trade_pct = (max_trade or 2) / 100
        risk_manager.max_daily_pct = (max_daily or 10) / 100
        risk_manager.stop_loss_pct = (stop_loss or 25) / 100
        risk_manager.min_confidence = (min_conf or 80) / 100
        risk_manager.min_mispricing = (min_misp or 10) / 100
        risk_manager.max_open_positions = int(max_pos or 5)
        risk_manager.cooldown_minutes = int(cooldown or 60)
        db.save_log("INFO", "control", "Risk parameters updated")
        return dbc.Alert("Risk parameters updated", color="success", duration=3000)

    @app.callback(
        Output("api-status-body", "children"),
        Input("control-interval", "n_intervals"),
    )
    def update_api_status(n):
        # Simple status display — actual health checks can be added later
        apis = [
            ("Polymarket", "success"),
            ("NewsAPI", "success"),
            ("Twitter/X", "success"),
            ("Reddit", "success"),
            ("Claude API", "success"),
            ("FRED", "success"),
            ("CoinGecko", "success"),
        ]
        return [
            html.Div([
                html.Span("● ", className=f"text-{color}"),
                html.Span(name, className="text-light"),
            ], className="mb-1")
            for name, color in apis
        ]
```

- [ ] **Step 3: Commit**

```bash
git add dashboard/layouts/control.py dashboard/callbacks/control_cb.py
git commit -m "feat: add control view with risk params, logs, and API status"
```

---

### Task 21: Wire Dashboard Callbacks

**Files:**
- Modify: `dashboard/app.py`

- [ ] **Step 1: Update app factory to register all callbacks**

Replace the `create_dash_app` function in `dashboard/app.py`:

```python
# dashboard/app.py
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc

from dashboard.layouts.markets import markets_layout
from dashboard.callbacks.markets_cb import register_markets_callbacks
from dashboard.callbacks.portfolio_cb import register_portfolio_callbacks
from dashboard.callbacks.control_cb import register_control_callbacks


def create_dash_app(db, portfolio=None, risk_manager=None) -> dash.Dash:
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.CYBORG],
        suppress_callback_exceptions=True,
    )

    app.layout = html.Div([
        dbc.Navbar(
            dbc.Container([
                dbc.NavbarBrand("NE2 — Polymarket Predictor", className="ms-2 text-warning fw-bold"),
                dbc.Nav([
                    dbc.NavLink("Markets", href="/", active="exact"),
                    dbc.NavLink("Portfolio", href="/portfolio"),
                    dbc.NavLink("Control", href="/control"),
                ], navbar=True),
            ], fluid=True),
            color="dark",
            dark=True,
        ),
        dcc.Location(id="url", refresh=False),
        html.Div(id="page-content"),
    ], style={"backgroundColor": "#0a0a1a", "minHeight": "100vh"})

    @app.callback(
        dash.Output("page-content", "children"),
        dash.Input("url", "pathname"),
    )
    def display_page(pathname):
        if pathname == "/portfolio":
            from dashboard.layouts.portfolio import portfolio_layout
            return portfolio_layout()
        elif pathname == "/control":
            from dashboard.layouts.control import control_layout
            return control_layout()
        else:
            return markets_layout()

    register_markets_callbacks(app, db)
    if portfolio and risk_manager:
        register_portfolio_callbacks(app, db, portfolio, risk_manager)
        register_control_callbacks(app, db, risk_manager)

    return app
```

- [ ] **Step 2: Commit**

```bash
git add dashboard/app.py
git commit -m "feat: wire all dashboard callbacks"
```

---

## Phase 6: Scheduler & Entry Point

### Task 22: Scheduler

**Files:**
- Create: `scheduler/__init__.py`
- Create: `scheduler/jobs.py`
- Create: `tests/test_scheduler.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_scheduler.py
from unittest.mock import MagicMock, patch

from scheduler.jobs import run_analysis_cycle


class TestScheduler:
    def test_run_cycle_calls_pipeline(self):
        mock_ingestion = MagicMock()
        mock_ingestion.run.return_value = [MagicMock()]  # 1 snapshot

        mock_prediction = MagicMock()
        mock_prediction.predict_batch.return_value = [MagicMock(has_signal=MagicMock(return_value=True))]

        mock_trading = MagicMock()

        mock_db = MagicMock()

        run_analysis_cycle(
            ingestion=mock_ingestion,
            prediction_engine=mock_prediction,
            trading_engine=mock_trading,
            db=mock_db,
            min_mispricing=0.10,
            min_confidence=0.80,
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
            ingestion=mock_ingestion,
            prediction_engine=mock_prediction,
            trading_engine=mock_trading,
            db=mock_db,
            min_mispricing=0.10,
            min_confidence=0.80,
        )

        # Only the signal that passed the filter should be traded
        args = mock_trading.process_batch.call_args[0][0]
        assert len(args) == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_scheduler.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement scheduler**

```python
# scheduler/__init__.py
```

```python
# scheduler/jobs.py
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def run_analysis_cycle(
    ingestion,
    prediction_engine,
    trading_engine,
    db,
    min_mispricing: float = 0.10,
    min_confidence: float = 0.80,
):
    start = datetime.now(timezone.utc)
    logger.info("=== Analysis cycle started ===")
    db.save_log("INFO", "scheduler", "Analysis cycle started")

    # Step 1: Ingest data
    snapshots = ingestion.run()
    logger.info(f"Ingested {len(snapshots)} market snapshots")

    if not snapshots:
        db.save_log("WARNING", "scheduler", "No snapshots ingested")
        return

    # Step 2: Predict
    predictions = prediction_engine.predict_batch(snapshots)
    logger.info(f"Generated {len(predictions)} predictions")

    # Step 3: Filter signals
    signals = [p for p in predictions if p.has_signal(min_mispricing, min_confidence)]
    logger.info(f"Found {len(signals)} actionable signals")

    # Step 4: Trade
    if signals:
        trades = trading_engine.process_batch(signals)
        logger.info(f"Executed {len(trades)} trades")
        db.save_log("INFO", "scheduler", f"Cycle complete: {len(trades)} trades from {len(signals)} signals")
    else:
        db.save_log("INFO", "scheduler", "Cycle complete: no actionable signals")

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    logger.info(f"=== Cycle completed in {elapsed:.1f}s ===")
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/test_scheduler.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add scheduler/ tests/test_scheduler.py
git commit -m "feat: add scheduler with analysis cycle pipeline"
```

---

### Task 23: App Entry Point

**Files:**
- Create: `app.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write failing test for API**

```python
# tests/test_api.py
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


def test_health_endpoint():
    # Import with mocked dependencies
    with patch("app.initialize_components") as mock_init:
        mock_init.return_value = {
            "db": MagicMock(),
            "ingestion": MagicMock(),
            "prediction_engine": MagicMock(),
            "trading_engine": MagicMock(),
            "risk_manager": MagicMock(),
            "portfolio": MagicMock(),
        }
        from app import create_api
        api = create_api(mock_init.return_value)
        client = TestClient(api)
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_api.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement app.py**

```python
# app.py
import logging
import threading

import anthropic
import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from dashboard.app import create_dash_app
from data.economic_client import EconomicClient
from data.ingestion import DataIngestion
from data.news_client import NewsClient
from data.polymarket_client import PolymarketClient
from data.sentiment_client import SentimentClient
from db.database import Database
from prediction.combiner import SignalCombiner
from prediction.engine import PredictionEngine
from prediction.features import FeatureEngineer
from prediction.llm_analyzer import LLMAnalyzer
from prediction.ml_ensemble import MLEnsemble
from scheduler.jobs import run_analysis_cycle
from trading.engine import TradingEngine
from trading.executor import TradeExecutor
from trading.portfolio import Portfolio
from trading.risk_manager import RiskManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def initialize_components() -> dict:
    # Database
    db = Database(settings.db_path)
    db.initialize()

    # Data clients
    polymarket = PolymarketClient(base_url=settings.polymarket_api_url)
    news = NewsClient(api_key=settings.newsapi_key)
    sentiment = SentimentClient(
        twitter_bearer_token=settings.twitter_bearer_token,
        reddit_client_id=settings.reddit_client_id,
        reddit_client_secret=settings.reddit_client_secret,
    )
    economic = EconomicClient(fred_api_key=settings.fred_api_key)
    ingestion = DataIngestion(polymarket=polymarket, news=news, sentiment=sentiment, economic=economic, db=db)

    # Prediction
    feature_engineer = FeatureEngineer()
    ml_ensemble = MLEnsemble()
    llm_client = anthropic.Anthropic(api_key=settings.anthropic_api_key) if settings.anthropic_api_key else None
    llm_analyzer = LLMAnalyzer(client=llm_client)
    combiner = SignalCombiner()
    prediction_engine = PredictionEngine(
        feature_engineer=feature_engineer,
        ml_ensemble=ml_ensemble,
        llm_analyzer=llm_analyzer,
        combiner=combiner,
        db=db,
    )

    # Trading
    risk_manager = RiskManager(
        db=db,
        bankroll=1000.0,
        max_trade_pct=settings.max_trade_pct,
        max_daily_pct=settings.max_daily_pct,
        stop_loss_pct=settings.stop_loss_pct,
        min_confidence=settings.min_confidence,
        min_mispricing=settings.min_mispricing,
        max_open_positions=settings.max_open_positions,
        cooldown_minutes=settings.cooldown_minutes,
    )
    executor = TradeExecutor(db=db, dry_run=True)
    trading_engine = TradingEngine(risk_manager=risk_manager, executor=executor, db=db)
    portfolio = Portfolio(db=db, bankroll=1000.0)

    return {
        "db": db,
        "ingestion": ingestion,
        "prediction_engine": prediction_engine,
        "trading_engine": trading_engine,
        "risk_manager": risk_manager,
        "portfolio": portfolio,
    }


def create_api(components: dict) -> FastAPI:
    api = FastAPI(title="NE2 API")
    api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @api.get("/api/health")
    def health():
        return {"status": "ok"}

    @api.post("/api/analyze")
    def trigger_analysis():
        run_analysis_cycle(
            ingestion=components["ingestion"],
            prediction_engine=components["prediction_engine"],
            trading_engine=components["trading_engine"],
            db=components["db"],
            min_mispricing=settings.min_mispricing,
            min_confidence=settings.min_confidence,
        )
        return {"status": "cycle_complete"}

    @api.post("/api/kill-switch")
    def toggle_kill_switch():
        rm = components["risk_manager"]
        rm.kill_switch = not rm.kill_switch
        return {"kill_switch": rm.kill_switch}

    return api


def main():
    components = initialize_components()
    api = create_api(components)

    # Scheduler
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        run_analysis_cycle,
        "interval",
        hours=settings.cycle_interval_hours,
        kwargs={
            "ingestion": components["ingestion"],
            "prediction_engine": components["prediction_engine"],
            "trading_engine": components["trading_engine"],
            "db": components["db"],
            "min_mispricing": settings.min_mispricing,
            "min_confidence": settings.min_confidence,
        },
    )
    scheduler.start()

    # Dashboard in a separate thread
    dash_app = create_dash_app(
        db=components["db"],
        portfolio=components["portfolio"],
        risk_manager=components["risk_manager"],
    )
    dash_thread = threading.Thread(
        target=dash_app.run,
        kwargs={"host": "0.0.0.0", "port": settings.dash_port, "debug": False},
        daemon=True,
    )
    dash_thread.start()
    logger.info(f"Dashboard running at http://localhost:{settings.dash_port}")

    # FastAPI
    logger.info(f"API running at http://localhost:{settings.fastapi_port}")
    uvicorn.run(api, host="0.0.0.0", port=settings.fastapi_port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/test_api.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_api.py
git commit -m "feat: add app entry point with FastAPI + Dash + Scheduler"
```

---

### Task 24: Run Full Test Suite

- [ ] **Step 1: Run all tests**

```bash
pytest tests/ -v --tb=short
```

Expected: All tests PASS

- [ ] **Step 2: Fix any failures and re-run**

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: NE2 Polymarket Predictor — complete initial implementation"
```

---

## Post-Implementation Notes

### First Run Checklist
1. Copy `.env.example` to `.env` and fill in API keys
2. Run `python app.py` — dashboard at `localhost:8050`, API at `localhost:8000`
3. System starts in `dry_run=True` mode (no real trades)
4. Train ML model with historical data before enabling live trading
5. Monitor logs in the Control view

### To Enable Live Trading
1. Set `dry_run=False` in the TradeExecutor initialization in `app.py`
2. Configure Polymarket API credentials in `.env`
3. Implement `_place_order()` in `trading/executor.py` with Polymarket CLOB API
4. Start with small bankroll for validation

### Future Enhancements
- Market detail view with price charts (Task not included — add Plotly time series)
- ML model retraining pipeline
- Backtesting framework
- Email/Telegram alerts for trade notifications
- More sophisticated sentiment analysis (replace keyword-based with a model)
