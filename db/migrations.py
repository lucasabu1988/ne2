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
