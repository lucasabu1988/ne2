# NE2 — Polymarket Predictive Trading System

## Overview

A local Python web application that monitors high-volume Polymarket events, predicts outcomes using a hybrid ML + LLM engine, and executes automated trades when high-confidence mispricing is detected.

**Audience:** Private dashboard for personal use.
**Deployment:** Local machine.
**Budget:** $50-100+/month for premium APIs.

---

## Architecture

Monolith Python application with 5 core modules:

```
┌─────────────────────────────────────────────────────┐
│                   NE2 - Polymarket Predictor         │
├─────────────────────────────────────────────────────┤
│  ┌──────────┐   ┌──────────────┐   ┌─────────────┐ │
│  │  Data     │──▶│  Prediction  │──▶│  Trading    │ │
│  │  Ingestion│   │  Engine      │   │  Engine     │ │
│  └──────────┘   └──────────────┘   └─────────────┘ │
│       │              │                    │          │
│       ▼              ▼                    ▼          │
│  ┌──────────────────────────────────────────────┐   │
│  │              SQLite Database                  │   │
│  └──────────────────────────────────────────────┘   │
│       │              │                    │          │
│       ▼              ▼                    ▼          │
│  ┌──────────────────────────────────────────────┐   │
│  │         Dash Dashboard (localhost:8050)       │   │
│  └──────────────────────────────────────────────┘   │
│                                                      │
│  ┌──────────────┐   ┌──────────────┐                │
│  │  Scheduler   │   │  Risk        │                │
│  │  (APScheduler)│   │  Manager     │                │
│  └──────────────┘   └──────────────┘                │
└─────────────────────────────────────────────────────┘
```

**Stack:**
- Python 3.11+
- FastAPI (internal API + control endpoints)
- Dash/Plotly (dashboard)
- SQLite (storage)
- scikit-learn, XGBoost (ML)
- Claude API (LLM reasoning)
- APScheduler (scheduling)

---

## Module 1: Data Ingestion

Four data sources consolidated into a unified `MarketSnapshot`:

### 1.1 Polymarket (Primary)
- CLOB API for active markets, prices, volume, order book depth
- Polling every 2 minutes for monitored markets
- Auto-filters top N markets by volume/liquidity
- Stores price history for trend detection

### 1.2 News
- NewsAPI or MediaStack (free tiers + premium plans)
- Searches for news relevant to each active market's topic
- Automatic relevance and urgency classification
- Breaking news detection (not yet reflected in price)

### 1.3 Sentiment (Social Media)
- Twitter/X API for mention volume and sentiment
- Reddit API (relevant subreddits by category)
- Normalized sentiment score (-1 to +1) per market
- Sudden sentiment change tracking as signal

### 1.4 Economic/Contextual Data
- FRED API (macroeconomic data, free)
- CoinGecko API (crypto, free)
- Scheduled events calendar (elections, earnings, etc.)

### Unified Data Structure

```python
MarketSnapshot:
    market_id: str
    question: str
    category: str
    polymarket_price: float
    volume_24h: float
    order_book_depth: dict
    news_score: float
    news_count: int
    latest_headlines: list[str]
    sentiment_score: float       # -1 to +1
    sentiment_velocity: float    # rate of change
    economic_indicators: dict
    timestamp: datetime
```

---

## Module 2: Prediction Engine

Hybrid two-layer system — the proprietary core of NE2.

### 2.1 Quantitative Layer (ML Ensemble)

Three models trained on Polymarket historical data + external signals:

- **XGBoost** — Captures non-linear relationships between features
- **Random Forest** — Robust against outliers, stability
- **Logistic Regression** — Interpretable baseline, feature weight insight

Combined via weighted average (weights optimized by backtesting).
Output: `ml_probability` (0-1) + `ml_confidence` (inter-model agreement).

**Key Features:**
- Current price and momentum (1h, 6h, 24h change)
- Volume and volume changes
- Order book imbalance (bids vs asks)
- Sentiment score and velocity
- News count and relevance
- Time until market resolution
- Relevant economic indicators

### 2.2 Qualitative Layer (Claude LLM)

For each market with an ML signal, Claude analyzes:
- Recent headlines and implications
- Historical context of similar events
- Factors ML cannot capture (political statements, nuance, causal logic)

Output: `llm_probability` (0-1) + `llm_reasoning` (explanatory text).

### 2.3 Final Combination

```
final_probability = (0.6 * ml_probability) + (0.4 * llm_probability)
mispricing = final_probability - polymarket_price
confidence = ml_confidence * agreement_factor(ml, llm)

Trade signal generated when:
  - |mispricing| > 0.10 (10+ points difference)
  - confidence > 0.80
```

### 2.4 Model Training
- Initially trained on Polymarket historical data
- Automatic weekly retraining with new data
- Backtesting before each retraining to validate improvement

---

## Module 3: Trading Engine

### 3.1 Execution Flow

```
Signal from Prediction Engine
  → Risk Manager validates
    → Calculate position size
      → Execute limit order on Polymarket
        → Monitor until resolution or stop-loss
```

- **Limit orders** (not market) to avoid slippage
- Buy YES if `final_probability > polymarket_price` (market underestimates)
- Buy NO if `final_probability < polymarket_price` (market overestimates)
- Continuous monitoring of open positions
- Auto-close if signal reverses or stop-loss is hit

### 3.2 Wallet Integration
- Connection via private key stored in `.env` (never in code)
- Real-time balance tracking
- Alerts if balance drops below configurable threshold

---

## Module 4: Risk Manager (Conservative)

| Parameter | Default Value |
|-----------|--------------|
| Max per trade | 2% of bankroll |
| Max daily | 10% of bankroll |
| Stop-loss per position | -25% of invested amount |
| Minimum confidence | >80% |
| Minimum mispricing | >10 points |
| Max open positions | 5 simultaneous |
| Post-loss cooldown | 1 hour, no new trades |

- All parameters configurable from dashboard
- Detailed log of every decision (trade executed or rejected with reason)
- Manual kill switch: dashboard button to pause all trading instantly

---

## Module 5: Dashboard

Local Dash app at `localhost:8050`. Dark theme. Trading terminal style.

### 5.1 Markets View (Main)
- Table of top markets by volume
- Per market: Polymarket price, NE2 prediction, mispricing, confidence
- Color coding: green (buy opportunity), red (sell opportunity), gray (no signal)
- Click to open detail

### 5.2 Market Detail View
- Historical price chart vs model prediction over time
- Breakdown: ML vs LLM output + reasoning text
- Recent news and sentiment timeline
- Feature importance for that prediction

### 5.3 Portfolio / Trading View
- Open positions with real-time P&L
- Trade history (executed and rejected with reason)
- Metrics: win rate, ROI, Sharpe ratio, max drawdown
- Wallet balance and evolution
- Kill switch (red button to pause trading)

### 5.4 Control View
- "Analyze Now" button for on-demand cycle
- Risk parameter configuration
- Scheduler status (next cycle, last cycle)
- Real-time system logs
- API connection status

---

## Module 6: Scheduler

- APScheduler for automatic cycles every X hours (configurable)
- Manual trigger endpoint via FastAPI (`POST /api/analyze`)
- Each cycle runs: Data Ingestion → Prediction → Signal Filter → Risk Check → Trade

---

## File Structure

```
ne2/
├── app.py                     # Entry point — starts FastAPI + Dash + Scheduler
├── config.py                  # Central config (risk params, API keys, etc.)
├── .env                       # Secrets (API keys, private key) — in .gitignore
├── requirements.txt
│
├── data/
│   ├── ingestion.py           # Ingestion orchestrator
│   ├── polymarket_client.py   # Polymarket CLOB API client
│   ├── news_client.py         # NewsAPI/MediaStack client
│   ├── sentiment_client.py    # Twitter/X + Reddit client
│   ├── economic_client.py     # FRED + CoinGecko client
│   └── models.py              # MarketSnapshot and other dataclasses
│
├── prediction/
│   ├── engine.py              # Prediction Engine orchestrator
│   ├── ml_ensemble.py         # XGBoost + RF + LogReg ensemble
│   ├── llm_analyzer.py        # Claude API for qualitative analysis
│   ├── features.py            # Feature engineering
│   └── combiner.py            # ML + LLM combination → final signal
│
├── trading/
│   ├── engine.py              # Main Trading Engine
│   ├── risk_manager.py        # Risk validation
│   ├── executor.py            # Order execution on Polymarket
│   └── portfolio.py           # Position and P&L tracking
│
├── dashboard/
│   ├── app.py                 # Main Dash app
│   ├── layouts/
│   │   ├── markets.py         # Markets view
│   │   ├── detail.py          # Detail view
│   │   ├── portfolio.py       # Portfolio view
│   │   └── control.py         # Control view
│   └── callbacks/             # Dash callbacks per view
│
├── scheduler/
│   └── jobs.py                # Automatic cycle definitions
│
└── db/
    ├── database.py            # SQLite connection + helpers
    └── migrations.py          # Initial schema and migrations
```

---

## End-to-End Data Flow

```
1. TRIGGER (Scheduler every X hours OR manual button)
   │
2. DATA INGESTION
   │  ├─ Polymarket API → top markets by volume
   │  ├─ Per market:
   │  │   ├─ NewsAPI → relevant news
   │  │   ├─ Twitter/Reddit → sentiment
   │  │   └─ FRED/CoinGecko → economic data
   │  └─ Consolidate → MarketSnapshot per market
   │  └─ Save to SQLite
   │
3. PREDICTION ENGINE (per market)
   │  ├─ Feature engineering → feature vector
   │  ├─ ML Ensemble → ml_probability + ml_confidence
   │  ├─ Claude LLM → llm_probability + llm_reasoning
   │  └─ Combiner → final_probability + confidence + mispricing
   │
4. SIGNAL FILTER
   │  └─ Only signals with |mispricing| > 10% AND confidence > 80% pass
   │
5. RISK MANAGER
   │  ├─ Verify per-trade limit (2% bankroll)
   │  ├─ Verify daily limit (10% bankroll)
   │  ├─ Verify max open positions (<5)
   │  ├─ Verify post-loss cooldown
   │  └─ Reject or approve with position size
   │
6. TRADING ENGINE
   │  ├─ Execute limit order on Polymarket
   │  └─ Record trade in SQLite
   │
7. DASHBOARD (continuous update)
   └─ Reflects everything: markets, signals, trades, portfolio, logs
```

---

## Market Categories

No category filter — the system prioritizes markets by liquidity/volume regardless of topic (politics, crypto, sports, culture, etc.).

---

## API Budget (~$50-100+/month)

| Service | Estimated Cost |
|---------|---------------|
| Claude API | $30-50/month |
| NewsAPI Premium | $20-30/month |
| Twitter/X API | $0-100/month (Basic tier) |
| Reddit API | Free |
| Polymarket API | Free |
| FRED API | Free |
| CoinGecko API | Free |
