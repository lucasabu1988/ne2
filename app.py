import logging
import os

import anthropic
from apscheduler.schedulers.background import BackgroundScheduler

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
    db = Database(settings.db_path)
    db.initialize()
    polymarket = PolymarketClient(base_url=settings.polymarket_api_url)
    news = NewsClient(api_key=settings.newsapi_key)
    sentiment = SentimentClient(
        twitter_bearer_token=settings.twitter_bearer_token,
        reddit_client_id=settings.reddit_client_id,
        reddit_client_secret=settings.reddit_client_secret,
    )
    economic = EconomicClient(fred_api_key=settings.fred_api_key)
    ingestion = DataIngestion(polymarket=polymarket, news=news, sentiment=sentiment, economic=economic, db=db)
    feature_engineer = FeatureEngineer()
    ml_ensemble = MLEnsemble()
    llm_client = anthropic.Anthropic(api_key=settings.anthropic_api_key) if settings.anthropic_api_key else None
    llm_analyzer = LLMAnalyzer(client=llm_client)
    combiner = SignalCombiner()
    prediction_engine = PredictionEngine(
        feature_engineer=feature_engineer, ml_ensemble=ml_ensemble,
        llm_analyzer=llm_analyzer, combiner=combiner, db=db,
    )
    risk_manager = RiskManager(
        db=db, bankroll=1000.0, max_trade_pct=settings.max_trade_pct,
        max_daily_pct=settings.max_daily_pct, stop_loss_pct=settings.stop_loss_pct,
        min_confidence=settings.min_confidence, min_mispricing=settings.min_mispricing,
        max_open_positions=settings.max_open_positions, cooldown_minutes=settings.cooldown_minutes,
    )
    executor = TradeExecutor(db=db, dry_run=True)
    trading_engine = TradingEngine(risk_manager=risk_manager, executor=executor, db=db)
    portfolio = Portfolio(db=db, bankroll=1000.0)
    return {
        "db": db, "ingestion": ingestion, "prediction_engine": prediction_engine,
        "trading_engine": trading_engine, "risk_manager": risk_manager, "portfolio": portfolio,
    }


# Initialize everything at module level so gunicorn can find the Dash server
components = initialize_components()

# Start scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(
    run_analysis_cycle, "interval", hours=settings.cycle_interval_hours,
    kwargs={
        "ingestion": components["ingestion"], "prediction_engine": components["prediction_engine"],
        "trading_engine": components["trading_engine"], "db": components["db"],
        "min_mispricing": settings.min_mispricing, "min_confidence": settings.min_confidence,
    },
)
scheduler.start()
logger.info("Scheduler started")

# Create Dash app — this is what gunicorn serves
dash_app = create_dash_app(
    db=components["db"], portfolio=components["portfolio"],
    risk_manager=components["risk_manager"], ingestion=components["ingestion"],
)

# Expose the Flask server for gunicorn
server = dash_app.server

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    dash_app.run(host="0.0.0.0", port=port, debug=False)
