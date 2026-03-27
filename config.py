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
