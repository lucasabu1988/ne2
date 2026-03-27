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
        token_id = self._get_yes_token_id(market)
        order_book_depth = {}
        if token_id:
            try:
                raw_book = self.polymarket.get_orderbook(token_id)
                order_book_depth = self.polymarket.parse_orderbook_depth(raw_book)
            except Exception as e:
                logger.warning(f"Failed to fetch orderbook for {market_id}: {e}")
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
        sentiment_score = 0.0
        try:
            sentiment_score, _ = self.sentiment.get_sentiment_for_query(question)
        except Exception as e:
            logger.warning(f"Failed to fetch sentiment for {market_id}: {e}")
        sentiment_velocity = self._calc_sentiment_velocity(market_id, sentiment_score)
        return MarketSnapshot(
            market_id=market_id, question=question, category=category,
            polymarket_price=price, volume_24h=market.get("volume_num", 0.0),
            order_book_depth=order_book_depth, news_score=news_score,
            news_count=news_count, latest_headlines=headlines,
            sentiment_score=sentiment_score, sentiment_velocity=sentiment_velocity,
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
