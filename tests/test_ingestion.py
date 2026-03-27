from unittest.mock import MagicMock
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
        mock_db.get_latest_snapshot.return_value = None

        ingestion = DataIngestion(
            polymarket=mock_poly, news=mock_news, sentiment=mock_sentiment,
            economic=mock_economic, db=mock_db,
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
            polymarket=mock_poly, news=MagicMock(), sentiment=MagicMock(),
            economic=MagicMock(), db=MagicMock(),
        )
        snapshots = ingestion.run()
        assert snapshots == []
