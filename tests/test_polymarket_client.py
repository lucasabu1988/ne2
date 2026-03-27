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
