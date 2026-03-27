import pytest
from data.polymarket_client import PolymarketClient

SAMPLE_GAMMA_RESPONSE = [
    {
        "conditionId": "0xabc123",
        "question": "Will Bitcoin reach $100k by June 2026?",
        "category": "crypto",
        "outcomePrices": '["0.65", "0.35"]',
        "clobTokenIds": '["yes_token_1", "no_token_1"]',
        "volume24hr": 250000.0,
        "closed": False,
    },
    {
        "conditionId": "0xdef456",
        "question": "Will ETH flip BTC by 2027?",
        "category": "crypto",
        "outcomePrices": '["0.10", "0.90"]',
        "clobTokenIds": '["yes_token_2", "no_token_2"]',
        "volume24hr": 50000.0,
        "closed": False,
    },
]

SAMPLE_ORDERBOOK_RESPONSE = {
    "bids": [{"price": "0.64", "size": "1000"}, {"price": "0.63", "size": "2000"}],
    "asks": [{"price": "0.66", "size": "800"}, {"price": "0.67", "size": "1500"}],
}


class TestPolymarketClient:
    def test_get_top_markets(self, httpx_mock):
        httpx_mock.add_response(
            url="https://gamma-api.polymarket.com/markets?closed=false&limit=20&order=volume24hr&ascending=false",
            json=SAMPLE_GAMMA_RESPONSE,
        )
        client = PolymarketClient()
        markets = client.get_top_markets(limit=20)
        assert len(markets) == 2
        assert markets[0]["condition_id"] == "0xabc123"
        assert markets[0]["volume_num"] == 250000.0
        assert markets[0]["tokens"][0]["outcome"] == "Yes"
        assert markets[0]["tokens"][0]["price"] == 0.65

    def test_get_market_price(self):
        client = PolymarketClient()
        market = {
            "tokens": [
                {"outcome": "Yes", "price": 0.65},
                {"outcome": "No", "price": 0.35},
            ]
        }
        assert client.get_market_price(market) == 0.65

    def test_parse_orderbook_depth(self):
        client = PolymarketClient()
        depth = client.parse_orderbook_depth(SAMPLE_ORDERBOOK_RESPONSE)
        assert depth["total_bid_size"] == 3000.0
        assert depth["total_ask_size"] == 2300.0
        assert depth["bid_ask_imbalance"] == pytest.approx(3000.0 / 5300.0, rel=1e-3)
