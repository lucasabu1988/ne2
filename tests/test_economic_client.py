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
