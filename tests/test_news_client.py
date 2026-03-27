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
