from data.news_client import NewsClient

SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
<item>
  <title>Bitcoin Surges Past $95,000 - Reuters</title>
  <description>&lt;a href="http://example.com"&gt;Major banks increase BTC exposure&lt;/a&gt;</description>
  <pubDate>Wed, 26 Mar 2026 10:00:00 GMT</pubDate>
</item>
<item>
  <title>Crypto Market Analysis: Bull Run Continues - Bloomberg</title>
  <description>Analysts predict further gains</description>
  <pubDate>Wed, 26 Mar 2026 08:00:00 GMT</pubDate>
</item>
</channel></rss>"""


class TestNewsClient:
    def test_search_news(self, httpx_mock):
        httpx_mock.add_response(text=SAMPLE_RSS, headers={"content-type": "application/xml"})
        client = NewsClient()
        articles = client.search("Bitcoin 100k")
        assert len(articles) == 2
        assert "Bitcoin" in articles[0]["title"]
        assert articles[0]["source"]["name"] == "Reuters"

    def test_compute_news_score(self):
        client = NewsClient()
        articles = [
            {"title": "Bitcoin surges past 95000", "description": "BTC exposure increases"},
            {"title": "Crypto bull run continues", "description": "Analysts predict gains"},
        ]
        score = client.compute_news_score(articles, "Bitcoin reach 100k")
        assert 0.0 <= score <= 1.0

    def test_empty_results(self, httpx_mock):
        httpx_mock.add_response(text='<?xml version="1.0"?><rss><channel></channel></rss>')
        client = NewsClient()
        articles = client.search("extremely obscure topic xyz")
        assert len(articles) == 0

    def test_html_cleaning(self):
        client = NewsClient()
        result = client._clean_html('<a href="url">Hello</a> &amp; <b>world</b>')
        assert result == "Hello & world"
