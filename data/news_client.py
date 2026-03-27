import httpx
from datetime import datetime, timedelta, timezone

class NewsClient:
    def __init__(self, api_key: str, base_url: str = "https://newsapi.org/v2"):
        self.api_key = api_key
        self.base_url = base_url
        self.client = httpx.Client(timeout=30.0)

    def search(self, query: str, days_back: int = 3, max_results: int = 20) -> list[dict]:
        from_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
        response = self.client.get(
            f"{self.base_url}/everything",
            params={"q": query, "from": from_date, "sortBy": "relevancy", "pageSize": max_results, "apiKey": self.api_key},
        )
        response.raise_for_status()
        data = response.json()
        return data.get("articles", [])

    def compute_news_score(self, articles: list[dict], market_question: str) -> float:
        if not articles:
            return 0.0
        question_words = set(market_question.lower().split())
        relevance_scores = []
        for article in articles:
            title = (article.get("title") or "").lower()
            desc = (article.get("description") or "").lower()
            text = title + " " + desc
            text_words = set(text.split())
            overlap = len(question_words & text_words)
            relevance = min(overlap / max(len(question_words), 1), 1.0)
            relevance_scores.append(relevance)
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        volume_factor = min(len(articles) / 10.0, 1.0)
        return (avg_relevance * 0.6) + (volume_factor * 0.4)

    def close(self):
        self.client.close()
