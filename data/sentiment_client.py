import httpx

POSITIVE_WORDS = {
    "bullish", "surge", "moon", "gain", "rise", "up", "great", "amazing",
    "wonderful", "love", "excellent", "strong", "inevitable", "positive",
    "optimistic", "win", "success", "profit", "buy", "breakout", "rally",
}
NEGATIVE_WORDS = {
    "bearish", "crash", "dump", "fall", "down", "terrible", "awful", "hate",
    "worst", "weak", "sell", "loss", "fear", "negative", "pessimistic",
    "fail", "decline", "drop", "collapse", "risk", "bubble",
}

class SentimentClient:
    def __init__(self, twitter_bearer_token: str = "", reddit_client_id: str = "", reddit_client_secret: str = ""):
        self.twitter_token = twitter_bearer_token
        self.reddit_client_id = reddit_client_id
        self.reddit_client_secret = reddit_client_secret

    def fetch_twitter(self, query: str, max_results: int = 50) -> list[str]:
        if not self.twitter_token:
            return []
        client = httpx.Client(timeout=30.0)
        response = client.get(
            "https://api.twitter.com/2/tweets/search/recent",
            params={"query": query, "max_results": min(max_results, 100)},
            headers={"Authorization": f"Bearer {self.twitter_token}"},
        )
        client.close()
        if response.status_code != 200:
            return []
        data = response.json()
        return [tweet["text"] for tweet in data.get("data", [])]

    def fetch_reddit(self, query: str, subreddit: str = "all", limit: int = 25) -> list[str]:
        if not self.reddit_client_id:
            return []
        client = httpx.Client(timeout=30.0)
        response = client.get(
            f"https://www.reddit.com/r/{subreddit}/search.json",
            params={"q": query, "limit": limit, "sort": "relevance", "t": "week"},
            headers={"User-Agent": "NE2-Bot/1.0"},
        )
        client.close()
        if response.status_code != 200:
            return []
        data = response.json()
        posts = data.get("data", {}).get("children", [])
        return [p["data"]["title"] + " " + p["data"].get("selftext", "") for p in posts]

    def analyze_sentiment(self, texts: list[str]) -> float:
        if not texts:
            return 0.0
        scores = []
        for text in texts:
            words = set(text.lower().split())
            pos = len(words & POSITIVE_WORDS)
            neg = len(words & NEGATIVE_WORDS)
            total = pos + neg
            if total == 0:
                scores.append(0.0)
            else:
                scores.append((pos - neg) / total)
        return max(-1.0, min(1.0, sum(scores) / len(scores)))

    def get_sentiment_for_query(self, query: str) -> tuple[float, int]:
        twitter_texts = self.fetch_twitter(query)
        reddit_texts = self.fetch_reddit(query)
        all_texts = twitter_texts + reddit_texts
        score = self.analyze_sentiment(all_texts)
        return score, len(all_texts)
