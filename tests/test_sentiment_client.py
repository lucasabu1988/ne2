from data.sentiment_client import SentimentClient

class TestSentimentClient:
    def test_analyze_texts_positive(self):
        client = SentimentClient(twitter_bearer_token="test", reddit_client_id="test", reddit_client_secret="test")
        texts = ["This is amazing and wonderful!", "I love this, great news!"]
        score = client.analyze_sentiment(texts)
        assert score > 0.0

    def test_analyze_texts_negative(self):
        client = SentimentClient(twitter_bearer_token="test", reddit_client_id="test", reddit_client_secret="test")
        texts = ["This is terrible and awful!", "I hate this, worst news ever!"]
        score = client.analyze_sentiment(texts)
        assert score < 0.0

    def test_analyze_texts_empty(self):
        client = SentimentClient(twitter_bearer_token="test", reddit_client_id="test", reddit_client_secret="test")
        score = client.analyze_sentiment([])
        assert score == 0.0

    def test_sentiment_score_range(self):
        client = SentimentClient(twitter_bearer_token="test", reddit_client_id="test", reddit_client_secret="test")
        texts = ["Mixed feelings about this situation"]
        score = client.analyze_sentiment(texts)
        assert -1.0 <= score <= 1.0
