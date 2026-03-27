"""News client using Google News RSS (free, no API key required)."""

import logging
import re
import xml.etree.ElementTree as ET
from html import unescape
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"


class NewsClient:
    def __init__(self, api_key: str = "", **kwargs):
        self.client = httpx.Client(timeout=30.0, follow_redirects=True)

    def search(self, query: str, max_results: int = 10) -> list[dict]:
        """Search Google News RSS for articles matching query."""
        try:
            # Clean query for RSS search
            clean_query = re.sub(r'[^\w\s]', ' ', query).strip()
            # Take first 5-6 meaningful words to avoid overly specific queries
            words = [w for w in clean_query.split() if len(w) > 2][:6]
            search_query = " ".join(words)

            response = self.client.get(
                GOOGLE_NEWS_RSS,
                params={"q": search_query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
            )
            if response.status_code != 200:
                return []

            return self._parse_rss(response.text, max_results)
        except Exception as e:
            logger.warning(f"Google News search failed for '{query[:50]}': {e}")
            return []

    def _parse_rss(self, xml_text: str, max_results: int) -> list[dict]:
        """Parse Google News RSS XML into article dicts."""
        articles = []
        try:
            root = ET.fromstring(xml_text)
            for item in root.findall(".//item"):
                if len(articles) >= max_results:
                    break
                title = item.findtext("title", "")
                # Google News description contains HTML — extract text
                desc_raw = item.findtext("description", "")
                description = self._clean_html(desc_raw)
                pub_date = item.findtext("pubDate", "")
                source = ""
                # Title format is "Headline - Source"
                if " - " in title:
                    parts = title.rsplit(" - ", 1)
                    title = parts[0]
                    source = parts[1] if len(parts) > 1 else ""

                articles.append({
                    "title": title,
                    "description": description,
                    "publishedAt": pub_date,
                    "source": {"name": source},
                })
        except ET.ParseError as e:
            logger.warning(f"Failed to parse RSS: {e}")
        return articles

    def _clean_html(self, html_text: str) -> str:
        """Strip HTML tags and decode entities."""
        text = re.sub(r'<[^>]+>', ' ', html_text)
        text = unescape(text)
        return re.sub(r'\s+', ' ', text).strip()

    def compute_news_score(self, articles: list[dict], market_question: str) -> float:
        """Score news relevance and volume for a market question."""
        if not articles:
            return 0.0

        question_words = set(market_question.lower().split())
        # Remove common words
        stop_words = {"will", "the", "a", "an", "in", "on", "by", "be", "of", "to", "and", "or", "is", "it"}
        question_words -= stop_words

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
