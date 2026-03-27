import json
import logging
import re
from data.models import MarketSnapshot

logger = logging.getLogger(__name__)

ANALYSIS_PROMPT = """You are a prediction market analyst. Analyze this market and estimate the true probability of the outcome.

Market Question: {question}
Category: {category}
Current Polymarket Price (implied probability): {price:.1%}
24h Volume: ${volume:,.0f}

Recent News (relevance score: {news_score:.2f}):
{headlines}

Social Sentiment Score: {sentiment:.2f} (-1 bearish to +1 bullish)
Sentiment Velocity: {sentiment_vel:+.2f}

Economic Context:
{economic}

Analyze:
1. What do the news headlines imply about this outcome?
2. Does the sentiment align with or diverge from the current price?
3. Are there factors the market might be underweighting or overweighting?
4. What is your estimated true probability?

Respond in JSON format:
{{"probability": 0.XX, "reasoning": "Your 2-3 sentence analysis"}}
"""

class LLMAnalyzer:
    def __init__(self, client=None, model: str = "claude-sonnet-4-6"):
        self.client = client
        self.model = model

    def analyze(self, snapshot: MarketSnapshot) -> tuple[float, str]:
        try:
            prompt = self._build_prompt(snapshot)
            response = self.client.messages.create(
                model=self.model, max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            return self._parse_response(text)
        except Exception as e:
            logger.error(f"LLM analysis failed for {snapshot.market_id}: {e}")
            return 0.5, f"Analysis failed: {e}"

    def _build_prompt(self, snap: MarketSnapshot) -> str:
        headlines = "\n".join(f"- {h}" for h in snap.latest_headlines) or "- No recent headlines"
        eco_items = []
        for k, v in snap.economic_indicators.items():
            if k == "crypto":
                for coin, data in v.items():
                    if isinstance(data, dict):
                        eco_items.append(f"- {coin}: ${data.get('usd', 0):,.0f}")
            else:
                eco_items.append(f"- {k}: {v}")
        economic = "\n".join(eco_items) or "- No data available"
        return ANALYSIS_PROMPT.format(
            question=snap.question, category=snap.category,
            price=snap.polymarket_price, volume=snap.volume_24h,
            news_score=snap.news_score, headlines=headlines,
            sentiment=snap.sentiment_score, sentiment_vel=snap.sentiment_velocity,
            economic=economic,
        )

    def _parse_response(self, text: str) -> tuple[float, str]:
        try:
            data = json.loads(text)
            prob = float(data["probability"])
            reasoning = data["reasoning"]
            return max(0.0, min(1.0, prob)), reasoning
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                prob = float(data.get("probability", 0.5))
                reasoning = data.get("reasoning", text)
                return max(0.0, min(1.0, prob)), reasoning
            except (json.JSONDecodeError, ValueError):
                pass
        numbers = re.findall(r'0\.\d+', text)
        if numbers:
            prob = float(numbers[0])
            return max(0.0, min(1.0, prob)), text
        return 0.5, text
