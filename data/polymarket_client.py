import httpx
import json as json_module


class PolymarketClient:
    def __init__(self, base_url: str = "https://clob.polymarket.com"):
        self.base_url = base_url
        self.gamma_url = "https://gamma-api.polymarket.com"
        self.client = httpx.Client(timeout=30.0)

    def get_top_markets(self, limit: int = 20) -> list[dict]:
        """Fetch top active markets by 24h volume from Gamma API."""
        response = self.client.get(
            f"{self.gamma_url}/markets",
            params={
                "closed": "false",
                "limit": limit,
                "order": "volume24hr",
                "ascending": "false",
            },
        )
        response.raise_for_status()
        markets = response.json()

        # Normalize to our expected format
        normalized = []
        for m in markets:
            prices = self._parse_prices(m.get("outcomePrices", ""))
            tokens = self._parse_tokens(m.get("clobTokenIds", ""), prices)
            normalized.append({
                "condition_id": m.get("conditionId", m.get("id", "")),
                "question": m.get("question", ""),
                "category": m.get("category", "unknown"),
                "tokens": tokens,
                "volume_num": float(m.get("volume24hr", 0)),
                "active": not m.get("closed", False),
            })
        return normalized

    def get_orderbook(self, token_id: str) -> dict:
        response = self.client.get(
            f"{self.base_url}/book",
            params={"token_id": token_id},
        )
        if response.status_code != 200:
            return {"bids": [], "asks": []}
        return response.json()

    def get_market_price(self, market: dict) -> float:
        tokens = market.get("tokens", [])
        for token in tokens:
            if token.get("outcome") == "Yes":
                return float(token.get("price", 0.0))
        return 0.0

    def parse_orderbook_depth(self, orderbook: dict) -> dict:
        total_bid = sum(float(b.get("size", 0)) for b in orderbook.get("bids", []))
        total_ask = sum(float(a.get("size", 0)) for a in orderbook.get("asks", []))
        total = total_bid + total_ask
        return {
            "total_bid_size": total_bid,
            "total_ask_size": total_ask,
            "bid_ask_imbalance": total_bid / total if total > 0 else 0.5,
        }

    def _parse_prices(self, prices_str) -> list[float]:
        """Parse outcomePrices which can be a JSON string or list."""
        if isinstance(prices_str, list):
            return [float(p) for p in prices_str]
        if isinstance(prices_str, str) and prices_str:
            try:
                parsed = json_module.loads(prices_str)
                return [float(p) for p in parsed]
            except (json_module.JSONDecodeError, ValueError):
                pass
        return [0.0, 0.0]

    def _parse_tokens(self, token_ids_str, prices: list[float]) -> list[dict]:
        """Parse clobTokenIds and pair with prices."""
        token_ids = []
        if isinstance(token_ids_str, list):
            token_ids = token_ids_str
        elif isinstance(token_ids_str, str) and token_ids_str:
            try:
                token_ids = json_module.loads(token_ids_str)
            except (json_module.JSONDecodeError, ValueError):
                pass

        tokens = []
        outcomes = ["Yes", "No"]
        for i, outcome in enumerate(outcomes):
            tokens.append({
                "token_id": token_ids[i] if i < len(token_ids) else "",
                "outcome": outcome,
                "price": prices[i] if i < len(prices) else 0.0,
            })
        return tokens

    def close(self):
        self.client.close()
