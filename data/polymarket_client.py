import httpx

class PolymarketClient:
    def __init__(self, base_url: str = "https://clob.polymarket.com"):
        self.base_url = base_url
        self.client = httpx.Client(base_url=base_url, timeout=30.0)

    def get_top_markets(self, limit: int = 20) -> list[dict]:
        response = self.client.get(
            "/markets",
            params={"active": "true", "limit": limit, "order": "volume_num", "ascending": "false"},
        )
        response.raise_for_status()
        data = response.json()
        return data.get("data", data) if isinstance(data, dict) else data

    def get_orderbook(self, token_id: str) -> dict:
        response = self.client.get("/book", params={"token_id": token_id})
        response.raise_for_status()
        return response.json()

    def get_market_price(self, market: dict) -> float:
        tokens = market.get("tokens", [])
        for token in tokens:
            if token.get("outcome") == "Yes":
                return float(token.get("price", 0.0))
        return 0.0

    def parse_orderbook_depth(self, orderbook: dict) -> dict:
        total_bid = sum(float(b["size"]) for b in orderbook.get("bids", []))
        total_ask = sum(float(a["size"]) for a in orderbook.get("asks", []))
        total = total_bid + total_ask
        return {
            "total_bid_size": total_bid,
            "total_ask_size": total_ask,
            "bid_ask_imbalance": total_bid / total if total > 0 else 0.5,
        }

    def close(self):
        self.client.close()
