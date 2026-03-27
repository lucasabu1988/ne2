import httpx


class EconomicClient:
    def __init__(self, fred_api_key: str = ""):
        self.fred_api_key = fred_api_key
        self.client = httpx.Client(timeout=30.0)

    def get_fred_latest(self, series_id: str) -> float | None:
        if not self.fred_api_key:
            return None
        response = self.client.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": series_id,
                "api_key": self.fred_api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 1,
            },
        )
        if response.status_code != 200:
            return None
        data = response.json()
        observations = data.get("observations", [])
        if not observations:
            return None
        value = observations[0].get("value", ".")
        if value == ".":
            return None
        return float(value)

    def get_crypto_prices(self, coin_ids: list[str] | None = None) -> dict:
        if coin_ids is None:
            coin_ids = ["bitcoin", "ethereum"]
        ids_str = ",".join(coin_ids)
        response = self.client.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={
                "ids": ids_str,
                "vs_currencies": "usd",
                "include_24hr_change": "true",
            },
        )
        if response.status_code != 200:
            return {}
        return response.json()

    def get_all_indicators(self) -> dict:
        indicators = {}
        fed_rate = self.get_fred_latest("FEDFUNDS")
        if fed_rate is not None:
            indicators["fed_funds_rate"] = fed_rate
        vix = self.get_fred_latest("VIXCLS")
        if vix is not None:
            indicators["vix"] = vix
        crypto = self.get_crypto_prices()
        if crypto:
            indicators["crypto"] = crypto
        return indicators

    def close(self):
        self.client.close()
