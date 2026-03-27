from data.models import TradeAction, TradeRecord, TradeStatus

class Portfolio:
    def __init__(self, db, bankroll: float = 1000.0):
        self.db = db
        self.bankroll = bankroll

    def get_open_positions(self) -> list[TradeRecord]:
        return self.db.get_open_trades()

    def total_invested(self) -> float:
        return sum(t.amount for t in self.get_open_positions())

    def available_balance(self) -> float:
        return self.bankroll - self.total_invested()

    def compute_metrics(self) -> dict:
        all_trades = self.db.get_all_trades(limit=1000)
        closed = [t for t in all_trades if t.status in (TradeStatus.CLOSED, TradeStatus.STOPPED)]
        if not closed:
            return {"total_trades": len(all_trades), "closed_trades": 0, "win_rate": 0.0, "total_pnl": 0.0, "avg_pnl": 0.0, "max_drawdown": 0.0}
        pnls = []
        for trade in closed:
            if trade.close_price is None:
                continue
            pnl = trade.unrealized_pnl(trade.close_price)
            pnls.append(pnl)
        wins = sum(1 for p in pnls if p > 0)
        total_pnl = sum(pnls)
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in pnls:
            cumulative += p
            peak = max(peak, cumulative)
            dd = peak - cumulative
            max_dd = max(max_dd, dd)
        return {
            "total_trades": len(all_trades), "closed_trades": len(closed),
            "win_rate": wins / len(pnls) if pnls else 0.0,
            "total_pnl": total_pnl, "avg_pnl": total_pnl / len(pnls) if pnls else 0.0,
            "max_drawdown": max_dd,
        }
