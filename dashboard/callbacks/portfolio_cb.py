import dash_bootstrap_components as dbc
from dash import Input, Output, html

def register_portfolio_callbacks(app, db, portfolio, risk_manager):
    @app.callback(Output("portfolio-metrics", "children"), Input("portfolio-interval", "n_intervals"))
    def update_metrics(n):
        metrics = portfolio.compute_metrics()
        balance = portfolio.available_balance()
        cards = [
            _metric_card("Balance", f"${balance:,.2f}", "primary"),
            _metric_card("Win Rate", f"{metrics['win_rate']:.0%}", "success" if metrics["win_rate"] > 0.5 else "warning"),
            _metric_card("Total P&L", f"${metrics['total_pnl']:,.2f}", "success" if metrics["total_pnl"] >= 0 else "danger"),
            _metric_card("Trades", str(metrics["total_trades"]), "info"),
            _metric_card("Max Drawdown", f"${metrics['max_drawdown']:,.2f}", "warning"),
        ]
        return [dbc.Col(c, width=2) for c in cards]

    @app.callback(Output("positions-table", "data"), Input("portfolio-interval", "n_intervals"))
    def update_positions(n):
        positions = portfolio.get_open_positions()
        return [{"trade_id": t.trade_id, "market_id": t.market_id[:16], "action": t.action.value,
                 "amount": t.amount, "price": t.price, "confidence": t.confidence,
                 "timestamp": t.timestamp.strftime("%m/%d %H:%M")} for t in positions]

    @app.callback(Output("history-table", "data"), Input("portfolio-interval", "n_intervals"))
    def update_history(n):
        trades = db.get_all_trades(limit=50)
        return [{"trade_id": t.trade_id, "market_id": t.market_id[:16], "action": t.action.value,
                 "amount": t.amount, "price": t.price, "close_price": t.close_price,
                 "status": t.status.value, "rejection_reason": t.rejection_reason or ""} for t in trades]

    @app.callback(Output("kill-switch-output", "children"), Input("btn-kill-switch", "n_clicks"), prevent_initial_call=True)
    def toggle_kill_switch(n_clicks):
        risk_manager.kill_switch = not risk_manager.kill_switch
        state = "ACTIVE" if risk_manager.kill_switch else "INACTIVE"
        color = "danger" if risk_manager.kill_switch else "success"
        return dbc.Alert(f"Kill switch: {state}", color=color, duration=5000)

def _metric_card(title, value, color):
    return dbc.Card(dbc.CardBody([
        html.P(title, className="text-muted mb-1", style={"fontSize": "0.8rem"}),
        html.H4(value, className=f"text-{color} mb-0"),
    ]), className="bg-dark border-secondary")
