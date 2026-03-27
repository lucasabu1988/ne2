from dash import Input, Output, State, html
import dash_bootstrap_components as dbc

def register_control_callbacks(app, db, risk_manager):
    @app.callback(Output("system-logs", "children"), Input("control-interval", "n_intervals"))
    def update_logs(n):
        logs = db.get_recent_logs(limit=50)
        return [html.Div(
            f"[{log['timestamp'][:19]}] [{log['level']}] {log['module']}: {log['message']}",
            className=f"text-{'danger' if log['level'] == 'ERROR' else 'warning' if log['level'] == 'WARNING' else 'light'}",
        ) for log in logs]

    @app.callback(
        Output("risk-update-output", "children"),
        Input("btn-update-risk", "n_clicks"),
        State("input-max-trade", "value"), State("input-max-daily", "value"),
        State("input-stop-loss", "value"), State("input-min-conf", "value"),
        State("input-min-misp", "value"), State("input-max-pos", "value"),
        State("input-cooldown", "value"), prevent_initial_call=True,
    )
    def update_risk_params(n, max_trade, max_daily, stop_loss, min_conf, min_misp, max_pos, cooldown):
        risk_manager.max_trade_pct = (max_trade or 2) / 100
        risk_manager.max_daily_pct = (max_daily or 10) / 100
        risk_manager.stop_loss_pct = (stop_loss or 25) / 100
        risk_manager.min_confidence = (min_conf or 80) / 100
        risk_manager.min_mispricing = (min_misp or 10) / 100
        risk_manager.max_open_positions = int(max_pos or 5)
        risk_manager.cooldown_minutes = int(cooldown or 60)
        db.save_log("INFO", "control", "Risk parameters updated")
        return dbc.Alert("Risk parameters updated", color="success", duration=3000)

    @app.callback(Output("api-status-body", "children"), Input("control-interval", "n_intervals"))
    def update_api_status(n):
        apis = [("Polymarket", "success"), ("NewsAPI", "success"), ("Twitter/X", "success"),
                ("Reddit", "success"), ("Claude API", "success"), ("FRED", "success"), ("CoinGecko", "success")]
        return [html.Div([html.Span("● ", className=f"text-{color}"), html.Span(name, className="text-light")], className="mb-1")
                for name, color in apis]
