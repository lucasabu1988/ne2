from dash import html, dcc
import dash_bootstrap_components as dbc

def control_layout():
    return dbc.Container([
        html.H2("System Control", className="text-light mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Risk Parameters", className="bg-dark text-warning"),
                    dbc.CardBody([
                        _param_input("Max per trade (%)", "input-max-trade", 2, 0.5, 10, 0.5),
                        _param_input("Max daily (%)", "input-max-daily", 10, 1, 50, 1),
                        _param_input("Stop-loss (%)", "input-stop-loss", 25, 5, 50, 5),
                        _param_input("Min confidence (%)", "input-min-conf", 80, 50, 99, 5),
                        _param_input("Min mispricing (%)", "input-min-misp", 10, 5, 30, 1),
                        _param_input("Max positions", "input-max-pos", 5, 1, 20, 1),
                        _param_input("Cooldown (min)", "input-cooldown", 60, 0, 240, 15),
                        dbc.Button("Update Risk Params", id="btn-update-risk", color="warning", className="mt-2 w-100"),
                        html.Div(id="risk-update-output"),
                    ]),
                ], className="bg-dark border-secondary"),
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Scheduler", className="bg-dark text-warning"),
                    dbc.CardBody([
                        html.P(id="scheduler-status", className="text-light"),
                        _param_input("Cycle interval (hours)", "input-cycle-hours", 4, 1, 24, 1),
                        dbc.Button("Analyze Now", id="btn-run-cycle", color="success", className="mt-2 w-100"),
                        html.Div(id="cycle-output"),
                    ]),
                ], className="bg-dark border-secondary mb-3"),
                dbc.Card([
                    dbc.CardHeader("API Status", className="bg-dark text-warning"),
                    dbc.CardBody(id="api-status-body"),
                ], className="bg-dark border-secondary"),
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("System Logs", className="bg-dark text-warning"),
                    dbc.CardBody([
                        html.Div(id="system-logs", style={"maxHeight": "500px", "overflowY": "auto", "fontFamily": "monospace", "fontSize": "0.8rem"}),
                    ]),
                ], className="bg-dark border-secondary"),
            ], width=4),
        ]),
        dcc.Interval(id="control-interval", interval=10_000, n_intervals=0),
    ], fluid=True, className="py-3")

def _param_input(label, input_id, default, min_val, max_val, step):
    return dbc.Row([
        dbc.Col(html.Label(label, className="text-light"), width=7),
        dbc.Col(dbc.Input(id=input_id, type="number", value=default, min=min_val, max=max_val, step=step, size="sm"), width=5),
    ], className="mb-2")
