from dash import html, dash_table, dcc
import dash_bootstrap_components as dbc

def portfolio_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col(html.H2("Portfolio & Trading", className="text-light mb-3")),
            dbc.Col(dbc.Button("KILL SWITCH", id="btn-kill-switch", color="danger", className="float-end", size="lg"), width="auto"),
        ], className="mb-3"),
        dbc.Row(id="portfolio-metrics", className="mb-4"),
        dbc.Row([dbc.Col(html.H4("Open Positions", className="text-light mb-2"))]),
        dbc.Row([
            dbc.Col(dash_table.DataTable(
                id="positions-table",
                columns=[
                    {"name": "Trade ID", "id": "trade_id"},
                    {"name": "Market", "id": "market_id"},
                    {"name": "Action", "id": "action"},
                    {"name": "Amount", "id": "amount", "type": "numeric", "format": {"specifier": "$,.2f"}},
                    {"name": "Entry Price", "id": "price", "type": "numeric", "format": {"specifier": ".3f"}},
                    {"name": "Confidence", "id": "confidence", "type": "numeric", "format": {"specifier": ".0%"}},
                    {"name": "Time", "id": "timestamp"},
                ],
                style_header={"backgroundColor": "#1a1a2e", "color": "#e0e0e0"},
                style_cell={"backgroundColor": "#16213e", "color": "#e0e0e0", "border": "1px solid #0f3460"},
                page_size=10,
            )),
        ], className="mb-4"),
        dbc.Row([dbc.Col(html.H4("Trade History", className="text-light mb-2"))]),
        dbc.Row([
            dbc.Col(dash_table.DataTable(
                id="history-table",
                columns=[
                    {"name": "Trade ID", "id": "trade_id"},
                    {"name": "Market", "id": "market_id"},
                    {"name": "Action", "id": "action"},
                    {"name": "Amount", "id": "amount", "type": "numeric", "format": {"specifier": "$,.2f"}},
                    {"name": "Entry", "id": "price", "type": "numeric", "format": {"specifier": ".3f"}},
                    {"name": "Exit", "id": "close_price", "type": "numeric", "format": {"specifier": ".3f"}},
                    {"name": "Status", "id": "status"},
                    {"name": "Reason", "id": "rejection_reason"},
                ],
                style_header={"backgroundColor": "#1a1a2e", "color": "#e0e0e0"},
                style_cell={"backgroundColor": "#16213e", "color": "#e0e0e0", "border": "1px solid #0f3460"},
                page_size=20,
                sort_action="native",
            )),
        ]),
        dcc.Interval(id="portfolio-interval", interval=30_000, n_intervals=0),
        html.Div(id="kill-switch-output", className="mt-2"),
    ], fluid=True, className="py-3")
