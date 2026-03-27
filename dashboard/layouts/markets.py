from dash import html, dash_table, dcc
import dash_bootstrap_components as dbc

def markets_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col(html.H2("Monitored Markets", className="text-light mb-3")),
            dbc.Col(dbc.Button("Analyze Now", id="btn-analyze", color="primary", className="float-end"), width="auto"),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([
                dash_table.DataTable(
                    id="markets-table",
                    columns=[
                        {"name": "Market", "id": "question"},
                        {"name": "Category", "id": "category"},
                        {"name": "PM Price", "id": "polymarket_price", "type": "numeric", "format": {"specifier": ".1%"}},
                        {"name": "NE2 Pred", "id": "prediction", "type": "numeric", "format": {"specifier": ".1%"}},
                        {"name": "Mispricing", "id": "mispricing", "type": "numeric", "format": {"specifier": "+.1%"}},
                        {"name": "Confidence", "id": "confidence", "type": "numeric", "format": {"specifier": ".0%"}},
                        {"name": "Volume 24h", "id": "volume_24h", "type": "numeric", "format": {"specifier": "$,.0f"}},
                        {"name": "Signal", "id": "signal"},
                    ],
                    style_table={"overflowX": "auto"},
                    style_header={"backgroundColor": "#1a1a2e", "color": "#e0e0e0", "fontWeight": "bold"},
                    style_cell={"backgroundColor": "#16213e", "color": "#e0e0e0", "border": "1px solid #0f3460"},
                    style_data_conditional=[
                        {"if": {"filter_query": "{signal} = BUY_YES"}, "backgroundColor": "#0a3d0a", "color": "#4caf50"},
                        {"if": {"filter_query": "{signal} = BUY_NO"}, "backgroundColor": "#3d0a0a", "color": "#f44336"},
                    ],
                    row_selectable="single",
                    page_size=20,
                    sort_action="native",
                ),
            ]),
        ]),
        dcc.Interval(id="markets-interval", interval=120_000, n_intervals=0),
        dcc.Store(id="selected-market-store"),
    ], fluid=True, className="py-3")
