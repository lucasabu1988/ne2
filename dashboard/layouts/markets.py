from dash import html, dash_table, dcc
import dash_bootstrap_components as dbc


def markets_layout():
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H2("Monitored Markets", className="text-light mb-1"),
                html.Small(id="last-updated", className="text-muted"),
            ]),
            dbc.Col(
                dbc.Button("Analyze Now", id="btn-analyze", color="primary", className="float-end"),
                width="auto",
            ),
        ], className="mb-3"),

        # Summary cards
        dbc.Row(id="market-summary-cards", className="mb-3"),

        # Status message
        dbc.Row([
            dbc.Col(html.Div(id="analyze-status")),
        ], className="mb-2"),

        # Markets table
        dbc.Row([
            dbc.Col([
                dash_table.DataTable(
                    id="markets-table",
                    columns=[
                        {"name": "Market", "id": "question"},
                        {"name": "Category", "id": "category"},
                        {"name": "PM Price", "id": "polymarket_price", "type": "numeric",
                         "format": {"specifier": ".1%"}},
                        {"name": "NE2 Pred", "id": "prediction", "type": "numeric",
                         "format": {"specifier": ".1%"}},
                        {"name": "Mispricing", "id": "mispricing", "type": "numeric",
                         "format": {"specifier": "+.1%"}},
                        {"name": "Confidence", "id": "confidence", "type": "numeric",
                         "format": {"specifier": ".0%"}},
                        {"name": "Volume 24h", "id": "volume_24h", "type": "numeric",
                         "format": {"specifier": "$,.0f"}},
                        {"name": "Signal", "id": "signal"},
                    ],
                    style_table={"overflowX": "auto"},
                    style_header={
                        "backgroundColor": "#1a1a2e", "color": "#e0e0e0",
                        "fontWeight": "bold", "textAlign": "center",
                    },
                    style_cell={
                        "backgroundColor": "#16213e", "color": "#e0e0e0",
                        "border": "1px solid #0f3460", "textAlign": "center",
                        "padding": "8px 12px", "fontSize": "0.9rem",
                    },
                    style_cell_conditional=[
                        {"if": {"column_id": "question"}, "textAlign": "left", "minWidth": "250px"},
                        {"if": {"column_id": "category"}, "minWidth": "80px"},
                    ],
                    style_data_conditional=[
                        {"if": {"filter_query": "{signal} = BUY_YES"},
                         "backgroundColor": "#0a3d0a", "color": "#4caf50", "fontWeight": "bold"},
                        {"if": {"filter_query": "{signal} = BUY_NO"},
                         "backgroundColor": "#3d0a0a", "color": "#f44336", "fontWeight": "bold"},
                        # Positive mispricing = green text
                        {"if": {"filter_query": "{mispricing} > 0", "column_id": "mispricing"},
                         "color": "#4caf50"},
                        # Negative mispricing = red text
                        {"if": {"filter_query": "{mispricing} < 0", "column_id": "mispricing"},
                         "color": "#f44336"},
                        # High confidence = brighter
                        {"if": {"filter_query": "{confidence} > 0.5", "column_id": "confidence"},
                         "color": "#ffc107"},
                    ],
                    sort_action="native",
                    sort_by=[{"column_id": "volume_24h", "direction": "desc"}],
                    page_size=20,
                    row_selectable="single",
                ),
            ]),
        ]),

        # Market detail panel (shows when a row is selected)
        dbc.Row([
            dbc.Col(html.Div(id="market-detail-panel"), className="mt-3"),
        ]),

        dcc.Interval(id="markets-interval", interval=120_000, n_intervals=0),
        dcc.Store(id="selected-market-store"),
    ], fluid=True, className="py-3")
