import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
from dashboard.layouts.markets import markets_layout
from dashboard.callbacks.markets_cb import register_markets_callbacks
from dashboard.callbacks.portfolio_cb import register_portfolio_callbacks
from dashboard.callbacks.control_cb import register_control_callbacks

def create_dash_app(db, portfolio=None, risk_manager=None) -> dash.Dash:
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], suppress_callback_exceptions=True)
    app.layout = html.Div([
        dbc.Navbar(
            dbc.Container([
                dbc.NavbarBrand("NE2 — Polymarket Predictor", className="ms-2 text-warning fw-bold"),
                dbc.Nav([
                    dbc.NavLink("Markets", href="/", active="exact"),
                    dbc.NavLink("Portfolio", href="/portfolio"),
                    dbc.NavLink("Control", href="/control"),
                ], navbar=True),
            ], fluid=True),
            color="dark", dark=True,
        ),
        dcc.Location(id="url", refresh=False),
        html.Div(id="page-content"),
    ], style={"backgroundColor": "#0a0a1a", "minHeight": "100vh"})

    @app.callback(dash.Output("page-content", "children"), dash.Input("url", "pathname"))
    def display_page(pathname):
        if pathname == "/portfolio":
            from dashboard.layouts.portfolio import portfolio_layout
            return portfolio_layout()
        elif pathname == "/control":
            from dashboard.layouts.control import control_layout
            return control_layout()
        else:
            return markets_layout()

    register_markets_callbacks(app, db)
    if portfolio and risk_manager:
        register_portfolio_callbacks(app, db, portfolio, risk_manager)
        register_control_callbacks(app, db, risk_manager)
    return app
