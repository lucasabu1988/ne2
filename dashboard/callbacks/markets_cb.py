import logging
from datetime import datetime, timezone

import dash_bootstrap_components as dbc
from dash import Input, Output, State, ctx, html, no_update

logger = logging.getLogger(__name__)


def register_markets_callbacks(app, db, ingestion=None, prediction_engine=None):

    @app.callback(
        Output("markets-table", "data"),
        Output("analyze-status", "children"),
        Output("last-updated", "children"),
        Output("market-summary-cards", "children"),
        Input("markets-interval", "n_intervals"),
        Input("btn-analyze", "n_clicks"),
    )
    def update_markets_table(n, n_clicks):
        triggered = ctx.triggered_id
        status_msg = None
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # If button clicked, run full pipeline
        if triggered == "btn-analyze" and n_clicks:
            snapshots = []
            if ingestion:
                try:
                    snapshots = ingestion.run(top_n=20)
                    logger.info(f"Ingested {len(snapshots)} markets")
                except Exception as e:
                    logger.error(f"Ingestion failed: {e}")
                    status_msg = dbc.Alert(f"Ingestion error: {e}", color="danger", duration=10000)

            if snapshots and prediction_engine:
                try:
                    predictions = prediction_engine.predict_batch(snapshots)
                    signals = sum(1 for p in predictions if p.has_signal(0.10, 0.50))
                    status_msg = dbc.Alert(
                        f"Analysis complete: {len(snapshots)} markets, {len(predictions)} predictions, {signals} signals",
                        color="success", duration=8000,
                    )
                except Exception as e:
                    logger.error(f"Prediction failed: {e}")
                    status_msg = dbc.Alert(f"Prediction error: {e}", color="warning", duration=10000)

        # Load from DB
        rows = []
        try:
            market_rows = db.conn.execute(
                "SELECT DISTINCT market_id FROM snapshots ORDER BY timestamp DESC LIMIT 20"
            ).fetchall()
            for mr in market_rows:
                mid = mr["market_id"]
                snap = db.get_latest_snapshot(mid)
                pred = db.get_latest_prediction(mid)
                if not snap:
                    continue
                rows.append({
                    "market_id": mid,
                    "question": snap.question[:80],
                    "category": snap.category,
                    "polymarket_price": snap.polymarket_price,
                    "prediction": pred.final_probability if pred else None,
                    "mispricing": pred.mispricing if pred else None,
                    "confidence": pred.confidence if pred else None,
                    "volume_24h": snap.volume_24h,
                    "signal": _signal_label(pred) if pred else "-",
                })
        except Exception:
            pass

        # Summary cards
        summary = _build_summary(rows)

        if not rows and not status_msg:
            status_msg = dbc.Alert(
                "No markets loaded yet. Click 'Analyze Now' to fetch live data from Polymarket.",
                color="info",
            )

        return rows, status_msg, f"Last updated: {now}" if rows else "", summary

    @app.callback(
        Output("market-detail-panel", "children"),
        Input("markets-table", "selected_rows"),
        State("markets-table", "data"),
    )
    def show_market_detail(selected_rows, data):
        if not selected_rows or not data:
            return None
        row = data[selected_rows[0]]
        mid = row.get("market_id", "")

        # Get prediction reasoning from DB
        pred = db.get_latest_prediction(mid)
        reasoning = pred.llm_reasoning if pred else "No LLM analysis available"
        snap = db.get_latest_snapshot(mid)

        return dbc.Card([
            dbc.CardHeader(
                html.H5(row["question"], className="text-warning mb-0"),
                className="bg-dark",
            ),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("Polymarket Price", className="text-muted"),
                        html.H3(f"{row['polymarket_price']:.1%}", className="text-light"),
                    ], width=2),
                    dbc.Col([
                        html.H6("NE2 Prediction", className="text-muted"),
                        html.H3(
                            f"{row['prediction']:.1%}" if row.get("prediction") else "—",
                            className="text-info",
                        ),
                    ], width=2),
                    dbc.Col([
                        html.H6("Mispricing", className="text-muted"),
                        html.H3(
                            f"{row['mispricing']:+.1%}" if row.get("mispricing") else "—",
                            className="text-success" if (row.get("mispricing") or 0) > 0 else "text-danger",
                        ),
                    ], width=2),
                    dbc.Col([
                        html.H6("Confidence", className="text-muted"),
                        html.H3(
                            f"{row['confidence']:.0%}" if row.get("confidence") else "—",
                            className="text-warning",
                        ),
                    ], width=2),
                    dbc.Col([
                        html.H6("Volume 24h", className="text-muted"),
                        html.H3(f"${row['volume_24h']:,.0f}", className="text-light"),
                    ], width=2),
                    dbc.Col([
                        html.H6("Signal", className="text-muted"),
                        html.H3(
                            row.get("signal", "-"),
                            className="text-success" if row.get("signal") == "BUY_YES"
                            else "text-danger" if row.get("signal") == "BUY_NO"
                            else "text-secondary",
                        ),
                    ], width=2),
                ], className="mb-3"),
                html.Hr(className="border-secondary"),
                html.H6("LLM Analysis (Claude)", className="text-muted mb-2"),
                html.P(reasoning, className="text-light", style={"whiteSpace": "pre-wrap"}),
                html.Hr(className="border-secondary"),
                dbc.Row([
                    dbc.Col([
                        html.Small("News Score", className="text-muted d-block"),
                        html.Span(f"{snap.news_score:.2f}" if snap else "—", className="text-light"),
                    ], width=2),
                    dbc.Col([
                        html.Small("Sentiment", className="text-muted d-block"),
                        html.Span(f"{snap.sentiment_score:+.2f}" if snap else "—", className="text-light"),
                    ], width=2),
                    dbc.Col([
                        html.Small("News Count", className="text-muted d-block"),
                        html.Span(f"{snap.news_count}" if snap else "—", className="text-light"),
                    ], width=2),
                    dbc.Col([
                        html.Small("Category", className="text-muted d-block"),
                        html.Span(row.get("category", "—"), className="text-light"),
                    ], width=2),
                    dbc.Col([
                        html.Small("ML Probability", className="text-muted d-block"),
                        html.Span(f"{pred.ml_probability:.1%}" if pred else "—", className="text-light"),
                    ], width=2),
                    dbc.Col([
                        html.Small("LLM Probability", className="text-muted d-block"),
                        html.Span(f"{pred.llm_probability:.1%}" if pred else "—", className="text-light"),
                    ], width=2),
                ]),
            ]),
        ], className="bg-dark border-secondary")

    @app.callback(
        Output("selected-market-store", "data"),
        Input("markets-table", "selected_rows"),
        Input("markets-table", "data"),
    )
    def store_selected_market(selected_rows, data):
        if not selected_rows or not data:
            return no_update
        return data[selected_rows[0]].get("market_id")


def _signal_label(pred) -> str:
    if not pred or abs(pred.mispricing) < 0.10 or pred.confidence < 0.50:
        return "-"
    return "BUY_YES" if pred.mispricing > 0 else "BUY_NO"


def _build_summary(rows):
    if not rows:
        return []
    total = len(rows)
    with_pred = sum(1 for r in rows if r.get("prediction") is not None)
    buy_yes = sum(1 for r in rows if r.get("signal") == "BUY_YES")
    buy_no = sum(1 for r in rows if r.get("signal") == "BUY_NO")
    avg_misp = 0
    misp_vals = [abs(r["mispricing"]) for r in rows if r.get("mispricing") is not None]
    if misp_vals:
        avg_misp = sum(misp_vals) / len(misp_vals)

    cards = [
        _card("Markets", str(total), "info"),
        _card("Predictions", str(with_pred), "primary"),
        _card("BUY YES", str(buy_yes), "success"),
        _card("BUY NO", str(buy_no), "danger"),
        _card("Avg Mispricing", f"{avg_misp:.1%}", "warning"),
    ]
    return [dbc.Col(c, width=True) for c in cards]


def _card(title, value, color):
    return dbc.Card(
        dbc.CardBody([
            html.P(title, className="text-muted mb-1", style={"fontSize": "0.75rem"}),
            html.H4(value, className=f"text-{color} mb-0"),
        ]),
        className="bg-dark border-secondary text-center",
    )
