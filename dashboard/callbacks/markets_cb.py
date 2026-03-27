import logging
from dash import Input, Output, ctx, no_update

logger = logging.getLogger(__name__)


def register_markets_callbacks(app, db, ingestion=None, prediction_engine=None):
    @app.callback(
        Output("markets-table", "data"),
        Input("markets-interval", "n_intervals"),
        Input("btn-analyze", "n_clicks"),
    )
    def update_markets_table(n, n_clicks):
        triggered = ctx.triggered_id

        # If button was clicked, run full pipeline: ingest → predict
        if triggered == "btn-analyze" and n_clicks:
            snapshots = []
            if ingestion:
                try:
                    snapshots = ingestion.run(top_n=20)
                    logger.info(f"Ingested {len(snapshots)} markets")
                except Exception as e:
                    logger.error(f"Ingestion failed: {e}")

            if snapshots and prediction_engine:
                try:
                    predictions = prediction_engine.predict_batch(snapshots)
                    logger.info(f"Generated {len(predictions)} predictions")
                except Exception as e:
                    logger.error(f"Prediction failed: {e}")

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
        return rows

    @app.callback(
        Output("selected-market-store", "data"),
        Input("markets-table", "selected_rows"),
        Input("markets-table", "data"),
    )
    def store_selected_market(selected_rows, data):
        if not selected_rows or not data:
            return no_update
        idx = selected_rows[0]
        return data[idx].get("market_id")


def _signal_label(pred) -> str:
    if not pred or abs(pred.mispricing) < 0.10 or pred.confidence < 0.80:
        return "-"
    return "BUY_YES" if pred.mispricing > 0 else "BUY_NO"
