import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

def run_analysis_cycle(ingestion, prediction_engine, trading_engine, db,
                       min_mispricing: float = 0.10, min_confidence: float = 0.80):
    start = datetime.now(timezone.utc)
    logger.info("=== Analysis cycle started ===")
    db.save_log("INFO", "scheduler", "Analysis cycle started")
    snapshots = ingestion.run()
    logger.info(f"Ingested {len(snapshots)} market snapshots")
    if not snapshots:
        db.save_log("WARNING", "scheduler", "No snapshots ingested")
        return
    predictions = prediction_engine.predict_batch(snapshots)
    logger.info(f"Generated {len(predictions)} predictions")
    signals = [p for p in predictions if p.has_signal(min_mispricing, min_confidence)]
    logger.info(f"Found {len(signals)} actionable signals")
    if signals:
        trades = trading_engine.process_batch(signals)
        logger.info(f"Executed {len(trades)} trades")
        db.save_log("INFO", "scheduler", f"Cycle complete: {len(trades)} trades from {len(signals)} signals")
    else:
        db.save_log("INFO", "scheduler", "Cycle complete: no actionable signals")
    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    logger.info(f"=== Cycle completed in {elapsed:.1f}s ===")
