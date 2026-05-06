"""Deterministic human-input scoring pipeline for Claude Code sessions.

Computes friction signals (rejections, aborts, corrections) per session and
per tool, written to two sibling gold tables. Recomputable on every silver
refresh — unlike the LLM-judge pipeline in scorer.py, scores here are not
immutable.
"""

from __future__ import annotations

import logging
import os
from argparse import ArgumentParser
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from claude_otel_session_scorer._spark import create_spark_session

logger = logging.getLogger(__name__)

# Window for "USER_PROMPT after TOOL_RESULT counts as a correction."
# 30 seconds is the detection threshold; the predicate is boundary-inclusive (<=).
_CORRECTION_WINDOW_SECONDS = 30

# Shared weights for the friction-score formula (spec AC#6).
# Both the pure-Python helper and the SQL expression are derived from this dict
# so they're guaranteed to stay in sync.
_SCORE_WEIGHTS = {"reject": 0.4, "abort": 0.3, "correction": 0.3}


def compute_friction_score(
    reject_rate: float | None,
    abort_rate: float | None,
    correction_intensity: float | None,
    signal_strength: bool,
) -> float | None:
    """Pure-Python mirror of the spec AC#6 SQL formula.

    Returns NULL (None) when signal_strength is False — preserving the spec's
    "NULL not 0" contract for sessions with no human-input signal.
    """
    if not signal_strength:
        return None
    raw = 100.0 * (
        _SCORE_WEIGHTS["reject"] * (reject_rate or 0.0)
        + _SCORE_WEIGHTS["abort"] * (abort_rate or 0.0)
        + _SCORE_WEIGHTS["correction"] * (correction_intensity or 0.0)
    )
    return min(100.0, max(0.0, raw))


def run_human_signals(
    spark: SparkSession,
    silver_schema: str,
    gold_schema: str,
) -> None:
    """Run the deterministic human-input scoring pipeline.

    Reads from silver.session_events / session_summary / session_metrics, writes
    to gold.session_human_signals (one row per session) and
    gold.session_human_signals_by_tool (one row per (session_id, tool_name)).
    Both tables are recomputable on every run via MERGE WHEN MATCHED UPDATE SET *.
    """
    silver_summary = f"{silver_schema}.session_summary"
    silver_events = f"{silver_schema}.session_events"
    silver_metrics = f"{silver_schema}.session_metrics"
    gold_session = f"{gold_schema}.session_human_signals"
    gold_by_tool = f"{gold_schema}.session_human_signals_by_tool"

    completed_sessions = spark.table(silver_summary).filter(
        F.col("session_end") < F.current_timestamp() - F.expr("INTERVAL 2 HOURS")
    )
    session_keys = completed_sessions.select(
        "session_id", "user_id", "session_start", "num_interactions"
    )

    count = session_keys.count()
    if count == 0:
        logger.info("No completed sessions to score.")
        return

    logger.info("Scoring %d completed sessions (recomputable).", count)

    events = spark.table(silver_events).join(session_keys.select("session_id"), "session_id")

    # --- per-session counts -------------------------------------------------
    decision_counts = (
        events.filter(F.col("event_type") == "TOOL_DECISION")
        .filter(F.col("detail_name").isin("accept", "reject"))
        .groupBy("session_id")
        .agg(
            F.count("*").alias("num_tool_decisions"),
            F.sum(F.when(F.col("detail_name") == "reject", 1).otherwise(0)).alias(
                "num_tool_rejects"
            ),
            F.sum(F.when(F.col("detail_name") == "accept", 1).otherwise(0)).alias(
                "num_tool_accepts"
            ),
        )
    )

    abort_counts = (
        events.filter(F.col("event_type") == "USER_ABORTED")
        .groupBy("session_id")
        .agg(F.count("*").alias("num_user_aborts"))
    )

    # --- per-session corrections via deterministic window ------------------
    # event_ts has integer-second precision; monotonically_increasing_id() is
    # a stable intra-plan tiebreaker for same-second events.
    correction_window = Window.partitionBy("session_id").orderBy(
        F.col("event_ts").asc(),
        F.col("event_type").asc(),
        F.monotonically_increasing_id().asc(),
    )
    flagged = (
        events.withColumn("event_ts_sec", F.col("event_ts").cast("long"))
        .withColumn("_prev_event_type", F.lag("event_type").over(correction_window))
        .withColumn("_prev_event_ts_sec", F.lag("event_ts_sec").over(correction_window))
    )
    correction_counts = (
        flagged.filter(F.col("event_type") == "USER_PROMPT")
        .filter(F.col("_prev_event_type") == "TOOL_RESULT")
        .filter(F.col("event_ts_sec") - F.col("_prev_event_ts_sec") <= _CORRECTION_WINDOW_SECONDS)
        .groupBy("session_id")
        .agg(F.count("*").alias("num_corrections"))
    )

    # --- assemble session-grain row ----------------------------------------
    session_metrics = spark.table(silver_metrics).select("session_id", "primary_model")

    session_grain = (
        session_keys.join(decision_counts, "session_id", "left")
        .join(abort_counts, "session_id", "left")
        .join(correction_counts, "session_id", "left")
        .join(session_metrics, "session_id", "left")
        .withColumn("num_tool_decisions", F.coalesce(F.col("num_tool_decisions"), F.lit(0)))
        .withColumn("num_tool_rejects", F.coalesce(F.col("num_tool_rejects"), F.lit(0)))
        .withColumn("num_tool_accepts", F.coalesce(F.col("num_tool_accepts"), F.lit(0)))
        .withColumn("num_user_aborts", F.coalesce(F.col("num_user_aborts"), F.lit(0)))
        .withColumn("num_corrections", F.coalesce(F.col("num_corrections"), F.lit(0)))
        .withColumn(
            "reject_rate",
            F.expr(
                "CASE WHEN num_tool_decisions > 0 "
                "THEN CAST(num_tool_rejects AS DOUBLE) / CAST(num_tool_decisions AS DOUBLE) "
                "ELSE NULL END"
            ),
        )
        .withColumn(
            "abort_rate",
            F.expr(
                "CASE WHEN num_interactions > 0 "
                "THEN CAST(num_user_aborts AS DOUBLE) / CAST(num_interactions AS DOUBLE) "
                "ELSE NULL END"
            ),
        )
        .withColumn(
            "correction_intensity",
            F.expr(
                "CASE WHEN num_interactions > 0 "
                "THEN CAST(num_corrections AS DOUBLE) / CAST(num_interactions AS DOUBLE) "
                "ELSE NULL END"
            ),
        )
        .withColumn(
            "signal_strength",
            F.expr("num_tool_decisions >= 1 OR num_user_aborts >= 1 OR num_corrections >= 1"),
        )
        .withColumn(
            "human_friction_score",
            F.expr(
                "CASE WHEN signal_strength THEN "
                "LEAST(100.0, GREATEST(0.0, "
                f"100.0 * ({_SCORE_WEIGHTS['reject']} * COALESCE(reject_rate, 0.0) "
                f"+ {_SCORE_WEIGHTS['abort']} * COALESCE(abort_rate, 0.0) "
                f"+ {_SCORE_WEIGHTS['correction']} * COALESCE(correction_intensity, 0.0)))) "
                "ELSE NULL END"
            ),
        )
        .withColumn(
            "computed_at",
            F.current_timestamp(),
        )
        .select(
            "session_id",
            "user_id",
            "primary_model",
            "session_start",
            "num_interactions",
            "num_tool_decisions",
            "num_tool_rejects",
            "num_tool_accepts",
            "num_user_aborts",
            "num_corrections",
            "reject_rate",
            "abort_rate",
            "correction_intensity",
            "human_friction_score",
            "signal_strength",
            "computed_at",
        )
    )

    # --- per-tool grain -----------------------------------------------------
    by_tool_base = (
        events.filter(F.col("event_type") == "TOOL_DECISION")
        .filter(F.col("detail_name").isin("accept", "reject"))
        .filter(F.col("tool_name").isNotNull())
        .groupBy("session_id", "tool_name")
        .agg(
            F.count("*").alias("num_tool_decisions"),
            F.sum(F.when(F.col("detail_name") == "reject", 1).otherwise(0)).alias(
                "num_tool_rejects"
            ),
            F.sum(F.when(F.col("detail_name") == "accept", 1).otherwise(0)).alias(
                "num_tool_accepts"
            ),
        )
    )
    by_tool = (
        by_tool_base.join(session_keys.select("session_id", "user_id"), "session_id", "inner")
        .join(session_metrics, "session_id", "left")
        .withColumn(
            "reject_rate",
            F.expr(
                "CASE WHEN num_tool_decisions > 0 "
                "THEN CAST(num_tool_rejects AS DOUBLE) / CAST(num_tool_decisions AS DOUBLE) "
                "ELSE NULL END"
            ),
        )
        .withColumn(
            "computed_at",
            F.current_timestamp(),
        )
        .select(
            "session_id",
            "user_id",
            "primary_model",
            "tool_name",
            "num_tool_decisions",
            "num_tool_rejects",
            "num_tool_accepts",
            "reject_rate",
            "computed_at",
        )
    )

    # --- write ----------------------------------------------------------------
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {gold_schema}")
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {gold_session} (
            session_id STRING,
            user_id STRING,
            primary_model STRING,
            session_start TIMESTAMP,
            num_interactions LONG,
            num_tool_decisions LONG,
            num_tool_rejects LONG,
            num_tool_accepts LONG,
            num_user_aborts LONG,
            num_corrections LONG,
            reject_rate DOUBLE,
            abort_rate DOUBLE,
            correction_intensity DOUBLE,
            human_friction_score DOUBLE,
            signal_strength BOOLEAN,
            computed_at TIMESTAMP
        ) USING DELTA
        CLUSTER BY AUTO
        """
    )
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {gold_by_tool} (
            session_id STRING,
            user_id STRING,
            primary_model STRING,
            tool_name STRING,
            num_tool_decisions LONG,
            num_tool_rejects LONG,
            num_tool_accepts LONG,
            reject_rate DOUBLE,
            computed_at TIMESTAMP
        ) USING DELTA
        CLUSTER BY AUTO
        """
    )

    session_grain.createOrReplaceTempView("session_human_signals_updates")
    spark.sql(
        f"""
        MERGE INTO {gold_session} AS target
        USING session_human_signals_updates AS source
        ON target.session_id = source.session_id
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
        """
    )

    # Per-tool: delete stale rows for recomputed sessions, then MERGE the
    # current rows. This handles the disappearing-tool case where a session
    # no longer has a TOOL_DECISION for some tool that was present last run.
    session_keys.select("session_id").createOrReplaceTempView("human_signals_recomputed_sessions")
    spark.sql(
        f"DELETE FROM {gold_by_tool} "
        f"WHERE session_id IN (SELECT session_id FROM human_signals_recomputed_sessions)"
    )
    by_tool.createOrReplaceTempView("session_human_signals_by_tool_updates")
    spark.sql(
        f"""
        MERGE INTO {gold_by_tool} AS target
        USING session_human_signals_by_tool_updates AS source
        ON target.session_id = source.session_id
            AND target.tool_name = source.tool_name
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
        """
    )
    logger.info(
        "Wrote human-input signals for %d sessions to %s and %s.",
        count,
        gold_session,
        gold_by_tool,
    )


def main() -> None:
    """Entry point: parse args, create Spark session, run pipeline, stop Spark."""
    parser = ArgumentParser(
        description="Score Claude Code sessions on deterministic human-input signals"
    )
    parser.add_argument(
        "--silver-schema",
        required=True,
        help="Full catalog.schema, e.g. prod.claude_silver",
    )
    parser.add_argument(
        "--gold-schema",
        required=True,
        help="Full catalog.schema, e.g. prod.claude_gold",
    )
    args, _ = parser.parse_known_args()
    spark = create_spark_session()
    try:
        run_human_signals(spark, args.silver_schema, args.gold_schema)
    finally:
        if os.environ.get("DATABRICKS_RUNTIME_VERSION") is None:
            spark.stop()


if __name__ == "__main__":
    main()
