"""Incremental LLM-as-judge scoring pipeline for Claude Code sessions."""

from __future__ import annotations

import logging
import os
from argparse import ArgumentParser
from datetime import datetime, timezone

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

logger = logging.getLogger(__name__)

_REPLAY_CHAR_BUDGET = 30_000
_KEEP_INTERACTIONS = 2

RESPONSE_FORMAT = (
    '{"type":"json_object","schema":{"type":"object","properties":{"judgment":{"type":"object",'
    '"properties":{"task_clarity":{"type":"integer"},"agent_effectiveness":{"type":"integer"},'
    '"tool_strategy":{"type":"integer"},"error_handling":{"type":"integer"},'
    '"cost_efficiency":{"type":"integer"},"overall_score":{"type":"integer"},'
    '"summary":{"type":"string"},"recommendations":{"type":"string"}}}}}}'
)

JUDGMENT_SCHEMA = StructType(
    [
        StructField(
            "judgment",
            StructType(
                [
                    StructField("task_clarity", IntegerType()),
                    StructField("agent_effectiveness", IntegerType()),
                    StructField("tool_strategy", IntegerType()),
                    StructField("error_handling", IntegerType()),
                    StructField("cost_efficiency", IntegerType()),
                    StructField("overall_score", IntegerType()),
                    StructField("summary", StringType()),
                    StructField("recommendations", StringType()),
                ]
            ),
        )
    ]
)


def create_spark_session() -> SparkSession:
    """Return (or reuse) the active SparkSession for this job."""
    if os.environ.get("DATABRICKS_RUNTIME_VERSION") is None:
        try:
            from databricks.connect import DatabricksSession

            return DatabricksSession.builder.serverless().getOrCreate()
        except ImportError:
            return SparkSession.builder.getOrCreate()
    return SparkSession.builder.getOrCreate()


def format_event_line(row) -> str:
    """Render a single session event as a human-readable line for the LLM replay."""
    ts = getattr(row, "event_ts", "")
    etype = getattr(row, "event_type", "")
    name = getattr(row, "detail_name", "")
    model = getattr(row, "model", "")
    inp = getattr(row, "input_tokens", 0) or 0
    out = getattr(row, "output_tokens", 0) or 0
    cost = getattr(row, "cost_usd", 0.0) or 0.0
    err = getattr(row, "error_category", "") or ""
    preview = (getattr(row, "content_preview", "") or "")[:200]
    return f"{ts} [{etype}] {name} model={model} tokens={inp}/{out} cost=${cost:.4f} err={err} | {preview}"


def split_into_interactions(rows: list) -> list[list]:
    """Partition events into interactions, splitting at each USER_PROMPT boundary."""
    interactions: list[list] = []
    current: list = []
    for row in rows:
        if getattr(row, "event_type", "") == "USER_PROMPT" and current:
            interactions.append(current)
            current = []
        current.append(row)
    if current:
        interactions.append(current)
    return interactions or [[]]


def compress_interaction(events: list) -> str:
    """Summarise a middle interaction as a single compressed line to save replay budget."""
    from collections import Counter

    counts = Counter(getattr(r, "event_type", "UNKNOWN") for r in events)
    type_str = " ".join(f"{k}={v}" for k, v in counts.most_common())
    total_cost = sum(getattr(r, "cost_usd", 0.0) or 0.0 for r in events)
    total_ms = sum(getattr(r, "duration_ms", 0) or 0 for r in events)
    return f"--- [compressed: {len(events)} events, types: {type_str}, cost=${total_cost:.4f}, duration={total_ms}ms] ---"


def build_replay_text(
    rows: list,
    replay_char_budget: int = _REPLAY_CHAR_BUDGET,
    keep_interactions: int = _KEEP_INTERACTIONS,
) -> str:
    """Build the session replay string, keeping head/tail verbatim and compressing the middle."""
    interactions = split_into_interactions(rows)
    if len(interactions) <= keep_interactions * 2:
        text = "\n".join(format_event_line(r) for r in rows)
    else:
        keep_head = interactions[:keep_interactions]
        keep_tail = interactions[-keep_interactions:]
        middle = interactions[keep_interactions:-keep_interactions]
        parts: list[str] = []
        for chunk in keep_head:
            parts.extend(format_event_line(r) for r in chunk)
        for chunk in middle:
            parts.append(compress_interaction(chunk))
        for chunk in keep_tail:
            parts.extend(format_event_line(r) for r in chunk)
        text = "\n".join(parts)
    return text[:replay_char_budget]


@F.udf(StringType())
def _build_replay_udf(rows):
    """Spark UDF wrapper — collect_list rows → replay text for each session."""
    return build_replay_text(rows or [])


@F.udf(StringType())
def _build_prompt_udf(
    replay_text,
    num_interactions,
    total_cost_usd,
    cache_hit_rate,
    tool_success_rate,
    auto_accept_rate,
):
    """Spark UDF that renders the LLM judge prompt from session metrics and replay text."""
    replay = replay_text or "(no events)"
    cost = total_cost_usd or 0.0
    chr_rate = f"{(cache_hit_rate or 0):.2%}"
    tsr = f"{(tool_success_rate or 0):.2f}"
    aar = f"{(auto_accept_rate or 0):.2f}"
    return (
        "You are evaluating a Claude Code AI agent session. Score each dimension 0-100.\n\n"
        "IMPORTANT: INTERNAL_ERROR, BACKGROUND_ABORTED, and USER_ABORTED events are normal "
        "operational events - do NOT penalize for them.\n\n"
        f"Session replay:\n{replay}\n\n"
        f"Session metrics:\n"
        f"- interactions: {num_interactions}, cost: ${cost:.4f}, cache_hit_rate: {chr_rate}\n"
        f"- tool_success_rate: {tsr}, auto_accept_rate: {aar}\n\n"
        "Score dimensions (0-100):\n"
        "- task_clarity (15%): how clearly defined and focused was the task?\n"
        "- agent_effectiveness (25%): did the agent accomplish its goals?\n"
        "- tool_strategy (25%): were tools used efficiently and appropriately?\n"
        "- error_handling (15%): how well were errors managed?\n"
        "- cost_efficiency (20%): was token/cost usage appropriate?\n"
        "- overall_score: weighted summary of the above\n"
        "- summary: one paragraph\n"
        "- recommendations: actionable suggestions\n\n"
        "Return JSON only."
    )


def run_scoring(
    spark: SparkSession,
    target_catalog: str,
    target_schema: str,
    gold_schema: str,
    replay_char_budget: int = _REPLAY_CHAR_BUDGET,
    keep_interactions: int = _KEEP_INTERACTIONS,
) -> None:
    """Run the full incremental scoring pipeline: discover → replay → score → judge → merge."""
    silver_summary = f"{target_catalog}.{target_schema}.session_summary"
    silver_events = f"{target_catalog}.{target_schema}.session_events"
    gold_scores = f"{gold_schema}.session_scores"

    # Only score sessions whose last span ended before the start of today (UTC),
    # ensuring the session is complete before committing an immutable score.
    completed_sessions_df = spark.table(silver_summary).filter(
        F.col("session_end") < F.date_trunc("day", F.current_timestamp())
    )

    if spark.catalog.tableExists(gold_scores):
        existing = spark.table(gold_scores).select("session_id")
        new_sessions_df = completed_sessions_df.select("session_id").join(
            existing, "session_id", "left_anti"
        )
    else:
        new_sessions_df = completed_sessions_df.select("session_id")

    count = new_sessions_df.count()
    if count == 0:
        logger.info("No new sessions to score.")
        return

    logger.info("Scoring %d new sessions.", count)

    events_df = (
        spark.table(silver_events)
        .join(new_sessions_df, "session_id")
        .orderBy("session_id", "event_ts")
    )
    replay_df = (
        events_df.groupBy("session_id")
        .agg(
            F.collect_list(
                F.struct(
                    "event_ts",
                    "event_type",
                    "detail_name",
                    "model",
                    "input_tokens",
                    "output_tokens",
                    "cost_usd",
                    "error_category",
                    "content_preview",
                    "duration_ms",
                )
            ).alias("events")
        )
        .withColumn("replay_text", _build_replay_udf(F.col("events")))
        .select("session_id", "replay_text")
    )

    visible_errors = (
        spark.table(silver_events)
        .join(new_sessions_df, "session_id")
        .filter(F.col("error_category") == "user_visible")
        .groupBy("session_id")
        .agg(F.count("*").alias("visible_error_count"))
    )

    silver_df = spark.table(silver_summary).join(new_sessions_df, "session_id")

    scored_df = (
        silver_df.join(visible_errors, "session_id", "left")
        .withColumn(
            "visible_error_count", F.coalesce(F.col("visible_error_count"), F.lit(0))
        )
        .withColumn(
            "efficiency_score",
            F.least(
                F.lit(100.0),
                F.col("cache_hit_rate") * 60
                + F.greatest(
                    F.lit(0.0),
                    F.lit(40.0)
                    - (
                        F.col("total_cost_usd")
                        / F.greatest(
                            F.col("num_interactions").cast("double"), F.lit(1.0)
                        )
                    )
                    * 400,
                ),
            ),
        )
        .withColumn(
            "productivity_score",
            F.least(
                F.lit(100.0),
                F.least(
                    F.lit(50.0),
                    (
                        F.col("num_tool_calls")
                        / F.greatest(
                            F.col("num_interactions").cast("double"), F.lit(1.0)
                        )
                    )
                    * 25,
                )
                + F.least(F.lit(50.0), F.col("num_interactions").cast("double") * 10),
            ),
        )
        .withColumn(
            "quality_score",
            F.least(
                F.lit(100.0),
                F.coalesce(F.col("tool_success_rate"), F.lit(1.0)) * 70
                + F.greatest(
                    F.lit(0.0),
                    F.lit(30.0) - F.col("visible_error_count").cast("double") * 15,
                ),
            ),
        )
        .withColumn(
            "autonomy_score", F.coalesce(F.col("auto_accept_rate"), F.lit(0.0)) * 100
        )
        .withColumn(
            "engagement_score",
            F.least(
                F.lit(100.0),
                F.least(
                    F.lit(50.0),
                    F.col("session_duration_s").cast("double") / 60 * 50,
                )
                + F.least(
                    F.lit(50.0), F.coalesce(F.col("avg_prompt_length"), F.lit(0.0))
                ),
            ),
        )
        .withColumn(
            "composite_score",
            F.col("efficiency_score") * 0.20
            + F.col("productivity_score") * 0.25
            + F.col("quality_score") * 0.20
            + F.col("autonomy_score") * 0.15
            + F.col("engagement_score") * 0.20,
        )
    )

    with_prompt = scored_df.join(replay_df, "session_id", "left").withColumn(
        "full_prompt",
        _build_prompt_udf(
            F.col("replay_text"),
            F.col("num_interactions"),
            F.col("total_cost_usd"),
            F.col("cache_hit_rate"),
            F.col("tool_success_rate"),
            F.col("auto_accept_rate"),
        ),
    )

    ai_result = with_prompt.withColumn(
        "ai_response",
        F.expr(
            f"ai_query('databricks-claude-sonnet-4', full_prompt, responseFormat => '{RESPONSE_FORMAT}')"
        ),
    ).withColumn(
        "judgment",
        F.from_json(F.col("ai_response"), JUDGMENT_SCHEMA).getField("judgment"),
    )

    gold_df = ai_result.select(
        "session_id",
        "user_id",
        "session_start",
        "num_interactions",
        "total_cost_usd",
        "efficiency_score",
        "productivity_score",
        "quality_score",
        "autonomy_score",
        "engagement_score",
        "composite_score",
        F.col("judgment.task_clarity").alias("llm_task_clarity"),
        F.col("judgment.agent_effectiveness").alias("llm_agent_effectiveness"),
        F.col("judgment.tool_strategy").alias("llm_tool_strategy"),
        F.col("judgment.error_handling").alias("llm_error_handling"),
        F.col("judgment.cost_efficiency").alias("llm_cost_efficiency"),
        F.col("judgment.overall_score").alias("llm_overall_score"),
        F.col("judgment.summary").alias("llm_summary"),
        F.col("judgment.recommendations").alias("llm_recommendations"),
        F.lit(datetime.now(timezone.utc)).cast("timestamp").alias("scored_at"),
    )

    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {gold_scores} (
            session_id STRING,
            user_id STRING,
            session_start TIMESTAMP,
            num_interactions LONG,
            total_cost_usd DOUBLE,
            efficiency_score DOUBLE,
            productivity_score DOUBLE,
            quality_score DOUBLE,
            autonomy_score DOUBLE,
            engagement_score DOUBLE,
            composite_score DOUBLE,
            llm_task_clarity INT,
            llm_agent_effectiveness INT,
            llm_tool_strategy INT,
            llm_error_handling INT,
            llm_cost_efficiency INT,
            llm_overall_score INT,
            llm_summary STRING,
            llm_recommendations STRING,
            scored_at TIMESTAMP
        ) USING DELTA
        CLUSTER BY AUTO
        """
    )

    gold_df.createOrReplaceTempView("session_scores_updates")
    spark.sql(
        f"""
        MERGE INTO {gold_scores} AS target
        USING session_scores_updates AS source
        ON target.session_id = source.session_id
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
        """
    )
    logger.info("Scored %d sessions into %s.", count, gold_scores)


def main() -> None:
    """Entry point: parse args, create Spark session, run scoring, stop Spark."""
    parser = ArgumentParser(description="Score Claude Code sessions using LLM-as-judge")
    parser.add_argument("--target-catalog", required=True)
    parser.add_argument("--target-schema", required=True)
    parser.add_argument(
        "--gold-schema",
        required=True,
        help="Full catalog.schema, e.g. prod.claude_gold",
    )
    args = parser.parse_args()
    spark = create_spark_session()
    try:
        run_scoring(spark, args.target_catalog, args.target_schema, args.gold_schema)
    finally:
        spark.stop()
