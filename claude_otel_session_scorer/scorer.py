"""Incremental LLM-as-judge scoring pipeline for Claude Code sessions.

IMMUTABILITY CONTRACT — READ THIS BEFORE MODIFYING THE SCORING LOGIC
======================================================================

The 2-hour idle heuristic
--------------------------
This pipeline considers a session "complete" when its ``session_end`` timestamp
is more than **2 hours** in the past (``session_end < current_timestamp() - INTERVAL
2 HOURS``).  This is a **practical heuristic**, not a true completion signal.
Claude Code does not emit an explicit "session closed" event; we infer
completeness from observed inactivity.

Consequence: sessions that resume after a long idle gap
--------------------------------------------------------
If a user pauses a session for more than 2 hours and then resumes it:

1. The session looks "complete" at the 2-hour mark — it clears the idle filter.
2. On the next pipeline run the ``left_anti`` join (see below) finds no existing
   gold row, so the *partial* session gets scored and written to
   ``gold.session_scores``.
3. When the user resumes and new silver events arrive, the session's
   ``session_end`` advances — but the pipeline's ``left_anti`` join now
   *excludes* it because a gold row already exists.
4. **The gold score is never updated.**  The original (partial) score silently
   remains as the permanent record for that session.

The ``left_anti`` immutability guard
-------------------------------------
``gold.session_scores`` is an **append-only, write-once** table.  The
``left_anti`` join against existing rows in ``run_scoring`` is the enforcement
mechanism: once a session_id appears in ``gold.session_scores`` it is
permanently excluded from future scoring runs.  This is intentional — re-scoring
would be expensive (each score costs an ``ai_query`` LLM call) and would
destroy the stable historical record that downstream dashboards and analyses
depend on.

When this behaviour is safe
-----------------------------
- **Interactive development sessions** where the user actively works and wraps
  up within a couple of hours.  These are the expected common case.
- **Short automated agent runs** that complete well within the 2-hour window.

When this behaviour is NOT safe (silent data loss)
----------------------------------------------------
- **Long-running overnight or multi-day jobs** — a session started in the
  evening and resumed the next morning will be silently split: the first
  partial chunk is scored permanently, and the resumed work is treated as a
  brand-new session.
- **Sessions that span multiple working days** — same problem; each >2-hour
  idle gap creates a new effective session boundary.

If these cases apply to your workload, consider:
- Increasing or making configurable the idle threshold (``INTERVAL 2 HOURS``).
- Emitting an explicit session-close signal from the OTLP proxy so the filter
  can key on a real completion event instead of inferred inactivity.
- Relaxing immutability to allow re-scoring within a grace window after the
  first score is written.

Schema and locking notes
-------------------------
``RESPONSE_FORMAT``, ``FLAT_SCHEMA``, and the ``CREATE TABLE IF NOT EXISTS``
DDL for ``gold.session_scores`` must always stay in sync.  Adding or removing
a judgment field requires updating all three and accompanying a PR with an
explicit ``ALTER TABLE`` migration note (see AGENTS.md §3).
"""

from __future__ import annotations

import logging
import os
from argparse import ArgumentParser
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

from claude_otel_session_scorer._spark import create_spark_session

logger = logging.getLogger(__name__)

_REPLAY_CHAR_BUDGET = 30_000
_KEEP_INTERACTIONS = 2

RESPONSE_FORMAT = (
    "STRUCT<judgment STRUCT<"
    "task_clarity INT, agent_effectiveness INT, tool_strategy INT, "
    "error_handling INT, cost_efficiency INT, overall_score INT, "
    "summary STRING, recommendations STRING>>"
)

FLAT_SCHEMA = (
    "task_clarity INT, agent_effectiveness INT, tool_strategy INT, "
    "error_handling INT, cost_efficiency INT, overall_score INT, "
    "summary STRING, recommendations STRING"
)


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
    silver_schema: str,
    gold_schema: str,
    replay_char_budget: int = _REPLAY_CHAR_BUDGET,
    keep_interactions: int = _KEEP_INTERACTIONS,
) -> None:
    """Run the full incremental scoring pipeline: discover → replay → score → judge → merge."""
    silver_summary = f"{silver_schema}.session_summary"
    silver_events = f"{silver_schema}.session_events"
    gold_scores = f"{gold_schema}.session_scores"

    # 2-hour idle gap heuristic — not a true completion signal.
    # Sessions resuming after >2h are treated as new. (See module docstring.)
    completed_sessions_df = spark.table(silver_summary).filter(
        F.col("session_end") < F.current_timestamp() - F.expr("INTERVAL 2 HOURS")
    )

    if spark.catalog.tableExists(gold_scores):
        existing = spark.table(gold_scores).select("session_id")
        # Immutability guard: sessions already scored are excluded. See module docstring.
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

    events_df = spark.table(silver_events).join(new_sessions_df, "session_id")
    replay_df = (
        events_df.groupBy("session_id")
        .agg(
            F.sort_array(
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
        .withColumn("visible_error_count", F.expr("COALESCE(visible_error_count, 0)"))
        .withColumn(
            "efficiency_score",
            F.expr("""
                LEAST(100.0,
                    COALESCE(cache_hit_rate, 0.0) * 60
                    + GREATEST(0.0,
                        40.0 - (COALESCE(total_cost_usd, 0.0)
                                / GREATEST(CAST(num_interactions AS DOUBLE), 1.0)) * 400
                    )
                )
            """),
        )
        .withColumn(
            "productivity_score",
            F.expr("""
                LEAST(100.0,
                    LEAST(50.0,
                        COALESCE(num_tool_calls, 0)
                        / GREATEST(CAST(num_interactions AS DOUBLE), 1.0) * 25
                    )
                    + LEAST(50.0, CAST(num_interactions AS DOUBLE) * 10)
                )
            """),
        )
        .withColumn(
            "quality_score",
            F.expr("""
                LEAST(100.0,
                    COALESCE(tool_success_rate, 1.0) * 70
                    + GREATEST(0.0, 30.0 - CAST(visible_error_count AS DOUBLE) * 15)
                )
            """),
        )
        .withColumn("autonomy_score", F.expr("COALESCE(auto_accept_rate, 0.0) * 100"))
        .withColumn(
            "engagement_score",
            F.expr("""
                LEAST(100.0,
                    LEAST(50.0, COALESCE(CAST(session_duration_s AS DOUBLE), 0.0) / 60 * 50)
                    + LEAST(50.0, COALESCE(avg_prompt_length, 0.0))
                )
            """),
        )
        .withColumn(
            "composite_score",
            F.expr("""
                efficiency_score * 0.20
                + productivity_score * 0.25
                + quality_score * 0.20
                + autonomy_score * 0.15
                + engagement_score * 0.20
            """),
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
        F.from_json(F.col("ai_response"), FLAT_SCHEMA),
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
        F.current_timestamp().alias("scored_at"),
    )
    gold_df = gold_df.filter(F.col("llm_overall_score").isNotNull())

    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {gold_schema}")
    # gold.session_scores is write-once / immutable per session_id.
    # The left_anti join above enforces this: existing rows are never updated.
    # See module docstring for the immutability contract and its trade-offs.
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
        run_scoring(spark, args.silver_schema, args.gold_schema)
    finally:
        if os.environ.get("DATABRICKS_RUNTIME_VERSION") is None:
            spark.stop()
