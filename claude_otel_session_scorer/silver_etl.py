"""Silver ETL job: transform bronze OTEL tables into session-level silver tables.

Reads from <source_catalog>.<source_schema>.{traces, metrics, logs} and writes
session_summary, session_events, and session_metrics into the target schema.
"""

import argparse
import logging
import os

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

logger = logging.getLogger(__name__)


def create_spark_session() -> SparkSession:
    """Create or get the active Spark session.

    Mirrors the pattern used in main.py: use DatabricksSession serverless when
    running locally with databricks-connect, otherwise fall back to standard
    PySpark (used both on Databricks runtime and in tests).
    """
    if os.environ.get("DATABRICKS_RUNTIME_VERSION") is None:
        try:
            from databricks.connect import DatabricksSession

            return DatabricksSession.builder.serverless().getOrCreate()
        except ImportError:
            print("Databricks Connect not available. Falling back to standard Spark session.")
            return SparkSession.builder.getOrCreate()
    else:
        return SparkSession.builder.getOrCreate()


def _safe_event_attr(attr_key: str) -> Column:
    """Return the value of ``events[0].attributes[attr_key]`` if any events exist.

    Avoids errors on rows where ``events`` is empty/null by guarding with
    ``size(events) > 0``.
    """
    return F.when(
        F.size(F.col("events")) > 0,
        F.col("events")[0]["attributes"].getItem(attr_key),
    )


# ---------------------------------------------------------------------------
# session_summary
# ---------------------------------------------------------------------------


def transform_session_summary(traces_df: DataFrame, metrics_df: DataFrame) -> DataFrame:
    """Aggregate bronze traces + metrics into a per-session summary row."""

    interactions = (
        traces_df.filter(F.col("name") == "claude_code.interaction")
        .groupBy(F.col("attributes").getItem("session.id").alias("session_id"))
        .agg(
            F.first(F.col("attributes").getItem("user.id")).alias("user_id"),
            F.count("*").alias("num_interactions"),
            F.min((F.col("start_time_unix_nano") / 1e9).cast("timestamp")).alias("session_start"),
            F.max((F.col("end_time_unix_nano") / 1e9).cast("timestamp")).alias("session_end"),
            F.avg(F.col("attributes").getItem("user_prompt_length").cast("int")).alias(
                "avg_prompt_length"
            ),
            F.first(F.col("resource.attributes").getItem("service.version")).alias(
                "service_version"
            ),
            F.first(F.col("resource.attributes").getItem("os.type")).alias("os_type"),
            F.first(F.col("attributes").getItem("terminal.type")).alias("terminal_type"),
        )
    )

    llm_stats = (
        traces_df.filter(F.col("name") == "claude_code.llm_request")
        .groupBy(F.col("attributes").getItem("session.id").alias("session_id"))
        .agg(
            F.count("*").alias("num_llm_requests"),
            F.sum(F.col("attributes").getItem("input_tokens").cast("long")).alias(
                "total_input_tokens"
            ),
            F.sum(F.col("attributes").getItem("output_tokens").cast("long")).alias(
                "total_output_tokens"
            ),
            F.sum(F.col("attributes").getItem("cache_read_tokens").cast("long")).alias(
                "total_cache_read"
            ),
            F.sum(F.col("attributes").getItem("cache_creation_tokens").cast("long")).alias(
                "total_cache_creation"
            ),
            F.avg(F.col("attributes").getItem("ttft_ms").cast("double")).alias("avg_ttft_ms"),
            F.avg(F.col("attributes").getItem("duration_ms").cast("double")).alias(
                "avg_llm_duration_ms"
            ),
        )
    )

    tool_stats = (
        traces_df.filter(F.col("name") == "claude_code.tool")
        .groupBy(F.col("attributes").getItem("session.id").alias("session_id"))
        .agg(
            F.count("*").alias("num_tool_calls"),
            F.countDistinct(F.col("attributes").getItem("tool_name")).alias("distinct_tools"),
        )
    )

    tool_exec = (
        traces_df.filter(F.col("name") == "claude_code.tool.execution")
        .groupBy(F.col("attributes").getItem("session.id").alias("session_id"))
        .agg(
            F.count("*").alias("num_tool_executions"),
            F.sum(
                F.when(F.col("attributes").getItem("success") == "true", 1).otherwise(0)
            ).alias("tool_successes"),
        )
    )

    autonomy = (
        traces_df.filter(F.col("name") == "claude_code.tool.blocked_on_user")
        .groupBy(F.col("attributes").getItem("session.id").alias("session_id"))
        .agg(
            F.count("*").alias("num_blocked_on_user"),
            F.sum(
                F.when(F.col("attributes").getItem("decision") == "accept", 1).otherwise(0)
            ).alias("auto_accepted"),
        )
    )

    cost = (
        metrics_df.filter(F.col("name") == "claude_code.cost.usage")
        .groupBy(F.col("sum.attributes").getItem("session.id").alias("session_id"))
        .agg(F.sum("sum.value").alias("total_cost_usd"))
    )

    active_time = (
        metrics_df.filter(F.col("name") == "claude_code.active_time.total")
        .groupBy(F.col("sum.attributes").getItem("session.id").alias("session_id"))
        .agg(F.sum("sum.value").alias("total_active_time_s"))
    )

    session_summary = (
        interactions.join(llm_stats, "session_id", "left")
        .join(tool_stats, "session_id", "left")
        .join(tool_exec, "session_id", "left")
        .join(autonomy, "session_id", "left")
        .join(cost, "session_id", "left")
        .join(active_time, "session_id", "left")
        .withColumn(
            "session_duration_s",
            F.unix_timestamp("session_end") - F.unix_timestamp("session_start"),
        )
        .withColumn(
            "cache_hit_rate",
            F.coalesce(F.col("total_cache_read"), F.lit(0))
            / F.greatest(
                F.col("total_input_tokens")
                + F.col("total_output_tokens")
                + F.coalesce(F.col("total_cache_read"), F.lit(0))
                + F.coalesce(F.col("total_cache_creation"), F.lit(0)),
                F.lit(1),
            ),
        )
        .withColumn(
            "tool_success_rate",
            F.when(
                F.col("num_tool_executions") > 0,
                F.col("tool_successes") / F.col("num_tool_executions"),
            ),
        )
        .withColumn(
            "auto_accept_rate",
            F.when(
                F.col("num_blocked_on_user") > 0,
                F.col("auto_accepted") / F.col("num_blocked_on_user"),
            ),
        )
        .withColumn(
            "tools_per_interaction",
            F.when(
                F.col("num_interactions") > 0,
                F.col("num_tool_calls") / F.col("num_interactions"),
            ),
        )
        .withColumn(
            "llm_calls_per_interaction",
            F.when(
                F.col("num_interactions") > 0,
                F.col("num_llm_requests") / F.col("num_interactions"),
            ),
        )
        .select(
            "session_id",
            "user_id",
            "session_start",
            "session_end",
            "session_duration_s",
            "num_interactions",
            "num_llm_requests",
            "num_tool_calls",
            "total_input_tokens",
            "total_output_tokens",
            "total_cache_read",
            "total_cache_creation",
            "cache_hit_rate",
            "avg_ttft_ms",
            "avg_llm_duration_ms",
            "tool_success_rate",
            "auto_accept_rate",
            "tools_per_interaction",
            "llm_calls_per_interaction",
            "avg_prompt_length",
            "total_cost_usd",
            "total_active_time_s",
            "service_version",
            "os_type",
            "terminal_type",
        )
    )

    return session_summary


# ---------------------------------------------------------------------------
# session_events
# ---------------------------------------------------------------------------


def transform_session_events(traces_df: DataFrame, logs_df: DataFrame) -> DataFrame:
    """Build a unified per-session timeline of events.

    Sources six event types from traces (user prompts, tool calls, tool results,
    api errors, internal errors, llm requests) plus warning/error log records.
    """

    session_id = F.col("attributes").getItem("session.id").alias("session_id")
    start_ts = (F.col("start_time_unix_nano") / 1e9).cast("timestamp").alias("event_time")

    # ── USER_PROMPT events from claude_code.interaction spans ──
    user_prompts = (
        traces_df.filter(F.col("name") == "claude_code.interaction")
        .select(
            session_id,
            start_ts,
            F.lit("USER_PROMPT").alias("event_type"),
            F.lit(None).cast("string").alias("error_category"),
            F.col("attributes").getItem("user_prompt_length").cast("int").alias("prompt_length"),
            F.lit(None).cast("string").alias("tool_name"),
            F.lit(None).cast("string").alias("error_message"),
            F.substring(_safe_event_attr("prompt"), 1, 500).alias("content_preview"),
        )
    )

    # ── TOOL_CALL ──
    tool_calls = (
        traces_df.filter(F.col("name") == "claude_code.tool")
        .select(
            session_id,
            start_ts,
            F.lit("TOOL_CALL").alias("event_type"),
            F.lit(None).cast("string").alias("error_category"),
            F.lit(None).cast("int").alias("prompt_length"),
            F.col("attributes").getItem("tool_name").alias("tool_name"),
            F.lit(None).cast("string").alias("error_message"),
            F.lit(None).cast("string").alias("content_preview"),
        )
    )

    # ── TOOL_RESULT ──
    tool_results = (
        traces_df.filter(F.col("name") == "claude_code.tool.execution")
        .select(
            session_id,
            start_ts,
            F.lit("TOOL_RESULT").alias("event_type"),
            F.lit(None).cast("string").alias("error_category"),
            F.lit(None).cast("int").alias("prompt_length"),
            F.col("attributes").getItem("tool_name").alias("tool_name"),
            F.lit(None).cast("string").alias("error_message"),
            F.substring(_safe_event_attr("result"), 1, 500).alias("content_preview"),
        )
    )

    # ── api_error spans → ERROR / BACKGROUND_ABORTED / USER_ABORTED ──
    api_errors = (
        traces_df.filter(F.col("name") == "claude_code.api_error")
        .withColumn("query_source", F.col("attributes").getItem("query_source"))
        .withColumn("error_msg", F.col("attributes").getItem("error"))
        .select(
            session_id,
            start_ts,
            F.when(F.col("query_source") == "away_summary", F.lit("BACKGROUND_ABORTED"))
            .when(F.col("error_msg") == "Request was aborted.", F.lit("USER_ABORTED"))
            .otherwise(F.lit("ERROR"))
            .alias("event_type"),
            F.when(F.col("query_source") == "away_summary", F.lit("invisible"))
            .when(F.col("error_msg") == "Request was aborted.", F.lit("invisible"))
            .otherwise(F.lit("user_visible"))
            .alias("error_category"),
            F.lit(None).cast("int").alias("prompt_length"),
            F.lit(None).cast("string").alias("tool_name"),
            F.col("error_msg").alias("error_message"),
            F.lit(None).cast("string").alias("content_preview"),
        )
    )

    # ── internal_error spans → INTERNAL_ERROR / invisible ──
    internal_errors = (
        traces_df.filter(F.col("name") == "claude_code.internal_error")
        .select(
            session_id,
            start_ts,
            F.lit("INTERNAL_ERROR").alias("event_type"),
            F.lit("invisible").alias("error_category"),
            F.lit(None).cast("int").alias("prompt_length"),
            F.lit(None).cast("string").alias("tool_name"),
            F.col("attributes").getItem("error").alias("error_message"),
            F.lit(None).cast("string").alias("content_preview"),
        )
    )

    # ── LLM_REQUEST ──
    llm_requests = (
        traces_df.filter(F.col("name") == "claude_code.llm_request")
        .select(
            session_id,
            start_ts,
            F.lit("LLM_REQUEST").alias("event_type"),
            F.lit(None).cast("string").alias("error_category"),
            F.lit(None).cast("int").alias("prompt_length"),
            F.lit(None).cast("string").alias("tool_name"),
            F.lit(None).cast("string").alias("error_message"),
            F.lit(None).cast("string").alias("content_preview"),
        )
    )

    columns = [
        "session_id",
        "event_time",
        "event_type",
        "error_category",
        "prompt_length",
        "tool_name",
        "error_message",
        "content_preview",
    ]

    unioned = (
        user_prompts.select(*columns)
        .unionByName(tool_calls.select(*columns))
        .unionByName(tool_results.select(*columns))
        .unionByName(api_errors.select(*columns))
        .unionByName(internal_errors.select(*columns))
        .unionByName(llm_requests.select(*columns))
    )

    if logs_df is not None and "body" in logs_df.columns:
        log_events = logs_df.select(
            F.col("attributes").getItem("session.id").alias("session_id"),
            F.lit(None).cast("timestamp").alias("event_time"),
            F.lit("LOG").alias("event_type"),
            F.lit(None).cast("string").alias("error_category"),
            F.lit(None).cast("int").alias("prompt_length"),
            F.lit(None).cast("string").alias("tool_name"),
            F.lit(None).cast("string").alias("error_message"),
            F.substring(F.col("body"), 1, 500).alias("content_preview"),
        ).filter(F.col("session_id").isNotNull())
        unioned = unioned.unionByName(log_events.select(*columns))

    return unioned


# ---------------------------------------------------------------------------
# session_metrics
# ---------------------------------------------------------------------------


def transform_session_metrics(metrics_df: DataFrame) -> DataFrame:
    """Aggregate bronze metrics into per-session token + cost + model snapshot."""

    tokens_df = (
        metrics_df.filter(F.col("name") == "claude_code.token.usage")
        .groupBy(F.col("sum.attributes").getItem("session.id").alias("session_id"))
        .agg(
            F.sum(
                F.when(
                    F.col("sum.attributes").getItem("type") == "input", F.col("sum.value")
                ).otherwise(0)
            ).alias("input_tokens"),
            F.sum(
                F.when(
                    F.col("sum.attributes").getItem("type") == "output", F.col("sum.value")
                ).otherwise(0)
            ).alias("output_tokens"),
            F.sum(
                F.when(
                    F.col("sum.attributes").getItem("type") == "cacheRead", F.col("sum.value")
                ).otherwise(0)
            ).alias("cache_read_tokens"),
            F.sum(
                F.when(
                    F.col("sum.attributes").getItem("type") == "cacheCreation",
                    F.col("sum.value"),
                ).otherwise(0)
            ).alias("cache_creation_tokens"),
        )
    )

    cost_df = (
        metrics_df.filter(F.col("name") == "claude_code.cost.usage")
        .groupBy(F.col("sum.attributes").getItem("session.id").alias("session_id"))
        .agg(F.sum("sum.value").alias("total_cost_usd"))
    )

    active_time_df = (
        metrics_df.filter(F.col("name") == "claude_code.active_time.total")
        .groupBy(F.col("sum.attributes").getItem("session.id").alias("session_id"))
        .agg(F.sum("sum.value").alias("total_active_time_s"))
    )

    lines_df = (
        metrics_df.filter(F.col("name") == "claude_code.code_edit_tool.decision")
        .groupBy(F.col("sum.attributes").getItem("session.id").alias("session_id"))
        .agg(F.sum("sum.value").alias("code_edit_decisions"))
    )

    model_effort = (
        metrics_df.filter(F.col("name") == "claude_code.cost.usage")
        .select(
            F.col("sum.attributes").getItem("session.id").alias("session_id"),
            F.col("sum.attributes").getItem("model").alias("model"),
            F.col("sum.attributes").getItem("effort").alias("effort"),
            F.col("sum.value").alias("cost"),
        )
        .groupBy("session_id", "model", "effort")
        .agg(F.sum("cost").alias("model_cost"))
        .withColumn(
            "rn",
            F.row_number().over(
                Window.partitionBy("session_id").orderBy(F.col("model_cost").desc())
            ),
        )
        .filter(F.col("rn") == 1)
        .select(
            "session_id",
            F.col("model").alias("primary_model"),
            F.col("effort").alias("effort_level"),
        )
    )

    session_metrics = (
        tokens_df.join(cost_df, "session_id", "outer")
        .join(active_time_df, "session_id", "outer")
        .join(lines_df, "session_id", "outer")
        .join(model_effort, "session_id", "left")
        .select(
            "session_id",
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_creation_tokens",
            "total_cost_usd",
            "total_active_time_s",
            "code_edit_decisions",
            "primary_model",
            "effort_level",
        )
    )

    return session_metrics


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _merge_table(spark: SparkSession, df: DataFrame, target_table: str, merge_key: str) -> None:
    """Upsert ``df`` into ``target_table`` keyed on ``merge_key`` using Delta MERGE.

    Creates the table from ``df`` on first run.
    """
    staging_view = f"_silver_etl_staging_{merge_key}"
    df.createOrReplaceTempView(staging_view)

    if not spark.catalog.tableExists(target_table):
        logger.info("Creating target table %s", target_table)
        df.write.format("delta").saveAsTable(target_table)
        return

    logger.info("Merging into %s on %s", target_table, merge_key)
    update_cols = [c for c in df.columns if c != merge_key]
    set_clause = ", ".join(f"t.{c} = s.{c}" for c in update_cols)
    insert_cols = ", ".join(df.columns)
    insert_vals = ", ".join(f"s.{c}" for c in df.columns)

    spark.sql(
        f"""
        MERGE INTO {target_table} t
        USING {staging_view} s
        ON t.{merge_key} = s.{merge_key}
        WHEN MATCHED THEN UPDATE SET {set_clause}
        WHEN NOT MATCHED THEN INSERT ({insert_cols}) VALUES ({insert_vals})
        """
    )


def _delete_append_table(spark: SparkSession, df: DataFrame, target_table: str) -> None:
    """Replace ``target_table`` contents with ``df`` (overwrite semantics)."""
    logger.info("Overwriting %s", target_table)
    (
        df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(target_table)
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_silver_etl(
    spark: SparkSession,
    source_catalog: str,
    source_schema: str,
    target_catalog: str,
    target_schema: str,
) -> None:
    """Read bronze tables, run the three transforms, write silver tables."""

    traces_table = f"{source_catalog}.{source_schema}.traces"
    metrics_table = f"{source_catalog}.{source_schema}.metrics"
    logs_table = f"{source_catalog}.{source_schema}.logs"

    logger.info(
        "Reading bronze tables: %s, %s, %s", traces_table, metrics_table, logs_table
    )
    traces_df = spark.read.table(traces_table)
    metrics_df = spark.read.table(metrics_table)
    logs_df = spark.read.table(logs_table)

    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {target_catalog}.{target_schema}")

    summary_df = transform_session_summary(traces_df, metrics_df)
    events_df = transform_session_events(traces_df, logs_df)
    metrics_silver_df = transform_session_metrics(metrics_df)

    _merge_table(
        spark,
        summary_df,
        f"{target_catalog}.{target_schema}.session_summary",
        "session_id",
    )
    _delete_append_table(
        spark, events_df, f"{target_catalog}.{target_schema}.session_events"
    )
    _merge_table(
        spark,
        metrics_silver_df,
        f"{target_catalog}.{target_schema}.session_metrics",
        "session_id",
    )


def main() -> None:
    """CLI entry point — parses args and runs the silver ETL."""
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run the silver ETL job.")
    parser.add_argument("--source-catalog", default="tanner_fevm_catalog")
    parser.add_argument("--source-schema", default="claude")
    parser.add_argument("--target-catalog", default="tanner_fevm_catalog")
    parser.add_argument("--target-schema", default="claude_silver")
    args = parser.parse_args()

    spark = create_spark_session()
    run_silver_etl(
        spark,
        source_catalog=args.source_catalog,
        source_schema=args.source_schema,
        target_catalog=args.target_catalog,
        target_schema=args.target_schema,
    )


if __name__ == "__main__":
    main()
