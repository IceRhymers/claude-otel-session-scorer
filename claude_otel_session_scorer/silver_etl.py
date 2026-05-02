import os
from argparse import ArgumentParser

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def create_spark_session() -> SparkSession:
    if os.environ.get("DATABRICKS_RUNTIME_VERSION") is None:
        try:
            from databricks.connect import DatabricksSession

            return DatabricksSession.builder.serverless().getOrCreate()
        except ImportError:
            print("Databricks Connect not available. Falling back to standard Spark session.")
            return SparkSession.builder.getOrCreate()
    else:
        return SparkSession.builder.getOrCreate()


def _safe_event_attr(attr_key: str):
    return F.when(F.size(F.col("events")) > 0, F.col("events")[0]["attributes"].getItem(attr_key))


def _ensure_table_with_clustering(spark: SparkSession, full_table_name: str, df: DataFrame) -> None:
    if not spark.catalog.tableExists(full_table_name):
        df.limit(0).write.format("delta").option("clusterByAuto", "true").saveAsTable(full_table_name)


def _build_session_summary(
    spark: SparkSession, bronze_traces: str, bronze_metrics: str
) -> DataFrame:
    traces = spark.table(bronze_traces)
    metrics = spark.table(bronze_metrics)

    interactions = (
        traces.filter(F.col("name") == "claude_code.interaction")
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
        traces.filter(F.col("name") == "claude_code.llm_request")
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
        traces.filter(F.col("name") == "claude_code.tool")
        .groupBy(F.col("attributes").getItem("session.id").alias("session_id"))
        .agg(
            F.count("*").alias("num_tool_calls"),
            F.countDistinct(F.col("attributes").getItem("tool_name")).alias("distinct_tools"),
        )
    )
    tool_exec = (
        traces.filter(F.col("name") == "claude_code.tool.execution")
        .groupBy(F.col("attributes").getItem("session.id").alias("session_id"))
        .agg(
            F.count("*").alias("num_tool_executions"),
            F.sum(F.when(F.col("attributes").getItem("success") == "true", 1).otherwise(0)).alias(
                "tool_successes"
            ),
        )
    )
    autonomy = (
        traces.filter(F.col("name") == "claude_code.tool.blocked_on_user")
        .groupBy(F.col("attributes").getItem("session.id").alias("session_id"))
        .agg(
            F.count("*").alias("num_blocked_on_user"),
            F.sum(
                F.when(F.col("attributes").getItem("decision") == "accept", 1).otherwise(0)
            ).alias("auto_accepted"),
        )
    )
    cost = (
        metrics.filter(F.col("name") == "claude_code.cost.usage")
        .groupBy(F.col("sum.attributes").getItem("session.id").alias("session_id"))
        .agg(F.sum("sum.value").alias("total_cost_usd"))
    )
    active_time = (
        metrics.filter(F.col("name") == "claude_code.active_time.total")
        .groupBy(F.col("sum.attributes").getItem("session.id").alias("session_id"))
        .agg(F.sum("sum.value").alias("total_active_time_s"))
    )

    return (
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


def _build_session_events(spark: SparkSession, bronze_traces: str, bronze_logs: str) -> DataFrame:
    traces = spark.table(bronze_traces)
    logs = spark.table(bronze_logs)

    prompt_events = logs.filter(F.col("body") == "claude_code.user_prompt").select(
        F.col("attributes").getItem("session.id").alias("session_id"),
        F.to_timestamp(F.col("attributes").getItem("event.timestamp")).alias("event_ts"),
        F.lit("USER_PROMPT").alias("event_type"),
        F.coalesce(F.col("attributes").getItem("command_name"), F.lit("")).alias("detail_name"),
        F.lit(None).cast("double").alias("duration_ms"),
        F.lit(None).cast("long").alias("input_tokens"),
        F.lit(None).cast("long").alias("output_tokens"),
        F.lit(None).cast("double").alias("cost_usd"),
        F.lit(None).cast("string").alias("success"),
        F.substring(F.col("attributes").getItem("prompt"), 1, 500).alias("content_preview"),
        F.col("attributes").getItem("prompt").alias("full_content"),
        F.lit("log").alias("event_source"),
        F.lit(None).cast("string").alias("model"),
        F.lit(None).cast("string").alias("tool_name"),
        F.lit(None).cast("string").alias("error_category"),
    )

    llm_events = logs.filter(F.col("body") == "claude_code.api_request").select(
        F.col("attributes").getItem("session.id").alias("session_id"),
        F.to_timestamp(F.col("attributes").getItem("event.timestamp")).alias("event_ts"),
        F.lit("LLM_CALL").alias("event_type"),
        F.col("attributes").getItem("query_source").alias("detail_name"),
        F.col("attributes").getItem("duration_ms").cast("double").alias("duration_ms"),
        F.col("attributes").getItem("input_tokens").cast("long").alias("input_tokens"),
        F.col("attributes").getItem("output_tokens").cast("long").alias("output_tokens"),
        F.col("attributes").getItem("cost_usd").cast("double").alias("cost_usd"),
        F.lit(None).cast("string").alias("success"),
        F.concat(
            F.col("attributes").getItem("model"),
            F.lit(" | in:"),
            F.col("attributes").getItem("input_tokens"),
            F.lit(" out:"),
            F.col("attributes").getItem("output_tokens"),
            F.lit(" cache_read:"),
            F.col("attributes").getItem("cache_read_tokens"),
        ).alias("content_preview"),
        F.lit(None).cast("string").alias("full_content"),
        F.lit("log").alias("event_source"),
        F.col("attributes").getItem("model").alias("model"),
        F.lit(None).cast("string").alias("tool_name"),
        F.lit(None).cast("string").alias("error_category"),
    )

    tool_call_events = traces.filter(F.col("name") == "claude_code.tool").select(
        F.col("attributes").getItem("session.id").alias("session_id"),
        (F.col("start_time_unix_nano") / 1e9).cast("timestamp").alias("event_ts"),
        F.lit("TOOL_CALL").alias("event_type"),
        F.coalesce(
            F.col("attributes").getItem("full_command"),
            F.col("attributes").getItem("file_path"),
            F.lit(""),
        ).alias("detail_name"),
        F.col("attributes").getItem("duration_ms").cast("double").alias("duration_ms"),
        F.lit(None).cast("long").alias("input_tokens"),
        F.lit(None).cast("long").alias("output_tokens"),
        F.lit(None).cast("double").alias("cost_usd"),
        F.lit(None).cast("string").alias("success"),
        F.substring(
            F.coalesce(_safe_event_attr("output"), _safe_event_attr("content")), 1, 500
        ).alias("content_preview"),
        F.coalesce(_safe_event_attr("output"), _safe_event_attr("content")).alias("full_content"),
        F.lit("trace").alias("event_source"),
        F.lit(None).cast("string").alias("model"),
        F.col("attributes").getItem("tool_name").alias("tool_name"),
        F.lit(None).cast("string").alias("error_category"),
    )

    tool_decision_events = logs.filter(F.col("body") == "claude_code.tool_decision").select(
        F.col("attributes").getItem("session.id").alias("session_id"),
        F.to_timestamp(F.col("attributes").getItem("event.timestamp")).alias("event_ts"),
        F.lit("TOOL_DECISION").alias("event_type"),
        F.col("attributes").getItem("decision").alias("detail_name"),
        F.lit(None).cast("double").alias("duration_ms"),
        F.lit(None).cast("long").alias("input_tokens"),
        F.lit(None).cast("long").alias("output_tokens"),
        F.lit(None).cast("double").alias("cost_usd"),
        F.lit(None).cast("string").alias("success"),
        F.concat(
            F.col("attributes").getItem("tool_name"),
            F.lit(" → "),
            F.col("attributes").getItem("decision"),
            F.lit(" (source: "),
            F.coalesce(F.col("attributes").getItem("source"), F.lit("unknown")),
            F.lit(")"),
        ).alias("content_preview"),
        F.lit(None).cast("string").alias("full_content"),
        F.lit("log").alias("event_source"),
        F.lit(None).cast("string").alias("model"),
        F.col("attributes").getItem("tool_name").alias("tool_name"),
        F.lit(None).cast("string").alias("error_category"),
    )

    tool_result_events = logs.filter(F.col("body") == "claude_code.tool_result").select(
        F.col("attributes").getItem("session.id").alias("session_id"),
        F.to_timestamp(F.col("attributes").getItem("event.timestamp")).alias("event_ts"),
        F.lit("TOOL_RESULT").alias("event_type"),
        F.concat(
            F.col("attributes").getItem("tool_name"),
            F.lit(" | in:"),
            F.coalesce(F.col("attributes").getItem("tool_input_size_bytes"), F.lit("0")),
            F.lit("B out:"),
            F.coalesce(F.col("attributes").getItem("tool_result_size_bytes"), F.lit("0")),
            F.lit("B"),
        ).alias("detail_name"),
        F.col("attributes").getItem("duration_ms").cast("double").alias("duration_ms"),
        F.lit(None).cast("long").alias("input_tokens"),
        F.lit(None).cast("long").alias("output_tokens"),
        F.lit(None).cast("double").alias("cost_usd"),
        F.col("attributes").getItem("success").alias("success"),
        F.substring(F.col("attributes").getItem("tool_input"), 1, 500).alias("content_preview"),
        F.col("attributes").getItem("tool_input").alias("full_content"),
        F.lit("log").alias("event_source"),
        F.lit(None).cast("string").alias("model"),
        F.col("attributes").getItem("tool_name").alias("tool_name"),
        F.lit(None).cast("string").alias("error_category"),
    )

    error_events = logs.filter(
        F.col("body").isin("claude_code.internal_error", "claude_code.api_error")
    ).select(
        F.col("attributes").getItem("session.id").alias("session_id"),
        F.to_timestamp(F.col("attributes").getItem("event.timestamp")).alias("event_ts"),
        F.when(F.col("body") == "claude_code.internal_error", F.lit("INTERNAL_ERROR"))
        .when(
            (F.col("body") == "claude_code.api_error")
            & (F.col("attributes").getItem("query_source") == "away_summary"),
            F.lit("BACKGROUND_ABORTED"),
        )
        .when(
            (F.col("body") == "claude_code.api_error")
            & (F.col("attributes").getItem("error") == "Request was aborted."),
            F.lit("USER_ABORTED"),
        )
        .otherwise(F.lit("ERROR"))
        .alias("event_type"),
        F.coalesce(
            F.col("attributes").getItem("error"),
            F.col("attributes").getItem("error_name"),
            F.lit("unknown"),
        ).alias("detail_name"),
        F.col("attributes").getItem("duration_ms").cast("double").alias("duration_ms"),
        F.lit(None).cast("long").alias("input_tokens"),
        F.lit(None).cast("long").alias("output_tokens"),
        F.lit(None).cast("double").alias("cost_usd"),
        F.lit(None).cast("string").alias("success"),
        F.coalesce(
            F.col("attributes").getItem("error"),
            F.concat(F.lit("error_name="), F.col("attributes").getItem("error_name")),
        ).alias("content_preview"),
        F.lit(None).cast("string").alias("full_content"),
        F.lit("log").alias("event_source"),
        F.col("attributes").getItem("model").alias("model"),
        F.lit(None).cast("string").alias("tool_name"),
        F.when(F.col("body") == "claude_code.internal_error", F.lit("invisible"))
        .when(
            (F.col("body") == "claude_code.api_error")
            & (
                (F.col("attributes").getItem("query_source") == "away_summary")
                | (F.col("attributes").getItem("error") == "Request was aborted.")
            ),
            F.lit("invisible"),
        )
        .otherwise(F.lit("user_visible"))
        .alias("error_category"),
    )

    return (
        prompt_events.unionByName(llm_events)
        .unionByName(tool_call_events)
        .unionByName(tool_decision_events)
        .unionByName(tool_result_events)
        .unionByName(error_events)
        .filter(F.col("session_id").isNotNull())
    )


def _build_session_metrics(spark: SparkSession, bronze_metrics: str) -> DataFrame:
    metrics = spark.table(bronze_metrics)

    cost_df = (
        metrics.filter(F.col("name") == "claude_code.cost.usage")
        .groupBy(F.col("sum.attributes").getItem("session.id").alias("session_id"))
        .agg(F.sum("sum.value").alias("total_cost_usd"))
    )
    tokens_df = (
        metrics.filter(F.col("name") == "claude_code.token.usage")
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
                    F.col("sum.attributes").getItem("type") == "cacheCreation", F.col("sum.value")
                ).otherwise(0)
            ).alias("cache_creation_tokens"),
        )
    )
    active_df = (
        metrics.filter(F.col("name") == "claude_code.active_time.total")
        .groupBy(F.col("sum.attributes").getItem("session.id").alias("session_id"))
        .agg(
            F.sum(
                F.when(
                    F.col("sum.attributes").getItem("type") == "cli", F.col("sum.value")
                ).otherwise(0)
            ).alias("active_time_cli_s"),
            F.sum(
                F.when(
                    F.col("sum.attributes").getItem("type") == "user", F.col("sum.value")
                ).otherwise(0)
            ).alias("active_time_user_s"),
        )
    )
    model_effort = (
        metrics.filter(F.col("name") == "claude_code.cost.usage")
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

    return (
        cost_df.join(tokens_df, "session_id", "outer")
        .join(active_df, "session_id", "outer")
        .join(model_effort, "session_id", "left")
        .select(
            "session_id",
            "total_cost_usd",
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_creation_tokens",
            "active_time_cli_s",
            "active_time_user_s",
            "primary_model",
            "effort_level",
        )
    )


def run_silver_etl(
    spark: SparkSession,
    source_catalog: str,
    source_schema: str,
    target_catalog: str,
    target_schema: str,
) -> None:
    bronze_traces = f"{source_catalog}.{source_schema}.claude_otel_traces"
    bronze_metrics = f"{source_catalog}.{source_schema}.claude_otel_metrics"
    bronze_logs = f"{source_catalog}.{source_schema}.claude_otel_logs"
    silver_summary = f"{target_catalog}.{target_schema}.session_summary"
    silver_events = f"{target_catalog}.{target_schema}.session_events"
    silver_metrics = f"{target_catalog}.{target_schema}.session_metrics"

    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {target_catalog}.{target_schema}")

    # session_summary — MERGE
    summary_df = _build_session_summary(spark, bronze_traces, bronze_metrics)
    _ensure_table_with_clustering(spark, silver_summary, summary_df)
    summary_df.createOrReplaceTempView("session_summary_updates")
    spark.sql(f"""
        MERGE INTO {silver_summary} AS target
        USING session_summary_updates AS source
        ON target.session_id = source.session_id
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)
    print(f"✔ {silver_summary}: {spark.table(silver_summary).count()} sessions")

    # session_events — delete-then-append per session
    events_df = _build_session_events(spark, bronze_traces, bronze_logs)
    _ensure_table_with_clustering(spark, silver_events, events_df)
    events_df.createOrReplaceTempView("session_events_updates")
    events_df.select("session_id").distinct().createOrReplaceTempView("incoming_session_ids")
    spark.sql(
        f"DELETE FROM {silver_events} WHERE session_id IN (SELECT session_id FROM incoming_session_ids)"
    )
    events_df.write.mode("append").saveAsTable(silver_events)
    print(f"✔ {silver_events}: {spark.table(silver_events).count()} events")

    # session_metrics — MERGE
    metrics_df = _build_session_metrics(spark, bronze_metrics)
    _ensure_table_with_clustering(spark, silver_metrics, metrics_df)
    metrics_df.createOrReplaceTempView("session_metrics_updates")
    spark.sql(f"""
        MERGE INTO {silver_metrics} AS target
        USING session_metrics_updates AS source
        ON target.session_id = source.session_id
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)
    print(f"✔ {silver_metrics}: {spark.table(silver_metrics).count()} sessions")


def main() -> None:
    parser = ArgumentParser(
        description="Silver ETL: transform bronze OTEL tables into silver session tables"
    )
    parser.add_argument("--source-catalog", "-sc", required=True)
    parser.add_argument("--source-schema", "-ss", required=True)
    parser.add_argument("--target-catalog", "-tc", required=True)
    parser.add_argument("--target-schema", "-ts", required=True)
    args = parser.parse_args()

    spark = create_spark_session()
    try:
        run_silver_etl(
            spark, args.source_catalog, args.source_schema, args.target_catalog, args.target_schema
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
