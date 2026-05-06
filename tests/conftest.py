"""Shared pytest fixtures and helpers for the test suite.

This file gives every test access to:

* `_make_mock_spark` / `_sql_calls` — the historical `MagicMock`-based Spark
  doubles, kept for tests that genuinely need to assert on emitted SQL/DDL
  shape rather than runtime behavior.
* `spark` — a session-scoped, in-process `SparkSession` configured with the
  Delta extensions, used by the behavioral tests in `test_human_signals.py`.
* `bronze_schema` / `silver_schema` / `gold_schema` — function-scoped helpers
  that allocate a fresh database (UUID-suffixed) on the live `spark` and drop
  it after the test, so each behavioral test runs against hermetic state.
* `make_silver_tables` — convenience factory that materializes the three
  silver tables from in-memory Python rows so each test can spell out only
  the columns it cares about.
* `make_bronze_tables` — convenience factory that materializes the three
  bronze OTEL tables (traces / logs / metrics) from in-memory Python rows
  so silver_etl behavioral tests can drive `run_silver_etl` end-to-end.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta
from typing import Iterable
from unittest.mock import MagicMock

import pytest
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    LongType,
    MapType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


# ---------------------------------------------------------------------------
# Mock-based helpers (shared by test_scorer.py and the few mock-only checks
# that remain in test_human_signals.py).
# ---------------------------------------------------------------------------


def _make_mock_spark(
    *,
    table_exists: bool = False,
    completed_session_count: int | None = None,
    new_session_count: int | None = None,
) -> MagicMock:
    """Build a `MagicMock` SparkSession that records `.sql(...)` calls.

    DataFrame fluent calls (`select`, `join`, `filter`, ...) all return the
    same mock so chains like `events.filter(...).groupBy(...).agg(...)` work
    without per-test boilerplate. `.count()` returns whichever of
    `completed_session_count` / `new_session_count` was supplied — the two
    are semantic aliases of each other, kept distinct so each call site reads
    naturally (`run_human_signals` thinks in completed sessions,
    `run_scoring` thinks in new-since-last-run sessions).
    """
    if completed_session_count is None and new_session_count is None:
        count_value = 1
    else:
        count_value = (
            completed_session_count if completed_session_count is not None else new_session_count
        )

    spark = MagicMock()
    spark.catalog.tableExists.return_value = table_exists
    df = MagicMock()
    df.select.return_value = df
    df.join.return_value = df
    df.filter.return_value = df
    df.groupBy.return_value = df
    df.agg.return_value = df
    df.withColumn.return_value = df
    df.orderBy.return_value = df
    df.count.return_value = count_value
    spark.table.return_value = df
    return spark


def _sql_calls(spark: MagicMock) -> list[str]:
    """Return every SQL string passed to `spark.sql(...)`, stripped."""
    return [c.args[0].strip() for c in spark.sql.call_args_list]


# ---------------------------------------------------------------------------
# Live Spark fixture (Delta-enabled, local[1]).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def spark(tmp_path_factory: pytest.TempPathFactory) -> Iterable[SparkSession]:
    """Session-scoped SparkSession with Delta + a tmp warehouse.

    Spinning up Spark is expensive (~5–10s with the delta-spark JAR fetch on
    first call), so we reuse a single session for the whole test run and rely
    on the `silver_schema` / `gold_schema` fixtures to give each test its own
    isolated databases.
    """
    warehouse = tmp_path_factory.mktemp("warehouse")
    builder = (
        SparkSession.builder.master("local[1]")
        .appName("cotss-tests")
        .config("spark.sql.warehouse.dir", str(warehouse))
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "1")
        # Match Databricks' implicit default: silver_etl's events append uses
        # `df.write.mode("append").option("mergeSchema", "true").saveAsTable`
        # without an explicit `.format("delta")`. On Databricks the delta
        # catalog makes that work; OSS Spark would otherwise pick parquet and
        # raise a format-mismatch against the existing Delta table.
        .config("spark.sql.sources.default", "delta")
    )
    session = configure_spark_with_delta_pip(builder).getOrCreate()
    session.sparkContext.setLogLevel("ERROR")

    # Databricks-only / Databricks-runtime-only SQL bits that OSS Spark +
    # delta-spark 3.3.x can't parse or execute. We rewrite them on the way in
    # so the behavioral tests can exercise the pipeline locally — the
    # production code stays unchanged. The mock-based DDL test still asserts
    # the original clauses are emitted.
    _cluster_by_auto_re = re.compile(r"\bCLUSTER\s+BY\s+AUTO\b", re.IGNORECASE)
    # `DELETE ... WHERE col IN (SELECT col FROM view)` — delta-spark 3.3 does
    # not implement subqueries in DELETE; we materialize the view and inline
    # its values as a literal IN-list.
    _delete_subquery_re = re.compile(
        r"DELETE\s+FROM\s+(\S+)\s+WHERE\s+(\w+)\s+IN\s*\(\s*"
        r"SELECT\s+(\w+)\s+FROM\s+(\w+)\s*\)",
        re.IGNORECASE,
    )
    _original_sql = session.sql

    def _local_compat_sql(query, *args, **kwargs):
        if isinstance(query, str):
            query = _cluster_by_auto_re.sub("", query)
            m = _delete_subquery_re.search(query)
            if m:
                table, col, sub_col, view = m.group(1), m.group(2), m.group(3), m.group(4)
                rows = _original_sql(f"SELECT DISTINCT {sub_col} FROM {view}").collect()
                values = [r[0] for r in rows if r[0] is not None]
                if not values:
                    new_clause = f"DELETE FROM {table} WHERE FALSE"
                else:
                    literal = ", ".join("'" + v.replace("'", "''") + "'" for v in values)
                    new_clause = f"DELETE FROM {table} WHERE {col} IN ({literal})"
                query = query[: m.start()] + new_clause + query[m.end() :]
        return _original_sql(query, *args, **kwargs)

    session.sql = _local_compat_sql

    try:
        yield session
    finally:
        session.stop()


def _new_schema(spark: SparkSession, prefix: str) -> str:
    name = f"{prefix}_{uuid.uuid4().hex[:8]}"
    spark.sql(f"CREATE DATABASE {name}")
    return name


@pytest.fixture
def bronze_schema(spark: SparkSession) -> Iterable[str]:
    """Allocate a fresh `bronze_<uuid>` database; drop it after the test."""
    name = _new_schema(spark, "bronze")
    try:
        yield name
    finally:
        spark.sql(f"DROP DATABASE IF EXISTS {name} CASCADE")


@pytest.fixture
def silver_schema(spark: SparkSession) -> Iterable[str]:
    """Allocate a fresh `silver_<uuid>` database; drop it after the test."""
    name = _new_schema(spark, "silver")
    try:
        yield name
    finally:
        spark.sql(f"DROP DATABASE IF EXISTS {name} CASCADE")


@pytest.fixture
def gold_schema(spark: SparkSession) -> Iterable[str]:
    """Allocate a fresh `gold_<uuid>` database; drop it after the test."""
    name = _new_schema(spark, "gold")
    try:
        yield name
    finally:
        spark.sql(f"DROP DATABASE IF EXISTS {name} CASCADE")


# ---------------------------------------------------------------------------
# Silver-table builder. Behavioral tests pass partial dict rows; this fills
# in the rest of the schema with sensible defaults so each test spells out
# only the columns it cares about.
# ---------------------------------------------------------------------------


_SUMMARY_SCHEMA = StructType(
    [
        StructField("session_id", StringType(), False),
        StructField("user_id", StringType(), True),
        StructField("session_start", TimestampType(), True),
        StructField("session_end", TimestampType(), True),
        StructField("num_interactions", LongType(), True),
    ]
)

_EVENTS_SCHEMA = StructType(
    [
        StructField("session_id", StringType(), False),
        StructField("event_ts", TimestampType(), True),
        StructField("event_type", StringType(), True),
        StructField("detail_name", StringType(), True),
        StructField("duration_ms", DoubleType(), True),
        StructField("input_tokens", LongType(), True),
        StructField("output_tokens", LongType(), True),
        StructField("cost_usd", DoubleType(), True),
        StructField("success", StringType(), True),
        StructField("content_preview", StringType(), True),
        StructField("full_content", StringType(), True),
        StructField("event_source", StringType(), True),
        StructField("model", StringType(), True),
        StructField("tool_name", StringType(), True),
        StructField("error_category", StringType(), True),
        StructField("prompt_id", StringType(), True),
        StructField("tool_use_id", StringType(), True),
        StructField("decision_source", StringType(), True),
    ]
)

_METRICS_SCHEMA = StructType(
    [
        StructField("session_id", StringType(), False),
        StructField("primary_model", StringType(), True),
    ]
)


_SUMMARY_DEFAULTS: dict = {
    "user_id": "u1",
    "session_start": datetime(2025, 1, 1, 0, 0, 0),
    "session_end": datetime(2025, 1, 1, 1, 0, 0) - timedelta(days=1),  # 1d 1h ago — completed
    "num_interactions": 1,
}

_EVENTS_DEFAULTS: dict = {
    "event_ts": datetime(2025, 1, 1, 0, 0, 0),
    "event_type": "USER_PROMPT",
    "detail_name": "",
    "duration_ms": None,
    "input_tokens": None,
    "output_tokens": None,
    "cost_usd": None,
    "success": None,
    "content_preview": "",
    "full_content": None,
    "event_source": "log",
    "model": None,
    "tool_name": None,
    "error_category": None,
    "prompt_id": None,
    "tool_use_id": None,
    "decision_source": None,
}

_METRICS_DEFAULTS: dict = {
    "primary_model": "claude-sonnet-4",
}


def _completed_ago(hours: float) -> datetime:
    """Return a naive local-time datetime `hours` ago.

    `human_signals` filters on `session_end < current_timestamp() - INTERVAL 2
    HOURS`, where `current_timestamp()` is the Spark session's local time.
    Using `datetime.now()` (local) keeps both sides on the same clock so the
    comparison doesn't quietly flip on machines whose TZ differs from UTC.
    """
    return datetime.now() - timedelta(hours=hours)


def _row_in_schema_order(row: dict, defaults: dict, schema: StructType) -> tuple:
    merged = {**defaults, **row}
    return tuple(merged.get(field.name) for field in schema.fields)


@pytest.fixture
def make_silver_tables(spark: SparkSession, silver_schema: str):
    """Factory that materializes `session_summary`, `session_events`, and
    `session_metrics` Delta tables in `silver_schema` from partial Python
    dict rows. Missing columns inherit defaults from the constants above.

    Returns a `build(*, summary, events, metrics)` callable so tests can call
    it once and write only the fields under test.
    """

    def build(
        *,
        summary: list[dict],
        events: list[dict],
        metrics: list[dict] | None = None,
    ) -> None:
        if metrics is None:
            metrics = [{"session_id": s["session_id"]} for s in summary]

        summary_rows = [
            _row_in_schema_order(s, _SUMMARY_DEFAULTS, _SUMMARY_SCHEMA) for s in summary
        ]
        events_rows = [_row_in_schema_order(e, _EVENTS_DEFAULTS, _EVENTS_SCHEMA) for e in events]
        metrics_rows = [
            _row_in_schema_order(m, _METRICS_DEFAULTS, _METRICS_SCHEMA) for m in metrics
        ]

        summary_df = spark.createDataFrame(summary_rows, schema=_SUMMARY_SCHEMA)
        events_df = spark.createDataFrame(events_rows, schema=_EVENTS_SCHEMA)
        metrics_df = spark.createDataFrame(metrics_rows, schema=_METRICS_SCHEMA)

        # Drop-then-create avoids the "does not support truncate in batch
        # mode" error that delta-spark 3.3.x raises for both
        # `saveAsTable(..., mode="overwrite")` and
        # `writeTo(...).createOrReplace()` against an existing managed Delta
        # table — recomputable tests need to overwrite silver between runs.
        for table, df in [
            ("session_summary", summary_df),
            ("session_events", events_df),
            ("session_metrics", metrics_df),
        ]:
            spark.sql(f"DROP TABLE IF EXISTS {silver_schema}.{table}")
            df.write.format("delta").saveAsTable(f"{silver_schema}.{table}")

    return build


# ---------------------------------------------------------------------------
# Bronze-table builder. Mirror of `make_silver_tables` for the bronze→silver
# side. Behavioral silver_etl tests pass partial dict rows and this factory
# fills in the rest of the schema with sensible defaults.
# ---------------------------------------------------------------------------


# Schemas are intentionally a strict subset of `docs/bronze-schema.sql` —
# only the columns `silver_etl.py` actually reads. If silver_etl ever starts
# reading a new bronze column, mirror the change here too. Most upstream
# changes to the OTLP-proxy schema therefore won't ripple into these tests.
_BRONZE_TRACES_SCHEMA = StructType(
    [
        StructField("name", StringType(), True),
        StructField("attributes", MapType(StringType(), StringType()), True),
        StructField("start_time_unix_nano", LongType(), True),
        StructField("end_time_unix_nano", LongType(), True),
        StructField(
            "events",
            ArrayType(
                StructType([StructField("attributes", MapType(StringType(), StringType()), True)])
            ),
            True,
        ),
        StructField(
            "resource",
            StructType([StructField("attributes", MapType(StringType(), StringType()), True)]),
            True,
        ),
    ]
)

_BRONZE_LOGS_SCHEMA = StructType(
    [
        StructField("body", StringType(), True),
        StructField("attributes", MapType(StringType(), StringType()), True),
    ]
)

_BRONZE_METRICS_SCHEMA = StructType(
    [
        StructField("name", StringType(), True),
        StructField(
            "sum",
            StructType(
                [
                    StructField("value", DoubleType(), True),
                    StructField("attributes", MapType(StringType(), StringType()), True),
                ]
            ),
            True,
        ),
    ]
)


_BRONZE_TRACES_DEFAULTS: dict = {
    "name": "",
    "attributes": {},
    "start_time_unix_nano": 0,
    "end_time_unix_nano": 0,
    "events": [],
    "resource": {"attributes": {}},
}

_BRONZE_LOGS_DEFAULTS: dict = {
    "body": "",
    "attributes": {},
}

_BRONZE_METRICS_DEFAULTS: dict = {
    "name": "",
    "sum": {"value": 0.0, "attributes": {}},
}


@pytest.fixture
def make_bronze_tables(spark: SparkSession, bronze_schema: str):
    """Factory that materializes `claude_otel_traces`, `claude_otel_logs`, and
    `claude_otel_metrics` Delta tables in `bronze_schema` from partial Python
    dict rows. Missing columns inherit defaults from the constants above.

    The mirror schema is intentionally a subset of `docs/bronze-schema.sql` —
    only the columns `silver_etl.py` actually reads:

    * `claude_otel_traces`: `name`, `attributes` (MAP), `start_time_unix_nano`,
      `end_time_unix_nano`, `events` (ARRAY<STRUCT<attributes: MAP>>),
      `resource` (STRUCT<attributes: MAP>).
    * `claude_otel_logs`: `body`, `attributes` (MAP).
    * `claude_otel_metrics`: `name`, `sum` (STRUCT<value: DOUBLE, attributes:
      MAP>).

    All three tables are always created (with whatever subset of rows the
    test provides — empty is fine) so silver_etl's `spark.table(...)` reads
    never miss. Returns a `build(*, traces, logs, metrics)` callable so each
    test spells out only the bronze fields under test.
    """

    def build(
        *,
        traces: list[dict] | None = None,
        logs: list[dict] | None = None,
        metrics: list[dict] | None = None,
    ) -> None:
        traces = traces or []
        logs = logs or []
        metrics = metrics or []

        traces_rows = [
            _row_in_schema_order(t, _BRONZE_TRACES_DEFAULTS, _BRONZE_TRACES_SCHEMA) for t in traces
        ]
        logs_rows = [
            _row_in_schema_order(log, _BRONZE_LOGS_DEFAULTS, _BRONZE_LOGS_SCHEMA) for log in logs
        ]
        metrics_rows = [
            _row_in_schema_order(m, _BRONZE_METRICS_DEFAULTS, _BRONZE_METRICS_SCHEMA)
            for m in metrics
        ]

        traces_df = spark.createDataFrame(traces_rows, schema=_BRONZE_TRACES_SCHEMA)
        logs_df = spark.createDataFrame(logs_rows, schema=_BRONZE_LOGS_SCHEMA)
        metrics_df = spark.createDataFrame(metrics_rows, schema=_BRONZE_METRICS_SCHEMA)

        for table, df in [
            ("claude_otel_traces", traces_df),
            ("claude_otel_logs", logs_df),
            ("claude_otel_metrics", metrics_df),
        ]:
            spark.sql(f"DROP TABLE IF EXISTS {bronze_schema}.{table}")
            df.write.format("delta").saveAsTable(f"{bronze_schema}.{table}")

    return build


# Re-export for tests that prefer `from .conftest import _completed_ago`. Pytest
# already auto-imports the fixtures themselves; these are the helpers tests
# call directly.
__all__ = ["_make_mock_spark", "_sql_calls", "_completed_ago"]
