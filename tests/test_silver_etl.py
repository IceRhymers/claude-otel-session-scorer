"""Tests for the silver_etl module.

Most invariants are exercised behaviorally against an in-process Delta-enabled
SparkSession (see `tests/conftest.py`). A handful of mock-based tests remain
for things that are genuinely shape-only — DDL emitted, MERGE keying, and the
`main()` entry-point wiring — and could not be more strongly tested by
running the pipeline.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from claude_otel_session_scorer.silver_etl import main, run_silver_etl
from tests.conftest import _make_mock_spark, _sql_calls


# ---------------------------------------------------------------------------
# Mock-only tests — DDL shape, MERGE keying, and entry-point wiring.
# ---------------------------------------------------------------------------


def test_run_silver_etl_calls_schema_create(spark):
    # `spark` fixture is requested only to keep a real SparkContext alive —
    # `F.col(...)` calls inside `run_silver_etl` need one even though every
    # other interaction here goes through the MagicMock.
    del spark
    mock_spark = _make_mock_spark()
    run_silver_etl(mock_spark, "cat.src", "cat.tgt")
    assert any("CREATE SCHEMA IF NOT EXISTS cat.tgt" in s for s in _sql_calls(mock_spark))


def test_session_summary_merge(spark):
    del spark
    mock_spark = _make_mock_spark()
    run_silver_etl(mock_spark, "cat.src", "cat.tgt")
    merge_calls = [s for s in _sql_calls(mock_spark) if "MERGE INTO cat.tgt.session_summary" in s]
    assert len(merge_calls) == 1
    assert "WHEN MATCHED THEN UPDATE SET *" in merge_calls[0]
    assert "WHEN NOT MATCHED THEN INSERT *" in merge_calls[0]


def test_session_events_delete_then_append(spark):
    del spark
    mock_spark = _make_mock_spark()
    run_silver_etl(mock_spark, "cat.src", "cat.tgt")

    delete_calls = [s for s in _sql_calls(mock_spark) if "DELETE FROM cat.tgt.session_events" in s]
    assert len(delete_calls) == 1

    # Verify saveAsTable was invoked with the silver events table name somewhere
    # in the MagicMock call graph (write.mode("append").saveAsTable).
    all_calls = str(mock_spark.mock_calls)
    assert "saveAsTable" in all_calls
    assert "cat.tgt.session_events" in all_calls


def test_session_metrics_merge(spark):
    del spark
    mock_spark = _make_mock_spark()
    run_silver_etl(mock_spark, "cat.src", "cat.tgt")
    merge_calls = [s for s in _sql_calls(mock_spark) if "MERGE INTO cat.tgt.session_metrics" in s]
    assert len(merge_calls) == 1
    assert "WHEN MATCHED THEN UPDATE SET *" in merge_calls[0]
    assert "WHEN NOT MATCHED THEN INSERT *" in merge_calls[0]


def test_main_creates_spark_and_stops():
    env_without_dbr = {k: v for k, v in os.environ.items() if k != "DATABRICKS_RUNTIME_VERSION"}
    with (
        patch("claude_otel_session_scorer.silver_etl.create_spark_session") as mock_create,
        patch("claude_otel_session_scorer.silver_etl.run_silver_etl") as mock_run,
        patch.dict(os.environ, env_without_dbr, clear=True),
    ):
        mock_spark = MagicMock()
        mock_create.return_value = mock_spark

        with patch(
            "sys.argv",
            [
                "silver_etl",
                "--bronze-schema",
                "sc.ss",
                "--silver-schema",
                "tc.ts",
            ],
        ):
            main()

        mock_create.assert_called_once()
        mock_run.assert_called_once_with(mock_spark, "sc.ss", "tc.ts")
        mock_spark.stop.assert_called_once()


def test_main_does_not_stop_spark_inside_databricks():
    """spark.stop() must be suppressed when DATABRICKS_RUNTIME_VERSION is set."""
    with (
        patch("claude_otel_session_scorer.silver_etl.create_spark_session") as mock_create,
        patch("claude_otel_session_scorer.silver_etl.run_silver_etl"),
        patch.dict(os.environ, {"DATABRICKS_RUNTIME_VERSION": "14.3"}),
    ):
        mock_spark = MagicMock()
        mock_create.return_value = mock_spark

        with patch("sys.argv", ["silver_etl", "--bronze-schema", "a.b", "--silver-schema", "c.d"]):
            main()

        mock_spark.stop.assert_not_called()


def test_session_metrics_written_before_summary(spark):
    """session_metrics must be merged before session_summary so summary can join cost/active_time."""
    del spark
    mock_spark = _make_mock_spark()
    run_silver_etl(mock_spark, "cat.src", "cat.tgt")
    sql_calls = _sql_calls(mock_spark)
    metrics_pos = next(
        i for i, s in enumerate(sql_calls) if "MERGE INTO cat.tgt.session_metrics" in s
    )
    summary_pos = next(
        i for i, s in enumerate(sql_calls) if "MERGE INTO cat.tgt.session_summary" in s
    )
    assert metrics_pos < summary_pos, "session_metrics MERGE must precede session_summary MERGE"


# ---------------------------------------------------------------------------
# Behavioral tests against a real in-process SparkSession. Every test below
# uses the `spark`, `bronze_schema`, `silver_schema`, and `make_bronze_tables`
# fixtures from `tests/conftest.py`.
#
# `_all_projection_bronze()` returns a single 6-row bronze fixture (1 trace +
# 5 logs) that drives all six silver event projections. The first three
# behavioral tests reuse it so the union-compatibility contract is exercised
# end-to-end on every assertion: if any projection diverged in column list or
# type, `unionByName` would fail before a single assertion ran.
# ---------------------------------------------------------------------------


def _all_projection_bronze():
    """(traces, logs) covering all six silver event projections.

    USER_PROMPT, LLM_CALL, TOOL_DECISION, TOOL_RESULT, and ERROR originate
    from logs; TOOL_CALL originates from a `claude_code.tool` trace. Every
    row is for `session.id="s1"` so a single run materializes one row per
    event_type into `silver.session_events`.
    """
    traces = [
        {
            "name": "claude_code.tool",
            "attributes": {
                "session.id": "s1",
                "tool_name": "Bash",
                "tool_use_id": "tu-call",
            },
        }
    ]
    logs = [
        {
            "body": "claude_code.user_prompt",
            "attributes": {
                "session.id": "s1",
                "prompt.id": "p-user",
                "prompt": "hi",
            },
        },
        {
            "body": "claude_code.api_request",
            "attributes": {
                "session.id": "s1",
                "prompt.id": "p-llm",
                "model": "sonnet",
                "input_tokens": "10",
                "output_tokens": "5",
                "cache_read_tokens": "0",
            },
        },
        {
            "body": "claude_code.tool_decision",
            "attributes": {
                "session.id": "s1",
                "prompt.id": "p-dec",
                "tool_name": "Bash",
                "tool_use_id": "tu-dec",
                "decision": "accept",
                "source": "config",
            },
        },
        {
            "body": "claude_code.tool_result",
            "attributes": {
                "session.id": "s1",
                "prompt.id": "p-res",
                "tool_name": "Bash",
                "tool_use_id": "tu-res",
            },
        },
        {
            "body": "claude_code.api_error",
            "attributes": {
                "session.id": "s1",
                "prompt.id": "p-err",
                "error": "boom",
            },
        },
    ]
    return traces, logs


def _events_by_type(spark, silver_schema):
    rows = spark.table(f"{silver_schema}.session_events").collect()
    return {r.event_type: r for r in rows}


# AGENTS.md §3 silver-events six-projection union rule: every projection must
# emit a `prompt_id` column. Log-driven projections source it from
# `attributes.getItem("prompt.id")`; the trace-driven TOOL_CALL projection
# explicitly NULLs it to keep prompt provenance log-only.
def test_silver_events_carry_prompt_id_across_all_projections(
    spark, bronze_schema, silver_schema, make_bronze_tables
):
    traces, logs = _all_projection_bronze()
    make_bronze_tables(traces=traces, logs=logs)

    run_silver_etl(spark, bronze_schema, silver_schema)

    rows = _events_by_type(spark, silver_schema)
    assert set(rows.keys()) == {
        "USER_PROMPT",
        "LLM_CALL",
        "TOOL_CALL",
        "TOOL_DECISION",
        "TOOL_RESULT",
        "ERROR",
    }
    assert rows["USER_PROMPT"].prompt_id == "p-user"
    assert rows["LLM_CALL"].prompt_id == "p-llm"
    # TOOL_CALL trace projection emits prompt_id as NULL (silver_etl L249).
    assert rows["TOOL_CALL"].prompt_id is None
    assert rows["TOOL_DECISION"].prompt_id == "p-dec"
    assert rows["TOOL_RESULT"].prompt_id == "p-res"
    assert rows["ERROR"].prompt_id == "p-err"


# AGENTS.md §3 silver-events six-projection union rule: every projection must
# emit a `tool_use_id` column. Only the three tool-shaped projections
# (TOOL_CALL, TOOL_DECISION, TOOL_RESULT) populate it; the others must emit
# the column as NULL so `unionByName` succeeds.
def test_silver_events_carry_tool_use_id_for_tool_projections(
    spark, bronze_schema, silver_schema, make_bronze_tables
):
    traces, logs = _all_projection_bronze()
    make_bronze_tables(traces=traces, logs=logs)

    run_silver_etl(spark, bronze_schema, silver_schema)

    rows = _events_by_type(spark, silver_schema)
    assert rows["TOOL_CALL"].tool_use_id == "tu-call"
    assert rows["TOOL_DECISION"].tool_use_id == "tu-dec"
    assert rows["TOOL_RESULT"].tool_use_id == "tu-res"
    assert rows["USER_PROMPT"].tool_use_id is None
    assert rows["LLM_CALL"].tool_use_id is None
    assert rows["ERROR"].tool_use_id is None


# AGENTS.md §3 silver-events six-projection union rule: `decision_source` is
# emitted by every projection (so `unionByName` succeeds) but is non-null only
# on the TOOL_DECISION projection, sourced from `attributes.getItem("source")`.
def test_silver_events_decision_source_only_on_tool_decision(
    spark, bronze_schema, silver_schema, make_bronze_tables
):
    traces, logs = _all_projection_bronze()
    make_bronze_tables(traces=traces, logs=logs)

    run_silver_etl(spark, bronze_schema, silver_schema)

    rows = _events_by_type(spark, silver_schema)
    assert rows["TOOL_DECISION"].decision_source == "config"
    for event_type in ("USER_PROMPT", "LLM_CALL", "TOOL_CALL", "TOOL_RESULT", "ERROR"):
        assert rows[event_type].decision_source is None, (
            f"decision_source must be NULL on {event_type}"
        )


# AGENTS.md §3 invariant 7: `silver_events` write uses `mergeSchema=true` with
# `saveAsTable(..., mode="append")`. Pre-create the table with a strict-subset
# schema (no `decision_source`, no `error_category`) and a baseline row for a
# different session, then run silver_etl. The append must (a) succeed via
# schema evolution, (b) populate the new column on the freshly-written row,
# and (c) leave the baseline row for the other session intact (per-session
# delete-then-append, not table-wide truncate).
def test_silver_events_append_evolves_schema_via_merge(
    spark, bronze_schema, silver_schema, make_bronze_tables
):
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {silver_schema}")
    spark.sql(
        f"""
        CREATE TABLE {silver_schema}.session_events (
            session_id STRING,
            event_ts TIMESTAMP,
            event_type STRING,
            detail_name STRING,
            duration_ms DOUBLE,
            input_tokens LONG,
            output_tokens LONG,
            cost_usd DOUBLE,
            success STRING,
            content_preview STRING,
            full_content STRING,
            event_source STRING,
            model STRING,
            tool_name STRING,
            prompt_id STRING,
            tool_use_id STRING
        ) USING DELTA
        """
    )
    spark.sql(
        f"""
        INSERT INTO {silver_schema}.session_events VALUES (
            'other',
            TIMESTAMP '2025-01-01 00:00:00',
            'USER_PROMPT',
            '',
            CAST(NULL AS DOUBLE),
            CAST(NULL AS BIGINT),
            CAST(NULL AS BIGINT),
            CAST(NULL AS DOUBLE),
            CAST(NULL AS STRING),
            '',
            CAST(NULL AS STRING),
            'log',
            CAST(NULL AS STRING),
            CAST(NULL AS STRING),
            CAST(NULL AS STRING),
            CAST(NULL AS STRING)
        )
        """
    )

    make_bronze_tables(
        logs=[
            {
                "body": "claude_code.tool_decision",
                "attributes": {
                    "session.id": "s1",
                    "prompt.id": "p-dec",
                    "tool_name": "Bash",
                    "tool_use_id": "tu-dec",
                    "decision": "accept",
                    "source": "config",
                },
            }
        ],
    )

    run_silver_etl(spark, bronze_schema, silver_schema)

    columns = set(spark.table(f"{silver_schema}.session_events").columns)
    # mergeSchema=true must have evolved the table to add both omitted columns.
    assert {"decision_source", "error_category"}.issubset(columns)

    rows = spark.table(f"{silver_schema}.session_events").collect()
    by_session = {r.session_id: r for r in rows}
    # Baseline row for "other" survived: the per-session DELETE only targets
    # incoming session_ids (s1), and the write is an append, not an overwrite.
    assert "other" in by_session
    # The s1 TOOL_DECISION row was written and the new column populated.
    assert by_session["s1"].event_type == "TOOL_DECISION"
    assert by_session["s1"].decision_source == "config"


# AGENTS.md §3 single-source-of-truth rule for cost/active_time: `summary`
# derives `total_active_time_s` and `total_cost_usd` by left-joining
# `session_metrics`, not by re-aggregating bronze. Active time only exists on
# `claude_code.active_time.total` metrics — there is no equivalent column on
# bronze traces — so a non-zero answer can only come from the metrics join.
# Left-join semantics also mean a session that's in traces but missing from
# metrics still produces a summary row, with NULL aggregates.
def test_summary_joins_total_active_time_and_cost_from_metrics(
    spark, bronze_schema, silver_schema, make_bronze_tables
):
    base_ns = 1_735_689_600 * 1_000_000_000  # 2025-01-01T00:00:00Z
    second_ns = 1_000_000_000
    make_bronze_tables(
        traces=[
            {
                "name": "claude_code.interaction",
                "attributes": {"session.id": "s1"},
                "start_time_unix_nano": base_ns,
                "end_time_unix_nano": base_ns + 60 * second_ns,
            },
            {
                "name": "claude_code.interaction",
                "attributes": {"session.id": "s2"},
                "start_time_unix_nano": base_ns,
                "end_time_unix_nano": base_ns + 30 * second_ns,
            },
        ],
        metrics=[
            {
                "name": "claude_code.active_time.total",
                "sum": {"value": 10.0, "attributes": {"session.id": "s1", "type": "cli"}},
            },
            {
                "name": "claude_code.active_time.total",
                "sum": {"value": 25.0, "attributes": {"session.id": "s1", "type": "user"}},
            },
            {
                "name": "claude_code.cost.usage",
                "sum": {
                    "value": 0.42,
                    "attributes": {
                        "session.id": "s1",
                        "model": "sonnet",
                        "effort": "high",
                    },
                },
            },
        ],
    )

    run_silver_etl(spark, bronze_schema, silver_schema)

    summary = spark.table(f"{silver_schema}.session_summary")
    rows = {r.session_id: r for r in summary.collect()}
    assert set(rows.keys()) == {"s1", "s2"}
    # 10.0 + 25.0 = 35.0 — the only source for cli/user breakdowns is the
    # metrics-table join, so this answer pins the join as the source.
    assert rows["s1"].total_active_time_s == pytest.approx(35.0)
    assert rows["s1"].total_cost_usd == pytest.approx(0.42)
    # Left-join semantics: s2 has no metrics, so the summary row exists but
    # the joined aggregates are NULL.
    assert rows["s2"].total_active_time_s is None
    assert rows["s2"].total_cost_usd is None
    # Per-bucket columns belong on `session_metrics`, not on summary —
    # confirms the summary projection only takes the derived total column.
    assert "active_time_cli_s" not in summary.columns
    assert "active_time_user_s" not in summary.columns
