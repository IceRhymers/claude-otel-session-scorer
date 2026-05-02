"""Local PySpark tests for the silver ETL transforms.

Uses a session-scoped local Spark session — no Databricks Connect.
"""

from __future__ import annotations

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    LongType,
    MapType,
    StringType,
    StructField,
    StructType,
)

from claude_otel_session_scorer.silver_etl import (
    transform_session_events,
    transform_session_metrics,
    transform_session_summary,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

TRACES_SCHEMA = StructType(
    [
        StructField("trace_id", StringType(), True),
        StructField("span_id", StringType(), True),
        StructField("name", StringType(), True),
        StructField("start_time_unix_nano", LongType(), True),
        StructField("end_time_unix_nano", LongType(), True),
        StructField("attributes", MapType(StringType(), StringType()), True),
        StructField(
            "resource",
            StructType(
                [
                    StructField("attributes", MapType(StringType(), StringType()), True),
                ]
            ),
            True,
        ),
        StructField(
            "events",
            ArrayType(
                StructType(
                    [
                        StructField("time_unix_nano", LongType(), True),
                        StructField("name", StringType(), True),
                        StructField(
                            "attributes", MapType(StringType(), StringType()), True
                        ),
                        StructField("dropped_attributes_count", IntegerType(), True),
                    ]
                )
            ),
            True,
        ),
    ]
)

METRICS_SCHEMA = StructType(
    [
        StructField("name", StringType(), True),
        StructField(
            "sum",
            StructType(
                [
                    StructField("value", DoubleType(), True),
                    StructField(
                        "attributes", MapType(StringType(), StringType()), True
                    ),
                ]
            ),
            True,
        ),
    ]
)

LOGS_SCHEMA = StructType(
    [
        StructField("body", StringType(), True),
        StructField("attributes", MapType(StringType(), StringType()), True),
    ]
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    return (
        SparkSession.builder.master("local[1]")
        .appName("test_silver_etl")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


def _trace_row(
    *,
    name: str,
    session_id: str,
    start: int = 1_000_000_000_000_000_000,
    end: int = 2_000_000_000_000_000_000,
    attributes: dict | None = None,
    resource_attrs: dict | None = None,
    events: list | None = None,
):
    attrs = {"session.id": session_id}
    if attributes:
        attrs.update(attributes)
    return (
        "trace",
        "span",
        name,
        start,
        end,
        attrs,
        {"attributes": resource_attrs or {}},
        events or [],
    )


def _metric_row(*, name: str, value: float, attributes: dict):
    return (name, {"value": value, "attributes": attributes})


# ---------------------------------------------------------------------------
# session_summary
# ---------------------------------------------------------------------------


def test_session_summary_joins_all_aggregates_and_derives_columns(spark: SparkSession):
    sid = "s1"
    traces = spark.createDataFrame(
        [
            # interactions (2 of them)
            _trace_row(
                name="claude_code.interaction",
                session_id=sid,
                attributes={
                    "user.id": "u1",
                    "user_prompt_length": "100",
                    "terminal.type": "iTerm",
                },
                resource_attrs={"service.version": "1.2.3", "os.type": "darwin"},
                start=1_700_000_000_000_000_000,
                end=1_700_000_010_000_000_000,
            ),
            _trace_row(
                name="claude_code.interaction",
                session_id=sid,
                attributes={
                    "user.id": "u1",
                    "user_prompt_length": "200",
                    "terminal.type": "iTerm",
                },
                resource_attrs={"service.version": "1.2.3", "os.type": "darwin"},
                start=1_700_000_020_000_000_000,
                end=1_700_000_030_000_000_000,
            ),
            # llm_request
            _trace_row(
                name="claude_code.llm_request",
                session_id=sid,
                attributes={
                    "input_tokens": "100",
                    "output_tokens": "50",
                    "cache_read_tokens": "200",
                    "cache_creation_tokens": "10",
                    "ttft_ms": "120.0",
                    "duration_ms": "1000.0",
                },
            ),
            # tool calls (4) — 4 / 2 interactions = 2.0 tools per interaction
            _trace_row(
                name="claude_code.tool",
                session_id=sid,
                attributes={"tool_name": "Read"},
            ),
            _trace_row(
                name="claude_code.tool",
                session_id=sid,
                attributes={"tool_name": "Read"},
            ),
            _trace_row(
                name="claude_code.tool",
                session_id=sid,
                attributes={"tool_name": "Edit"},
            ),
            _trace_row(
                name="claude_code.tool",
                session_id=sid,
                attributes={"tool_name": "Bash"},
            ),
            # tool execution: 3 of 4 succeed
            _trace_row(
                name="claude_code.tool.execution",
                session_id=sid,
                attributes={"success": "true"},
            ),
            _trace_row(
                name="claude_code.tool.execution",
                session_id=sid,
                attributes={"success": "true"},
            ),
            _trace_row(
                name="claude_code.tool.execution",
                session_id=sid,
                attributes={"success": "true"},
            ),
            _trace_row(
                name="claude_code.tool.execution",
                session_id=sid,
                attributes={"success": "false"},
            ),
            # blocked_on_user: 2 of 4 auto-accepted
            _trace_row(
                name="claude_code.tool.blocked_on_user",
                session_id=sid,
                attributes={"decision": "accept"},
            ),
            _trace_row(
                name="claude_code.tool.blocked_on_user",
                session_id=sid,
                attributes={"decision": "accept"},
            ),
            _trace_row(
                name="claude_code.tool.blocked_on_user",
                session_id=sid,
                attributes={"decision": "reject"},
            ),
            _trace_row(
                name="claude_code.tool.blocked_on_user",
                session_id=sid,
                attributes={"decision": "reject"},
            ),
        ],
        schema=TRACES_SCHEMA,
    )

    metrics = spark.createDataFrame(
        [
            _metric_row(
                name="claude_code.cost.usage",
                value=1.25,
                attributes={"session.id": sid, "model": "sonnet", "effort": "high"},
            ),
            _metric_row(
                name="claude_code.active_time.total",
                value=42.0,
                attributes={"session.id": sid},
            ),
        ],
        schema=METRICS_SCHEMA,
    )

    out = transform_session_summary(traces, metrics).collect()
    assert len(out) == 1
    row = out[0].asDict()

    assert row["session_id"] == sid
    assert row["user_id"] == "u1"
    assert row["num_interactions"] == 2
    assert row["num_llm_requests"] == 1
    assert row["num_tool_calls"] == 4
    assert row["total_input_tokens"] == 100
    assert row["total_output_tokens"] == 50
    assert row["total_cache_read"] == 200
    assert row["total_cache_creation"] == 10

    # cache_hit_rate = 200 / (100 + 50 + 200 + 10) = 200/360
    assert row["cache_hit_rate"] == pytest.approx(200 / 360)
    # tool_success_rate = 3/4
    assert row["tool_success_rate"] == pytest.approx(0.75)
    # auto_accept_rate = 2/4
    assert row["auto_accept_rate"] == pytest.approx(0.5)
    # tools_per_interaction = 4/2
    assert row["tools_per_interaction"] == pytest.approx(2.0)
    # llm_calls_per_interaction = 1/2
    assert row["llm_calls_per_interaction"] == pytest.approx(0.5)

    assert row["total_cost_usd"] == pytest.approx(1.25)
    assert row["total_active_time_s"] == pytest.approx(42.0)
    assert row["service_version"] == "1.2.3"
    assert row["os_type"] == "darwin"
    assert row["terminal_type"] == "iTerm"


# ---------------------------------------------------------------------------
# session_events
# ---------------------------------------------------------------------------


def test_session_events_error_classification_branches(spark: SparkSession):
    sid = "s1"
    traces = spark.createDataFrame(
        [
            # internal_error → INTERNAL_ERROR / invisible
            _trace_row(
                name="claude_code.internal_error",
                session_id=sid,
                attributes={"error": "Boom"},
            ),
            # api_error + away_summary → BACKGROUND_ABORTED / invisible
            _trace_row(
                name="claude_code.api_error",
                session_id=sid,
                attributes={"query_source": "away_summary", "error": "rate limited"},
            ),
            # api_error + Request was aborted. → USER_ABORTED / invisible
            _trace_row(
                name="claude_code.api_error",
                session_id=sid,
                attributes={
                    "query_source": "user",
                    "error": "Request was aborted.",
                },
            ),
            # api_error other → ERROR / user_visible
            _trace_row(
                name="claude_code.api_error",
                session_id=sid,
                attributes={"query_source": "user", "error": "500 server fail"},
            ),
        ],
        schema=TRACES_SCHEMA,
    )
    logs = spark.createDataFrame([], schema=LOGS_SCHEMA)

    out = transform_session_events(traces, logs).collect()
    by_type = {(r["event_type"], r["error_category"]) for r in out}

    assert ("INTERNAL_ERROR", "invisible") in by_type
    assert ("BACKGROUND_ABORTED", "invisible") in by_type
    assert ("USER_ABORTED", "invisible") in by_type
    assert ("ERROR", "user_visible") in by_type


def test_session_events_truncates_content_preview_at_500_chars(spark: SparkSession):
    sid = "s1"
    long_prompt = "p" * 1200
    long_result = "r" * 1200

    traces = spark.createDataFrame(
        [
            _trace_row(
                name="claude_code.interaction",
                session_id=sid,
                attributes={"user_prompt_length": "1200"},
                events=[
                    (
                        1_700_000_000_000_000_000,
                        "user_prompt",
                        {"prompt": long_prompt},
                        0,
                    )
                ],
            ),
            _trace_row(
                name="claude_code.tool.execution",
                session_id=sid,
                attributes={"tool_name": "Bash", "success": "true"},
                events=[
                    (
                        1_700_000_001_000_000_000,
                        "tool_result",
                        {"result": long_result},
                        0,
                    )
                ],
            ),
        ],
        schema=TRACES_SCHEMA,
    )
    logs = spark.createDataFrame([], schema=LOGS_SCHEMA)

    out = transform_session_events(traces, logs).collect()
    previews = {r["event_type"]: r["content_preview"] for r in out if r["content_preview"]}

    assert "USER_PROMPT" in previews
    assert "TOOL_RESULT" in previews
    assert len(previews["USER_PROMPT"]) == 500
    assert len(previews["TOOL_RESULT"]) == 500
    assert previews["USER_PROMPT"] == "p" * 500
    assert previews["TOOL_RESULT"] == "r" * 500


# ---------------------------------------------------------------------------
# session_metrics
# ---------------------------------------------------------------------------


def test_session_metrics_token_pivot_and_primary_model(spark: SparkSession):
    sid = "s1"
    metrics = spark.createDataFrame(
        [
            # token usage rows
            _metric_row(
                name="claude_code.token.usage",
                value=100.0,
                attributes={"session.id": sid, "type": "input"},
            ),
            _metric_row(
                name="claude_code.token.usage",
                value=50.0,
                attributes={"session.id": sid, "type": "output"},
            ),
            _metric_row(
                name="claude_code.token.usage",
                value=200.0,
                attributes={"session.id": sid, "type": "cacheRead"},
            ),
            _metric_row(
                name="claude_code.token.usage",
                value=10.0,
                attributes={"session.id": sid, "type": "cacheCreation"},
            ),
            # cost: cheap model first, then expensive model — expensive should win
            _metric_row(
                name="claude_code.cost.usage",
                value=0.10,
                attributes={"session.id": sid, "model": "haiku", "effort": "low"},
            ),
            _metric_row(
                name="claude_code.cost.usage",
                value=2.50,
                attributes={"session.id": sid, "model": "opus", "effort": "high"},
            ),
            _metric_row(
                name="claude_code.cost.usage",
                value=0.50,
                attributes={"session.id": sid, "model": "sonnet", "effort": "medium"},
            ),
        ],
        schema=METRICS_SCHEMA,
    )

    out = transform_session_metrics(metrics).collect()
    assert len(out) == 1
    row = out[0].asDict()

    assert row["session_id"] == sid
    assert row["input_tokens"] == pytest.approx(100.0)
    assert row["output_tokens"] == pytest.approx(50.0)
    assert row["cache_read_tokens"] == pytest.approx(200.0)
    assert row["cache_creation_tokens"] == pytest.approx(10.0)

    # primary_model should be the most expensive: opus
    assert row["primary_model"] == "opus"
    assert row["effort_level"] == "high"
    # total_cost_usd = 0.10 + 2.50 + 0.50
    assert row["total_cost_usd"] == pytest.approx(3.10)
