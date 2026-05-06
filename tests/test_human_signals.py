"""Tests for the human_signals module.

Most invariants are exercised behaviorally against an in-process Delta-enabled
SparkSession (see `tests/conftest.py`). A handful of mock-based tests remain
for things that are genuinely shape-only — DDL emitted, MERGE keying, the
`main()` entry-point wiring — and could not be more strongly tested by
running the pipeline.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from claude_otel_session_scorer.human_signals import (
    _CORRECTION_WINDOW_SECONDS,
    _SCORE_WEIGHTS,
    compute_friction_score,
    main,
    run_human_signals,
)
from tests.conftest import _completed_ago, _make_mock_spark, _sql_calls


# ---------------------------------------------------------------------------
# Pure-Python helper tests (compute_friction_score). These have always been
# behavioral; just keeping them.
# ---------------------------------------------------------------------------


def test_signal_strength_true_with_rejects_only():
    score = compute_friction_score(
        reject_rate=0.5,
        abort_rate=None,
        correction_intensity=None,
        signal_strength=True,
    )
    assert score == 20.0


def test_signal_strength_true_with_aborts_only():
    score = compute_friction_score(
        reject_rate=None,
        abort_rate=0.5,
        correction_intensity=None,
        signal_strength=True,
    )
    assert score == 15.0


def test_signal_strength_false_zero_signals():
    # AGENTS.md §3 invariant 1: NULL ≠ 0 — when signal_strength is False the
    # score is None, never 0.0.
    score = compute_friction_score(
        reject_rate=None,
        abort_rate=None,
        correction_intensity=None,
        signal_strength=False,
    )
    assert score is None


def test_human_friction_score_exact_arithmetic():
    score = compute_friction_score(
        reject_rate=0.5,
        abort_rate=0.0,
        correction_intensity=0.2,
        signal_strength=True,
    )
    assert score == 26.0


def test_correction_window_constant_is_thirty():
    assert _CORRECTION_WINDOW_SECONDS == 30


def test_score_weights_sum_to_one():
    assert _SCORE_WEIGHTS["reject"] == 0.4
    assert _SCORE_WEIGHTS["abort"] == 0.3
    assert _SCORE_WEIGHTS["correction"] == 0.3
    assert abs(sum(_SCORE_WEIGHTS.values()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Mock-only tests — DDL shape and entry-point wiring. Cannot be done
# behaviorally without coupling tests to Delta's internal table layout.
# ---------------------------------------------------------------------------


def test_creates_gold_tables_with_delta_and_clustering(spark):
    # `spark` fixture is requested only to ensure a real SparkContext is
    # active — `F.col(...)` calls inside `run_human_signals` need one even
    # though every other interaction here goes through the MagicMock.
    del spark
    mock_spark = _make_mock_spark(table_exists=False, completed_session_count=2)
    run_human_signals(mock_spark, "cat.silver", "cat.gold")
    sql = _sql_calls(mock_spark)
    ddls = [s for s in sql if s.startswith("CREATE TABLE IF NOT EXISTS cat.gold.")]
    # Both gold tables get a DDL.
    assert any("session_human_signals " in d or "session_human_signals\n" in d for d in ddls)
    assert any("session_human_signals_by_tool" in d for d in ddls)
    for ddl in ddls:
        assert "USING DELTA" in ddl
        assert "CLUSTER BY AUTO" in ddl


def test_main_creates_spark_and_stops():
    with (
        patch("claude_otel_session_scorer.human_signals.create_spark_session") as mock_create,
        patch("claude_otel_session_scorer.human_signals.run_human_signals") as mock_run,
    ):
        mock_spark = MagicMock()
        mock_create.return_value = mock_spark
        import sys

        with patch.object(
            sys,
            "argv",
            [
                "score_human_signals",
                "--silver-schema",
                "tc.ts",
                "--gold-schema",
                "tc.gold",
            ],
        ):
            main()
        mock_create.assert_called_once()
        mock_run.assert_called_once_with(mock_spark, "tc.ts", "tc.gold")
        mock_spark.stop.assert_called_once()


# ---------------------------------------------------------------------------
# Behavioral tests against a real in-process SparkSession. Every test below
# uses the `spark`, `silver_schema`, `gold_schema`, and `make_silver_tables`
# fixtures from `tests/conftest.py` — see that file for the schema defaults
# each test inherits from.
# ---------------------------------------------------------------------------


def _ts(seconds: int, *, base: datetime = datetime(2025, 1, 1, 0, 0, 0)) -> datetime:
    """Helper: an event_ts at `base + seconds` for easy boundary tests."""
    return base + timedelta(seconds=seconds)


def _gold_session_rows(spark, gold_schema):
    return spark.table(f"{gold_schema}.session_human_signals").collect()


def _gold_by_tool_rows(spark, gold_schema):
    return spark.table(f"{gold_schema}.session_human_signals_by_tool").collect()


# AGENTS.md §3 invariant 6: modify decisions are excluded from both
# numerator and denominator of reject_rate.
def test_modify_decisions_excluded_from_reject_rate(
    spark, silver_schema, gold_schema, make_silver_tables
):
    make_silver_tables(
        summary=[
            {
                "session_id": "s1",
                "session_end": _completed_ago(3),
                "num_interactions": 3,
            }
        ],
        events=[
            {
                "session_id": "s1",
                "event_ts": _ts(0),
                "event_type": "TOOL_DECISION",
                "detail_name": "accept",
                "tool_name": "Bash",
            },
            {
                "session_id": "s1",
                "event_ts": _ts(1),
                "event_type": "TOOL_DECISION",
                "detail_name": "reject",
                "tool_name": "Bash",
            },
            {
                "session_id": "s1",
                "event_ts": _ts(2),
                "event_type": "TOOL_DECISION",
                "detail_name": "modify",
                "tool_name": "Bash",
            },
        ],
    )

    run_human_signals(spark, silver_schema, gold_schema)

    rows = _gold_session_rows(spark, gold_schema)
    assert len(rows) == 1
    r = rows[0]
    assert r.session_id == "s1"
    # modify is dropped from BOTH the count and the rejects bucket.
    assert r.num_tool_decisions == 2
    assert r.num_tool_rejects == 1
    assert r.num_tool_accepts == 1
    assert r.reject_rate == pytest.approx(0.5)


# AGENTS.md §3 invariant 5: correction window is `<= 30s`, boundary-inclusive.
def test_correction_window_inclusive_30s(spark, silver_schema, gold_schema, make_silver_tables):
    # Three TOOL_RESULT → USER_PROMPT pairs at gaps 25s (counted), 30s
    # (boundary, counted), 35s (NOT counted). num_interactions doesn't affect
    # the count itself, only the intensity rate, so 99 keeps the math obvious.
    events = [
        {"session_id": "s1", "event_ts": _ts(100), "event_type": "TOOL_RESULT"},
        {"session_id": "s1", "event_ts": _ts(125), "event_type": "USER_PROMPT"},
        {"session_id": "s1", "event_ts": _ts(200), "event_type": "TOOL_RESULT"},
        {"session_id": "s1", "event_ts": _ts(230), "event_type": "USER_PROMPT"},
        {"session_id": "s1", "event_ts": _ts(300), "event_type": "TOOL_RESULT"},
        {"session_id": "s1", "event_ts": _ts(335), "event_type": "USER_PROMPT"},
    ]
    make_silver_tables(
        summary=[
            {
                "session_id": "s1",
                "session_end": _completed_ago(3),
                "num_interactions": 99,
            }
        ],
        events=events,
    )

    run_human_signals(spark, silver_schema, gold_schema)

    rows = _gold_session_rows(spark, gold_schema)
    assert len(rows) == 1
    # 25s and 30s gaps count; 35s does not.
    assert rows[0].num_corrections == 2


# AGENTS.md §3 invariant 5 (tiebreaker half): same `event_ts` events must
# resolve deterministically via the (event_ts, event_type) order — without
# the event_type tiebreaker, USER_PROMPT and TOOL_RESULT at the same second
# could land in either order and the prev-event lookup would be flaky.
#
# We insert USER_PROMPT BEFORE TOOL_RESULT in the input rows; the only thing
# that makes USER_PROMPT see TOOL_RESULT as its predecessor is the
# event_type ASC tiebreaker (T < U).
def test_orderby_event_ts_then_event_type_tiebreaker(
    spark, silver_schema, gold_schema, make_silver_tables
):
    make_silver_tables(
        summary=[
            {
                "session_id": "s1",
                "session_end": _completed_ago(3),
                "num_interactions": 1,
            }
        ],
        events=[
            # Input order is USER_PROMPT first, TOOL_RESULT second — the
            # tiebreaker must reorder them so the lag picks up TOOL_RESULT.
            {"session_id": "s1", "event_ts": _ts(50), "event_type": "USER_PROMPT"},
            {"session_id": "s1", "event_ts": _ts(50), "event_type": "TOOL_RESULT"},
        ],
    )

    run_human_signals(spark, silver_schema, gold_schema)

    rows = _gold_session_rows(spark, gold_schema)
    assert len(rows) == 1
    # 0s gap, prev=TOOL_RESULT after the alphabetical tiebreaker → counted.
    assert rows[0].num_corrections == 1


# AGENTS.md §3 invariant 3: completion guard excludes sessions whose
# session_end is within the last 2 hours.
def test_completion_guard_excludes_active_sessions(
    spark, silver_schema, gold_schema, make_silver_tables
):
    make_silver_tables(
        summary=[
            {
                "session_id": "active",
                "session_end": _completed_ago(1),  # 1h ago — STILL ACTIVE
                "num_interactions": 1,
            },
            {
                "session_id": "done",
                "session_end": _completed_ago(3),  # 3h ago — COMPLETED
                "num_interactions": 1,
            },
        ],
        events=[
            {
                "session_id": "active",
                "event_type": "TOOL_DECISION",
                "detail_name": "reject",
                "tool_name": "Bash",
            },
            {
                "session_id": "done",
                "event_type": "TOOL_DECISION",
                "detail_name": "reject",
                "tool_name": "Bash",
            },
        ],
    )

    run_human_signals(spark, silver_schema, gold_schema)

    rows = _gold_session_rows(spark, gold_schema)
    session_ids = {r.session_id for r in rows}
    assert session_ids == {"done"}


# AGENTS.md §3 invariant 2: scores are recomputable, NOT immutable. A second
# run for the same session must overwrite the first.
def test_recomputable_no_left_anti(spark, silver_schema, gold_schema, make_silver_tables):
    # First run: 1 reject out of 1 decision → reject_rate=1.0.
    make_silver_tables(
        summary=[
            {
                "session_id": "x",
                "session_end": _completed_ago(3),
                "num_interactions": 1,
            }
        ],
        events=[
            {
                "session_id": "x",
                "event_type": "TOOL_DECISION",
                "detail_name": "reject",
                "tool_name": "Bash",
            },
        ],
    )
    run_human_signals(spark, silver_schema, gold_schema)
    first_rows = _gold_session_rows(spark, gold_schema)
    assert len(first_rows) == 1
    assert first_rows[0].reject_rate == pytest.approx(1.0)

    # Second run: same session, but now 1 accept (not reject) → reject_rate=0.0.
    # If `human_signals` had a left_anti gate, the row would still show 1.0.
    make_silver_tables(
        summary=[
            {
                "session_id": "x",
                "session_end": _completed_ago(3),
                "num_interactions": 1,
            }
        ],
        events=[
            {
                "session_id": "x",
                "event_type": "TOOL_DECISION",
                "detail_name": "accept",
                "tool_name": "Bash",
            },
        ],
    )
    run_human_signals(spark, silver_schema, gold_schema)
    second_rows = _gold_session_rows(spark, gold_schema)
    assert len(second_rows) == 1
    assert second_rows[0].reject_rate == pytest.approx(0.0)
    assert second_rows[0].num_tool_rejects == 0
    assert second_rows[0].num_tool_accepts == 1


# AGENTS.md §3 invariant 4: per-tool delete-then-MERGE drops tools that
# disappear between runs. A plain MERGE would leave the stale row behind.
def test_per_tool_delete_then_merge_drops_disappeared_tools(
    spark, silver_schema, gold_schema, make_silver_tables
):
    # First run: session "x" has decisions for both Bash and Edit.
    make_silver_tables(
        summary=[
            {
                "session_id": "x",
                "session_end": _completed_ago(3),
                "num_interactions": 2,
            }
        ],
        events=[
            {
                "session_id": "x",
                "event_type": "TOOL_DECISION",
                "detail_name": "accept",
                "tool_name": "Bash",
            },
            {
                "session_id": "x",
                "event_type": "TOOL_DECISION",
                "detail_name": "accept",
                "tool_name": "Edit",
            },
        ],
    )
    run_human_signals(spark, silver_schema, gold_schema)
    after_first = _gold_by_tool_rows(spark, gold_schema)
    assert {r.tool_name for r in after_first if r.session_id == "x"} == {"Bash", "Edit"}

    # Second run: only Bash remains. Edit must disappear from gold_by_tool.
    make_silver_tables(
        summary=[
            {
                "session_id": "x",
                "session_end": _completed_ago(3),
                "num_interactions": 2,
            }
        ],
        events=[
            {
                "session_id": "x",
                "event_type": "TOOL_DECISION",
                "detail_name": "accept",
                "tool_name": "Bash",
            },
        ],
    )
    run_human_signals(spark, silver_schema, gold_schema)
    after_second = _gold_by_tool_rows(spark, gold_schema)
    assert {r.tool_name for r in after_second if r.session_id == "x"} == {"Bash"}


# AGENTS.md §3 invariant 1: human_friction_score is NULL when there is no
# friction signal at all (no decisions, no aborts, no corrections).
def test_friction_score_null_when_no_signal(spark, silver_schema, gold_schema, make_silver_tables):
    make_silver_tables(
        summary=[
            {
                "session_id": "quiet",
                "session_end": _completed_ago(3),
                "num_interactions": 5,
            }
        ],
        events=[
            # Just an LLM_CALL — no TOOL_DECISION, no USER_ABORTED, and no
            # USER_PROMPT-after-TOOL_RESULT pair to trigger a correction.
            {
                "session_id": "quiet",
                "event_ts": _ts(0),
                "event_type": "LLM_CALL",
                "detail_name": "user_query",
            },
        ],
    )

    run_human_signals(spark, silver_schema, gold_schema)

    rows = _gold_session_rows(spark, gold_schema)
    assert len(rows) == 1
    assert rows[0].signal_strength is False
    assert rows[0].human_friction_score is None  # NULL, not 0.0


# Per-tool aggregation produces exactly one row per (session_id, tool_name).
def test_per_tool_groupby_session_and_tool(spark, silver_schema, gold_schema, make_silver_tables):
    make_silver_tables(
        summary=[
            {
                "session_id": "x",
                "session_end": _completed_ago(3),
                "num_interactions": 3,
            }
        ],
        events=[
            # Bash: 1 accept + 1 reject → reject_rate=0.5
            {
                "session_id": "x",
                "event_type": "TOOL_DECISION",
                "detail_name": "accept",
                "tool_name": "Bash",
            },
            {
                "session_id": "x",
                "event_type": "TOOL_DECISION",
                "detail_name": "reject",
                "tool_name": "Bash",
            },
            # Edit: 1 accept → reject_rate=0.0
            {
                "session_id": "x",
                "event_type": "TOOL_DECISION",
                "detail_name": "accept",
                "tool_name": "Edit",
            },
        ],
    )

    run_human_signals(spark, silver_schema, gold_schema)

    rows = [r for r in _gold_by_tool_rows(spark, gold_schema) if r.session_id == "x"]
    by_tool = {r.tool_name: r for r in rows}
    assert set(by_tool.keys()) == {"Bash", "Edit"}
    assert by_tool["Bash"].num_tool_decisions == 2
    assert by_tool["Bash"].num_tool_rejects == 1
    assert by_tool["Bash"].num_tool_accepts == 1
    assert by_tool["Bash"].reject_rate == pytest.approx(0.5)
    assert by_tool["Edit"].num_tool_decisions == 1
    assert by_tool["Edit"].num_tool_accepts == 1
    assert by_tool["Edit"].reject_rate == pytest.approx(0.0)
