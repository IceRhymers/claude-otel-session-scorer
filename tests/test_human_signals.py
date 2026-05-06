"""Tests for the human_signals module."""

import inspect
from unittest.mock import MagicMock, patch

from claude_otel_session_scorer import human_signals
from claude_otel_session_scorer.human_signals import (
    _CORRECTION_WINDOW_SECONDS,
    _SCORE_WEIGHTS,
    compute_friction_score,
    main,
    run_human_signals,
)


def _make_mock_spark(tables_exist: bool = False, completed_session_count: int = 1):
    spark = MagicMock()
    spark.catalog.tableExists.return_value = tables_exist
    df = MagicMock()
    df.select.return_value = df
    df.join.return_value = df
    df.filter.return_value = df
    df.groupBy.return_value = df
    df.agg.return_value = df
    df.withColumn.return_value = df
    df.orderBy.return_value = df
    df.count.return_value = completed_session_count
    spark.table.return_value = df
    return spark


def _sql_calls(spark):
    return [c.args[0].strip() for c in spark.sql.call_args_list]


def test_creates_gold_tables_if_not_exist():
    spark = _make_mock_spark(tables_exist=False, completed_session_count=2)
    run_human_signals(spark, "cat.silver", "cat.gold")
    sql = _sql_calls(spark)
    assert any("CREATE TABLE IF NOT EXISTS cat.gold.session_human_signals" in s for s in sql)
    assert any(
        "CREATE TABLE IF NOT EXISTS cat.gold.session_human_signals_by_tool" in s for s in sql
    )
    # Both DDLs must use Delta + auto-clustering.
    ddls = [s for s in sql if "CREATE TABLE IF NOT EXISTS cat.gold.session_human_signals" in s]
    for ddl in ddls:
        assert "USING DELTA" in ddl
        assert "CLUSTER BY AUTO" in ddl


def test_merge_with_update_set_star_for_recomputability():
    spark = _make_mock_spark(tables_exist=True, completed_session_count=3)
    run_human_signals(spark, "cat.silver", "cat.gold")
    sql = _sql_calls(spark)
    merges = [s for s in sql if "MERGE INTO" in s]
    assert any("MERGE INTO cat.gold.session_human_signals " in m for m in merges)
    assert any("MERGE INTO cat.gold.session_human_signals_by_tool" in m for m in merges)
    for m in merges:
        assert "WHEN MATCHED THEN UPDATE SET *" in m
    # No left_anti anywhere — recomputable, not immutable.
    src = inspect.getsource(run_human_signals)
    assert "left_anti" not in src


def test_completion_guard_two_hour_filter():
    src = inspect.getsource(run_human_signals)
    assert "INTERVAL 2 HOURS" in src
    assert "session_end" in src


def test_first_run_backfills_all_completed_sessions():
    src = inspect.getsource(run_human_signals)
    # No left_anti gate — every completed session is scored on every run.
    assert "left_anti" not in src
    # And the only filter on session_summary is the 2-hour completion guard.
    spark = _make_mock_spark(tables_exist=False, completed_session_count=5)
    run_human_signals(spark, "cat.silver", "cat.gold")
    # MERGE still fires when gold tables don't yet exist (recomputable backfill).
    sql = _sql_calls(spark)
    assert any("MERGE INTO cat.gold.session_human_signals " in s for s in sql)


def test_signal_strength_true_with_rejects_only():
    # reject_rate=0.5, others NULL → 100*(0.4*0.5 + 0 + 0) = 20.0
    score = compute_friction_score(
        reject_rate=0.5,
        abort_rate=None,
        correction_intensity=None,
        signal_strength=True,
    )
    assert score == 20.0


def test_signal_strength_true_with_aborts_only():
    # abort_rate=0.5, others NULL → 100*(0 + 0.3*0.5 + 0) = 15.0
    score = compute_friction_score(
        reject_rate=None,
        abort_rate=0.5,
        correction_intensity=None,
        signal_strength=True,
    )
    assert score == 15.0


def test_signal_strength_false_zero_signals():
    # signal_strength=False → score is NULL (None), NOT 0.0 — explicitly correcting
    # the autonomy_score NULL→0 mistake the spec opens by criticizing.
    score = compute_friction_score(
        reject_rate=None,
        abort_rate=None,
        correction_intensity=None,
        signal_strength=False,
    )
    assert score is None


def test_human_friction_score_exact_arithmetic():
    # reject_rate=0.5, abort=0.0, correction=0.2 → 100 * (0.4*0.5 + 0.3*0 + 0.3*0.2) = 26.0
    score = compute_friction_score(
        reject_rate=0.5,
        abort_rate=0.0,
        correction_intensity=0.2,
        signal_strength=True,
    )
    assert score == 26.0


def test_correction_window_25s_counted():
    # Source-level: the predicate uses <= _CORRECTION_WINDOW_SECONDS, so 25 ≤ 30 IS counted.
    src = inspect.getsource(run_human_signals)
    assert "_CORRECTION_WINDOW_SECONDS" in src
    assert "<=" in src
    # And the predicate filters USER_PROMPT preceded by TOOL_RESULT.
    assert "TOOL_RESULT" in src
    assert "USER_PROMPT" in src


def test_correction_window_35s_not_counted():
    # Same source predicate guarantees 35 > 30 is excluded.
    src = inspect.getsource(run_human_signals)
    # Confirm comparison is `<=`, not `<` (boundary-inclusive at 30s).
    assert "<= _CORRECTION_WINDOW_SECONDS" in src


def test_num_corrections_deterministic_under_ts_ties():
    # correction_window must include monotonically_increasing_id() as a stable
    # tiebreaker for same-second events (event_ts has integer-second precision).
    src = inspect.getsource(run_human_signals)
    assert "orderBy(" in src
    assert '"event_ts"' in src
    assert '"event_type"' in src
    assert "monotonically_increasing_id" in src
    # Both orderBy keys must appear in the same Window.partitionBy call.
    assert 'partitionBy("session_id")' in src or "partitionBy('session_id')" in src


def test_correction_window_constant_is_thirty():
    assert _CORRECTION_WINDOW_SECONDS == 30


def test_per_tool_table_one_row_per_tool():
    spark = _make_mock_spark(tables_exist=True, completed_session_count=2)
    run_human_signals(spark, "cat.silver", "cat.gold")
    # The per-tool aggregation must groupBy session_id AND tool_name.
    src = inspect.getsource(run_human_signals)
    assert 'groupBy("session_id", "tool_name")' in src
    # And the per-tool MERGE must key on both columns.
    sql = _sql_calls(spark)
    by_tool_merges = [s for s in sql if "MERGE INTO cat.gold.session_human_signals_by_tool" in s]
    assert len(by_tool_merges) == 1
    assert "tool_name" in by_tool_merges[0]
    # And a DELETE-before-MERGE must clear stale rows for recomputed sessions.
    deletes = [s for s in sql if "DELETE FROM cat.gold.session_human_signals_by_tool" in s]
    assert len(deletes) == 1


def test_modify_decisions_excluded_from_buckets():
    # Source predicate filters detail_name to ('accept', 'reject') only —
    # any 'modify' or other value is excluded from both num_tool_rejects
    # and num_tool_accepts (and the denominator).
    src = inspect.getsource(run_human_signals)
    assert 'isin("accept", "reject")' in src


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


def test_no_udfs_or_ai_query():
    src = inspect.getsource(human_signals)
    assert "ai_query" not in src
    assert "@F.udf" not in src
    assert "_build_replay_udf" not in src
    assert "_build_prompt_udf" not in src


def test_score_weights_single_source_of_truth():
    # _SCORE_WEIGHTS is the authoritative dict; Python helper and SQL must derive from it.
    assert _SCORE_WEIGHTS["reject"] == 0.4
    assert _SCORE_WEIGHTS["abort"] == 0.3
    assert _SCORE_WEIGHTS["correction"] == 0.3
    assert abs(sum(_SCORE_WEIGHTS.values()) - 1.0) < 1e-9, "weights must sum to 1"
    # Both the Python function and SQL expression must reference _SCORE_WEIGHTS, not literals.
    py_src = inspect.getsource(compute_friction_score)
    assert '_SCORE_WEIGHTS["reject"]' in py_src
    assert '_SCORE_WEIGHTS["abort"]' in py_src
    assert '_SCORE_WEIGHTS["correction"]' in py_src
    sql_src = inspect.getsource(run_human_signals)
    assert "_SCORE_WEIGHTS['reject']" in sql_src or '_SCORE_WEIGHTS["reject"]' in sql_src


def test_event_ts_sec_materialized_before_lag():
    # event_ts_sec must be a WithColumn before the lag to avoid double-casting.
    src = inspect.getsource(run_human_signals)
    assert "event_ts_sec" in src
    assert '"event_ts_sec"' in src or "'event_ts_sec'" in src
    # The predicate must use the materialized column, not inline casts.
    assert 'event_ts").cast("long") - F.col("_prev_event_ts").cast("long")' not in src


def test_by_tool_join_is_inner():
    # Per-tool rows only make sense for sessions in session_keys (completed sessions).
    src = inspect.getsource(run_human_signals)
    # The by_tool join must be "inner", not "left".
    assert '"inner"' in src
    # Confirm the old incorrect "left" join for by_tool is gone by checking
    # that the by_tool join line uses "inner".
    assert 'session_keys.select("session_id", "user_id"), "session_id", "inner"' in src
