"""Tests for the scorer module."""

from unittest.mock import MagicMock, patch

from claude_otel_session_scorer.scorer import (
    build_replay_text,
    compress_interaction,
    format_event_line,
    main,
    run_scoring,
    split_into_interactions,
)


def _make_mock_spark(table_exists=False, new_session_count=1):
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
    df.count.return_value = new_session_count
    spark.table.return_value = df
    return spark


def _sql_calls(spark):
    return [c.args[0].strip() for c in spark.sql.call_args_list]


class _Row:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_format_event_line_defaults():
    row = _Row(
        event_ts="2025-01-01",
        event_type="TOOL_USE",
        detail_name="bash",
        model="sonnet",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        error_category="",
        content_preview="hello",
    )
    line = format_event_line(row)
    assert "TOOL_USE" in line
    assert "bash" in line
    assert "0.0010" in line


def test_split_into_interactions_empty():
    assert split_into_interactions([]) == [[]]


def test_split_into_interactions_splits_at_user_prompt():
    rows = [
        _Row(event_type="TOOL_USE"),
        _Row(event_type="USER_PROMPT"),
        _Row(event_type="TOOL_USE"),
    ]
    interactions = split_into_interactions(rows)
    assert len(interactions) == 2
    assert interactions[0][0].event_type == "TOOL_USE"
    assert interactions[1][0].event_type == "USER_PROMPT"


def test_compress_interaction_summary():
    rows = [_Row(event_type="TOOL_USE", cost_usd=0.01, duration_ms=100)] * 3
    result = compress_interaction(rows)
    assert "compressed" in result
    assert "3 events" in result
    assert "TOOL_USE=3" in result


def test_build_replay_text_truncates():
    rows = [
        _Row(
            event_type="TOOL_USE",
            event_ts="",
            detail_name="",
            model="",
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            error_category="",
            content_preview="x" * 200,
        )
    ]
    text = build_replay_text(rows)
    assert len(text) <= 30_000


def test_build_replay_text_compresses_middle():
    rows = []
    for i in range(10):
        rows.append(
            _Row(
                event_type="USER_PROMPT",
                event_ts=str(i),
                detail_name="",
                model="",
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                error_category="",
                content_preview="",
            )
        )
        rows.append(
            _Row(
                event_type="TOOL_USE",
                event_ts=str(i),
                detail_name="",
                model="",
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                error_category="",
                content_preview="",
            )
        )
    text = build_replay_text(rows)
    assert "compressed" in text


def test_early_exit_when_no_new_sessions():
    spark = _make_mock_spark(table_exists=True, new_session_count=0)
    run_scoring(spark, "cat", "silver", "cat.gold")
    merge_calls = [s for s in _sql_calls(spark) if "MERGE INTO" in s]
    assert len(merge_calls) == 0


def test_creates_gold_table_if_not_exists():
    spark = _make_mock_spark(table_exists=False, new_session_count=2)
    run_scoring(spark, "cat", "silver", "cat.gold")
    assert any("CREATE TABLE IF NOT EXISTS cat.gold.session_scores" in s for s in _sql_calls(spark))


def test_merge_called_when_new_sessions():
    spark = _make_mock_spark(table_exists=True, new_session_count=3)
    run_scoring(spark, "cat", "silver", "cat.gold")
    assert any("MERGE INTO cat.gold.session_scores" in s for s in _sql_calls(spark))


def test_main_creates_spark_and_stops():
    with (
        patch("claude_otel_session_scorer.scorer.create_spark_session") as mock_create,
        patch("claude_otel_session_scorer.scorer.run_scoring") as mock_run,
    ):
        mock_spark = MagicMock()
        mock_create.return_value = mock_spark
        import sys

        with patch.object(
            sys,
            "argv",
            [
                "score_sessions",
                "--target-catalog",
                "tc",
                "--target-schema",
                "ts",
                "--gold-schema",
                "tc.gold",
            ],
        ):
            main()
        mock_create.assert_called_once()
        mock_run.assert_called_once_with(mock_spark, "tc", "ts", "tc.gold")
        mock_spark.stop.assert_called_once()
