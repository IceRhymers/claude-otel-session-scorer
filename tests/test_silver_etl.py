"""
Tests for the silver_etl module
"""

from unittest.mock import MagicMock, call, patch

import pytest

from claude_otel_session_scorer.silver_etl import main, run_silver_etl


@pytest.fixture
def mock_spark():
    spark = MagicMock()
    # make spark.table() return a MagicMock that supports chained DataFrame ops
    mock_df = MagicMock()
    mock_df.filter.return_value = mock_df
    mock_df.groupBy.return_value = mock_df
    mock_df.agg.return_value = mock_df
    mock_df.join.return_value = mock_df
    mock_df.withColumn.return_value = mock_df
    mock_df.select.return_value = mock_df
    mock_df.distinct.return_value = mock_df
    mock_df.unionByName.return_value = mock_df
    mock_df.write = MagicMock()
    mock_df.write.mode.return_value = mock_df.write
    mock_df.count.return_value = 0
    spark.table.return_value = mock_df
    return spark


def test_run_silver_etl_creates_schema(mock_spark):
    run_silver_etl(mock_spark, "cat", "src", "cat", "tgt")
    sql_calls = [str(c.args[0]) for c in mock_spark.sql.call_args_list]
    assert any("CREATE SCHEMA IF NOT EXISTS cat.tgt" in s for s in sql_calls)


def test_run_silver_etl_merges_session_summary(mock_spark):
    run_silver_etl(mock_spark, "cat", "src", "cat", "tgt")
    sql_calls = [str(c.args[0]) for c in mock_spark.sql.call_args_list]
    assert any("MERGE INTO cat.tgt.session_summary" in s for s in sql_calls)


def test_run_silver_etl_merges_session_metrics(mock_spark):
    run_silver_etl(mock_spark, "cat", "src", "cat", "tgt")
    sql_calls = [str(c.args[0]) for c in mock_spark.sql.call_args_list]
    assert any("MERGE INTO cat.tgt.session_metrics" in s for s in sql_calls)


def test_run_silver_etl_delete_then_append_session_events(mock_spark):
    run_silver_etl(mock_spark, "cat", "src", "cat", "tgt")
    sql_calls = [str(c.args[0]) for c in mock_spark.sql.call_args_list]
    assert any("DELETE FROM cat.tgt.session_events" in s for s in sql_calls)
    mock_spark.table.return_value.write.mode.assert_called_with("append")


def test_main_creates_spark_and_calls_etl():
    with patch("claude_otel_session_scorer.silver_etl.create_spark_session") as mock_create, \
         patch("claude_otel_session_scorer.silver_etl.run_silver_etl") as mock_etl:
        mock_spark = MagicMock()
        mock_create.return_value = mock_spark

        import sys
        sys.argv = [
            "silver_etl",
            "--source-catalog", "sc",
            "--source-schema", "ss",
            "--target-catalog", "tc",
            "--target-schema", "ts",
        ]
        main()

        mock_etl.assert_called_once_with(mock_spark, "sc", "ss", "tc", "ts")
        mock_spark.stop.assert_called_once()
