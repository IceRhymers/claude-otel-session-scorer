"""
Tests for the silver_etl module
"""

from unittest.mock import MagicMock, patch

from claude_otel_session_scorer.silver_etl import main, run_silver_etl


def _make_mock_spark():
    spark = MagicMock()
    # spark.table() returns a MagicMock that supports arbitrarily chained DataFrame ops
    df = MagicMock()
    spark.table.return_value = df
    return spark


def _sql_calls(spark):
    return [c.args[0].strip() for c in spark.sql.call_args_list]


def test_run_silver_etl_calls_schema_create():
    spark = _make_mock_spark()
    run_silver_etl(spark, "cat.src", "cat.tgt")
    assert any("CREATE SCHEMA IF NOT EXISTS cat.tgt" in s for s in _sql_calls(spark))


def test_session_summary_merge():
    spark = _make_mock_spark()
    run_silver_etl(spark, "cat.src", "cat.tgt")
    merge_calls = [s for s in _sql_calls(spark) if "MERGE INTO cat.tgt.session_summary" in s]
    assert len(merge_calls) == 1
    assert "WHEN MATCHED THEN UPDATE SET *" in merge_calls[0]
    assert "WHEN NOT MATCHED THEN INSERT *" in merge_calls[0]


def test_session_events_delete_then_append():
    spark = _make_mock_spark()
    # Capture the DataFrame returned from the final unionByName chain so we can
    # check that .write.mode("append").saveAsTable(...) was called on it.
    run_silver_etl(spark, "cat.src", "cat.tgt")

    delete_calls = [s for s in _sql_calls(spark) if "DELETE FROM cat.tgt.session_events" in s]
    assert len(delete_calls) == 1

    # Verify saveAsTable was invoked with the silver events table name somewhere
    # in the MagicMock call graph (write.mode("append").saveAsTable).
    # Because the chained MagicMock records every call, we walk call_args_list of
    # all mock children to find saveAsTable("cat.tgt.session_events").
    all_calls = str(spark.mock_calls)
    assert "saveAsTable" in all_calls
    assert "cat.tgt.session_events" in all_calls


def test_session_metrics_merge():
    spark = _make_mock_spark()
    run_silver_etl(spark, "cat.src", "cat.tgt")
    merge_calls = [s for s in _sql_calls(spark) if "MERGE INTO cat.tgt.session_metrics" in s]
    assert len(merge_calls) == 1
    assert "WHEN MATCHED THEN UPDATE SET *" in merge_calls[0]
    assert "WHEN NOT MATCHED THEN INSERT *" in merge_calls[0]


def test_main_creates_spark_and_stops():
    with (
        patch("claude_otel_session_scorer.silver_etl.create_spark_session") as mock_create,
        patch("claude_otel_session_scorer.silver_etl.run_silver_etl") as mock_run,
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
