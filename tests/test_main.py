"""
Tests for the main module
"""
import pytest
from unittest.mock import patch, MagicMock

from claude_otel_session_scorer.main import scan_table, create_spark_session


@pytest.fixture
def mock_spark():
    """Create a mock Spark session for testing."""
    with patch("claude_otel_session_scorer.main.SparkSession") as mock:
        spark = MagicMock()
        mock.builder.getOrCreate.return_value = spark
        yield spark


def test_scan_table(mock_spark):
    """Test the scan_table function."""
    mock_df = MagicMock()
    mock_spark.table.return_value = mock_df

    mock_result_df = MagicMock()
    mock_df.limit.return_value = mock_result_df

    result = scan_table(mock_spark, "test_table", limit=5)

    mock_spark.table.assert_called_once_with("test_table")
    mock_df.limit.assert_called_once_with(5)
    assert result == mock_result_df
