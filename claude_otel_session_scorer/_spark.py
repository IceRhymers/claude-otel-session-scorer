"""Shared Spark session factory for all claude_otel_session_scorer pipelines."""

from __future__ import annotations

import os


def create_spark_session():
    """Return (or reuse) the active SparkSession for this job.

    When running inside a Databricks cluster (DATABRICKS_RUNTIME_VERSION is set)
    the ambient session is returned directly.  Outside a cluster, Databricks
    Connect (serverless) is tried first; if it is not installed a standard local
    SparkSession is returned with a clear warning.
    """
    from pyspark.sql import SparkSession

    if os.environ.get("DATABRICKS_RUNTIME_VERSION") is None:
        try:
            from databricks.connect import DatabricksSession

            return DatabricksSession.builder.serverless().getOrCreate()
        except ImportError:
            print("Databricks Connect not available. Falling back to standard Spark session.")
            return SparkSession.builder.getOrCreate()
    return SparkSession.builder.getOrCreate()
