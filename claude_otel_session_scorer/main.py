import os
from pyspark.sql import SparkSession, DataFrame
from argparse import ArgumentParser


def create_spark_session() -> SparkSession:
    if os.environ.get("DATABRICKS_RUNTIME_VERSION") is None:
        try:
            from databricks.connect import DatabricksSession

            return DatabricksSession.builder.serverless().getOrCreate()
        except ImportError:
            print("Databricks Connect not available. Falling back to standard Spark session.")
            return SparkSession.builder.getOrCreate()
    else:
        return SparkSession.builder.getOrCreate()


def scan_table(spark: SparkSession, table_name: str, limit: int = 10) -> DataFrame:
    df: DataFrame = spark.table(table_name)
    return df.limit(limit)


def main() -> None:
    parser = ArgumentParser(
        description="A series of data pipelines for Databricks to score claude code sessions collected from Open Telemetry"
    )
    parser.add_argument("--table-name", "-t", type=str, help="Name of the table to scan")
    args = parser.parse_args()

    spark: SparkSession = create_spark_session()

    try:
        result_df: DataFrame = scan_table(spark, args.table_name)
        print(f"First 10 rows of table '{args.table_name}':")
        result_df.show()
        print(f"Schema of table '{args.table_name}':")
        result_df.printSchema()
    except Exception as e:
        print(f"Error scanning table '{args.table_name}': {str(e)}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
