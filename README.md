# Claude OTEL Session Scorer

A series of data pipelines for Databricks to score claude code sessions collected from Open Telemetry

## Features

- PySpark application for Databricks deployment
- Databricks Connect integration for local development
- Poetry for dependency management
- Type hints for better code quality
- Command-line interface for easy usage

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

```bash
# Install Poetry if you don't have it
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install
```

## Usage

### Command Line Interface

```bash
poetry run claude_otel_session_scorer --table-name "your_catalog.your_schema.your_table"
```

### As a Python Module

```python
from claude_otel_session_scorer.main import scan_table, create_spark_session

spark = create_spark_session()
result_df = scan_table(spark, "your_catalog.your_schema.your_table")
result_df.show()
```

## Development

### Running Tests

```bash
poetry run pytest
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
