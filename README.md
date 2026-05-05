# Claude OTEL Session Scorer

Databricks data pipelines that turn raw Claude Code OpenTelemetry traces, logs, and metrics into per-session quality scores — both deterministic human-friction signals and LLM-as-judge ratings.

## What it does

The repo packages a PySpark wheel with three command-line entry points that run as a chained Databricks Asset Bundle job:

1. **`silver_etl`** — reads bronze OTEL Delta tables (`claude_otel_traces`, `claude_otel_logs`, `claude_otel_metrics`) emitted by an OTLP proxy and produces three silver tables: `session_summary`, `session_events`, and `session_metrics`.
2. **`score_sessions`** — LLM-as-judge pipeline. Builds a budget-bounded session "replay", calls `ai_query('databricks-claude-sonnet-4', ...)` with a structured response format, and writes one immutable row per completed session to `gold.session_scores`.
3. **`score_human_signals`** — deterministic friction-signal pipeline. Computes reject rate, abort rate, and correction intensity into `gold.session_human_signals` (one row per session) and `gold.session_human_signals_by_tool` (one row per (session, tool)). Recomputable on every run.

A Databricks AI/BI dashboard (`dashboards/Claude Code Session Scores V1.lvdash.json`) reads from the gold tables.

## Architecture

![Pipeline architecture](docs/pipeline.png)

| Stage  | Writers                  | Tables                                                                                  | Pattern                                                                               |
| ------ | ------------------------ | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Bronze | external OTLP proxy      | `claude_otel_traces`, `claude_otel_logs`, `claude_otel_metrics`                          | append (raw OTEL v1 schema; see `docs/bronze-schema.sql`)                              |
| Silver | `silver_etl`             | `session_summary`, `session_events`, `session_metrics`                                  | MERGE on `session_id` for summary/metrics; delete-then-append for events              |
| Gold   | `score_sessions`         | `session_scores`                                                                        | `left_anti` against existing rows — score once per session, then **immutable**         |
| Gold   | `score_human_signals`    | `session_human_signals`, `session_human_signals_by_tool`                                | every completed session re-MERGEd on each run — **recomputable**                       |

All silver and gold tables use Delta with `CLUSTER BY AUTO`. Scoring jobs only consider sessions whose `session_end` is more than 2 hours old, to avoid scoring in-flight work.

## Quickstart

### Local development

```bash
# 1. Install Poetry, then dependencies
curl -sSL https://install.python-poetry.org | python3 -
poetry install

# 2. Authenticate Databricks Connect (Python 3.12, runtime 18.0.5 client)
#    Configure DATABRICKS_HOST + DATABRICKS_TOKEN (or a profile) per
#    https://docs.databricks.com/dev-tools/databricks-connect.html

# 3. Run a pipeline locally against a remote workspace
poetry run silver_etl \
  --bronze-schema my_catalog.claude \
  --silver-schema my_catalog.claude_silver

poetry run score_sessions \
  --silver-schema my_catalog.claude_silver \
  --gold-schema my_catalog.claude_gold

poetry run score_human_signals \
  --silver-schema my_catalog.claude_silver \
  --gold-schema my_catalog.claude_gold
```

The `claude_otel_session_scorer` console script (`main.py`) is a small utility that scans and prints any Unity Catalog table — handy for ad-hoc inspection.

### Deploy to Databricks

The `databricks.yml` Asset Bundle defines a daily-at-1am-UTC job (`Claude OTEL Pipeline`) with `silver_etl` followed by `score_sessions` and `score_human_signals` running in parallel.

```bash
# Build the wheel and deploy the bundle
databricks bundle deploy --target dev

# Run on demand
databricks bundle run otel_pipeline_job --target dev
```

Schemas are passed as job parameters (`bronze_schema`, `silver_schema`, `gold_schema`) and default to `tanner_fevm_catalog.*`.

## Modules

| File                                              | Purpose                                                                                       |
| ------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `claude_otel_session_scorer/main.py`              | `claude_otel_session_scorer` CLI: scan-table utility + shared `create_spark_session()`         |
| `claude_otel_session_scorer/silver_etl.py`        | Bronze→silver ETL: builds `session_summary`, `session_events`, `session_metrics`               |
| `claude_otel_session_scorer/scorer.py`            | Silver→gold LLM-as-judge: replay compression, prompt build, `ai_query`, immutable scores       |
| `claude_otel_session_scorer/human_signals.py`     | Silver→gold deterministic friction signals (reject/abort/correction). Recomputable.            |
| `docs/bronze-schema.sql`                          | DDL for the bronze OTEL tables (provided by `IceRhymers/databricks-claude` OTLP proxy)         |
| `databricks.yml`                                  | Asset Bundle: wheel artifact, scheduled job, three task definitions                            |
| `dashboards/Claude Code Session Scores V1.lvdash.json` | AI/BI dashboard reading from gold tables                                                  |
| `tests/`                                          | PySpark unit tests using `MagicMock` Spark + `inspect.getsource` source-level invariants       |

## Configuration

The pipelines take their schema names from CLI flags; nothing else is configured via env vars. Two env vars influence Spark session creation:

| Variable                        | Effect                                                                                                                            |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `DATABRICKS_RUNTIME_VERSION`    | When **set** (i.e. running inside a Databricks cluster), `create_spark_session()` uses the ambient `SparkSession`.                |
| `DATABRICKS_HOST` / `DATABRICKS_TOKEN` (or profile) | When `DATABRICKS_RUNTIME_VERSION` is unset, `databricks-connect` opens a serverless session against the configured workspace. |

Unity Catalog three-part table names are required for every `--*-schema` flag (`catalog.schema`).

## Tests

```bash
poetry run pytest tests/ -v
# or
make test
```

Lint and format:

```bash
make lint        # ruff check . && ruff format --check .
make fmt         # ruff format .
```

CI (`.github/workflows/ci.yml`) runs ruff lint, ruff format check, pytest, and a `poetry build` + `twine check` on every push and PR to `main`.

## Contributing

- See [AGENTS.md](./AGENTS.md) for repo conventions, architecture context, and the contract autonomous agents must follow when editing this repo. (`CLAUDE.md` is a one-line redirect to the same file.)

## License

MIT — see `LICENSE`.
