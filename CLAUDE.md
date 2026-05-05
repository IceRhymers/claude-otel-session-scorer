# CLAUDE.md

Guidance for Claude Code (and other LLM coding agents) working in this repo. Keep this short and load-bearing — the README is the human-facing entry point.

## Project purpose

A Databricks PySpark project that scores Claude Code sessions captured as OpenTelemetry traces, logs, and metrics. It runs as a chained Databricks Asset Bundle job: a silver ETL converts bronze OTEL tables into per-session tables, then two parallel gold pipelines write scores — an LLM-as-judge pipeline (`scorer.py`) and a deterministic human-friction pipeline (`human_signals.py`).

## Architecture at a glance

```
bronze (OTEL v1)  →  silver_etl  →  session_summary / session_events / session_metrics
                                       │
                                       ├─ scorer.py        → gold.session_scores                (LLM judge, immutable)
                                       └─ human_signals.py → gold.session_human_signals (+_by_tool)  (deterministic, recomputable)
```

- **Bronze** is written by an OTLP proxy that lives in `IceRhymers/databricks-claude`. Schema is mirrored in `docs/bronze-schema.sql`. Do not assume column names beyond what's there.
- The bronze→silver join keys are always derived from `attributes.getItem("session.id")` (or `sum.attributes.getItem("session.id")` for metrics).
- Silver `session_events` is a **union of six event projections** (USER_PROMPT, LLM_CALL, TOOL_CALL, TOOL_DECISION, TOOL_RESULT, ERROR variants) — every projection must include the **same column list** in the same order, including `prompt_id`, `tool_use_id`, and `decision_source`.

## Critical invariants

1. **NULL ≠ 0 for friction scores.** `human_friction_score` is `NULL` when `signal_strength` is false. Don't replace with 0 — `compute_friction_score` and the SQL formula both encode this contract and tests assert it.
2. **Scores are immutable; signals are recomputable.**
   - `scorer.py` uses a `left_anti` join against existing `session_scores` so a session is scored exactly once.
   - `human_signals.py` has **no** `left_anti` gate — every completed session is re-MERGEd on every run with `WHEN MATCHED THEN UPDATE SET *`. The `test_no_left_anti` and `test_first_run_backfills_all_completed_sessions` tests enforce this distinction.
3. **Completion guard.** Both gold pipelines filter `session_end < current_timestamp() - INTERVAL 2 HOURS` before scoring, to avoid scoring active sessions. Keep that filter intact.
4. **Per-tool delete-then-MERGE.** `session_human_signals_by_tool` does a `DELETE FROM ... WHERE session_id IN (...)` *before* the MERGE, so tools that disappeared between runs are dropped. Don't replace with a plain MERGE.
5. **Correction window is `<=` 30s.** `_CORRECTION_WINDOW_SECONDS = 30` and the predicate is boundary-inclusive (`<=`). Window must `partitionBy("session_id").orderBy("event_ts", "event_type")` for deterministic ordering under timestamp ties.
6. **Modify decisions are excluded.** `detail_name.isin("accept", "reject")` — `modify` and any other value are excluded from both numerator and denominator of `reject_rate`.
7. **`silver_events` write uses `mergeSchema=true`** with `saveAsTable(..., mode="append")`. The `_ensure_table_with_clustering` helper creates an empty `CLUSTER BY AUTO` Delta table on first run.

## Module map

| Path                                              | What it does                                                                                                  |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `claude_otel_session_scorer/main.py`              | `claude_otel_session_scorer` CLI (table-scan utility) + canonical `create_spark_session()`                     |
| `claude_otel_session_scorer/silver_etl.py`        | Bronze→silver. Three builders: `_build_session_summary`, `_build_session_events`, `_build_session_metrics`     |
| `claude_otel_session_scorer/scorer.py`            | LLM-as-judge. `format_event_line` / `split_into_interactions` / `compress_interaction` / `build_replay_text`   |
| `claude_otel_session_scorer/human_signals.py`     | Deterministic friction signals. Pure-Python `compute_friction_score` mirrors the SQL formula                   |
| `tests/`                                          | All four production modules have a sibling `test_*.py`; tests use `MagicMock(SparkSession)` plus source-level inspection |
| `docs/bronze-schema.sql`                          | Reference DDL for bronze OTEL tables                                                                          |
| `databricks.yml`                                  | Asset Bundle: wheel artifact, scheduled `Claude OTEL Pipeline` job, three Python wheel tasks                  |
| `dashboards/`                                     | Lakeview dashboard JSON                                                                                       |

## Running tests

```bash
poetry install           # one-time
poetry run pytest tests/ -v
# or:
make test
```

Tests do **not** spin up a real Spark — they wrap `MagicMock` and validate behavior either via mock-call assertions on `spark.sql(...)` or via `inspect.getsource(...)` against pinned strings. When you add a feature, add the equivalent test in the matching style.

## Lint / format

```bash
make lint        # ruff check . && ruff format --check .
make fmt         # ruff format .
ruff check .     # direct
ruff format .    # direct
```

CI runs `ruff check`, `ruff format --check`, `pytest`, and `poetry build` + `twine check`. All four must pass.

`pyproject.toml` sets `tool.ruff.line-length = 100`.

## Databricks-specific notes

- **Three-part names everywhere.** All `--*-schema` flags take `catalog.schema`. Tables are then `f"{schema}.{name}"`. Don't hardcode catalogs.
- **`ai_query` response format.** `scorer.py` declares both `RESPONSE_FORMAT` (the structured-output schema) and `FLAT_SCHEMA` (the `from_json` schema). Keep them in lockstep when adding/removing judgment fields.
- **Auto-clustering.** Every silver/gold table is created with `CLUSTER BY AUTO` (`option("clusterByAuto", "true")` for the helper, `CLUSTER BY AUTO` in DDL strings).
- **MERGE pattern.** `WHEN MATCHED THEN UPDATE SET * / WHEN NOT MATCHED THEN INSERT *` is the canonical idempotent upsert. Use a `createOrReplaceTempView` source view, then a `MERGE INTO` SQL string.
- **Spark session.** Always use `create_spark_session()` from the module — it transparently picks Databricks Connect serverless locally vs. ambient session inside DBR. Don't `SparkSession.builder.getOrCreate()` directly.
- **`spark.stop()`** should only run when off-cluster. `scorer.py` and `human_signals.py` already gate on `DATABRICKS_RUNTIME_VERSION`; preserve that.

## What NOT to do

- Don't make `human_signals` immutable (no `left_anti` gate, no "skip if already scored").
- Don't make `scorer` recomputable (no MERGE-without-anti-join; rescoring a session would be wasteful and breaks the immutable-history contract).
- Don't replace NULL with 0 in friction outputs.
- Don't add `ai_query`, UDFs, or LLM calls to `human_signals.py` — `test_no_udfs_or_ai_query` explicitly forbids it.
- Don't widen the `TOOL_DECISION` filter beyond `("accept", "reject")` without updating the test that pins it.
- Don't drop `prompt_id`, `tool_use_id`, or `decision_source` columns from any of the six `_build_session_events` projections — tests count `alias("...")` occurrences and require `>= 6`.
- Don't change `_CORRECTION_WINDOW_SECONDS` to `<` instead of `<=`; tests assert boundary inclusivity.
- Don't introduce env-var-driven config; CLI flags are the only knob.
