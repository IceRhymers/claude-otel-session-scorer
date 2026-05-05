# AGENTS.md

Conventions and guardrails for AI coding agents (Claude Code, Cursor, Copilot agents, etc.) opening PRs against this repo. Pair this with [CLAUDE.md](./CLAUDE.md), which covers architecture and invariants in more depth.

## Scope

**Agents may autonomously:**

- Add or modify PySpark transforms inside `claude_otel_session_scorer/` so long as bronze→silver→gold contracts hold.
- Add tests under `tests/` for new behavior.
- Update CI (`.github/workflows/ci.yml`), `Makefile`, `pyproject.toml` dependencies — provided lint and the full test suite still pass.
- Edit `README.md`, `CLAUDE.md`, `AGENTS.md`, and `docs/` content.

**Agents must NOT autonomously:**

- Change the bronze table DDL in `docs/bronze-schema.sql` — it is mirrored from `IceRhymers/databricks-claude` and must stay in sync upstream first.
- Rotate Unity Catalog defaults in `databricks.yml` (`tanner_fevm_catalog.*`) without explicit human approval.
- Add, remove, or rename a `[tool.poetry.scripts]` entry point — these are referenced by `databricks.yml` job tasks and breaking them silently breaks production.
- Introduce real LLM/API calls in tests (tests must remain offline; use `MagicMock` Spark + source inspection).
- Commit any wheel, build artifact, or `.databrickscfg`-style credential.

## Data contract — do not break

The bronze→silver→gold schema contract is the load-bearing API of this repo. Treat the following as breaking changes that require an explicit human-reviewed migration:

- Removing or renaming any column on `session_summary`, `session_events`, `session_metrics`, `session_scores`, `session_human_signals`, or `session_human_signals_by_tool`.
- Changing the `MERGE` key for any of those tables (today: `session_id`, plus `tool_name` for the `_by_tool` table).
- Changing the meaning of `signal_strength`, `human_friction_score` (NULL-not-0), or any `*_rate` column.
- Changing the LLM judgment shape — `RESPONSE_FORMAT` and `FLAT_SCHEMA` in `scorer.py` must move together, and the `gold.session_scores` DDL must include any new column.

When a schema change *is* approved, include in the PR:

- The DDL change
- The transform change
- A note under "Migration" in the PR body describing what to run on existing tables (e.g. `ALTER TABLE ... ADD COLUMNS ...`, or a one-shot backfill script).

## Pipeline mutability — keep them straight

This repo runs **two gold pipelines with deliberately opposite mutability semantics**. Don't conflate them.

| Pipeline                  | Mutability       | How it's enforced                                                               |
| ------------------------- | ---------------- | ------------------------------------------------------------------------------- |
| `scorer.py` (LLM judge)   | **Immutable**    | `left_anti` join against `session_scores.session_id` — score once, never rewrite |
| `human_signals.py`        | **Recomputable** | No `left_anti`; every completed session re-MERGEd with `WHEN MATCHED UPDATE *`   |

If a refactor would make `human_signals` skip already-computed sessions, or make `scorer` overwrite existing rows, **stop and ask**. Tests pin both behaviors (`test_no_left_anti`, `test_first_run_backfills_all_completed_sessions`, the immutable-counterpart in `test_scorer.py`).

## Test requirements

Any new pipeline module — or any new transform inside an existing pipeline — must ship with a PySpark unit test in `tests/test_<module>.py`. Follow the established patterns:

1. **`MagicMock` Spark.** Use the `_make_mock_spark` / `_sql_calls` helpers as the template. Never instantiate a real `SparkSession` in tests.
2. **Assert on `spark.sql(...)` calls.** Walk `spark.sql.call_args_list` to verify DDL, MERGE, and DELETE statements landed.
3. **Source-level invariants where shape matters.** Use `inspect.getsource(fn)` and substring asserts to lock in things like `"INTERVAL 2 HOURS"`, `<= _CORRECTION_WINDOW_SECONDS`, `groupBy("session_id", "tool_name")`, etc.
4. **Pure helpers covered directly.** `compute_friction_score`, `split_into_interactions`, `build_replay_text`, `compress_interaction`, `format_event_line` all have direct unit tests — keep that.
5. **`main()` round-trip.** Each entry point has a `test_main_creates_spark_and_stops` that patches `create_spark_session` and `run_*`, asserts the right CLI args route through, and asserts `spark.stop()` is called. Mirror this for any new entry point.

## Lint and formatting

`ruff` is the only linter/formatter. CI fails on either `ruff check` or `ruff format --check` finding diffs.

```bash
make fmt         # auto-fix
make lint        # gate
```

Line length is 100 (configured in `pyproject.toml`).

## Commit messages

Use **conventional commit** prefixes — they're how `release-please`-style tooling and humans skim history. Branch-name prefixes do **not** count; the prefix must be in the commit message itself.

| Prefix      | Use for                                                                       |
| ----------- | ----------------------------------------------------------------------------- |
| `feat:`     | New user-visible capability (new column, new pipeline, new CLI flag)          |
| `fix:`      | Bug fix in an existing pipeline / transform / test                            |
| `chore:`    | Tooling, deps, CI, formatting-only changes                                    |
| `docs:`     | README, CLAUDE.md, AGENTS.md, comments, dashboards' descriptive fields        |
| `test:`     | Test-only additions or refactors (no production code change)                  |
| `refactor:` | Internal restructure with no behavioral change                                |

Keep the subject line ≤ 72 chars. Use the body to describe *why* and any migration impact.

## PR checklist

Before requesting review, verify:

- [ ] `make lint` passes (`ruff check` + `ruff format --check`)
- [ ] `make test` passes (`poetry run pytest tests/ -v`)
- [ ] `poetry build` succeeds and `twine check dist/*` is clean (CI does this — local optional)
- [ ] Conventional-commit prefix on every commit
- [ ] Any new gold/silver column is added to:
  - the corresponding builder / SELECT
  - the `CREATE TABLE` DDL string (if applicable)
  - the relevant tests
- [ ] If you touched `human_signals.py`: confirmed no `left_anti`, no `ai_query`, no UDFs, and `signal_strength=False ⇒ score is NULL`
- [ ] If you touched `scorer.py`: confirmed `left_anti` against `session_scores` is intact and `RESPONSE_FORMAT` / `FLAT_SCHEMA` / DDL move together
- [ ] If you changed any silver event projection: all six arms still emit the same column list, including `prompt_id`, `tool_use_id`, `decision_source`
- [ ] PR body calls out any schema-level migration step a Databricks operator must run

## When in doubt

- Read the matching `tests/test_*.py` first — it encodes the contract more precisely than prose.
- Prefer adding a new test that pins your new invariant rather than relaxing an existing one.
- If a test asserts a string like `'isin("accept", "reject")'` and your refactor would break that substring, the *test* is enforcing a contract — don't just delete the assertion. Either keep the substring or update the test deliberately and explain why in the PR.
