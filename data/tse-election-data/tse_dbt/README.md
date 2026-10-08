# TSE Analytics — dbt + DuckDB

Local analytical layer over the versioned TSE raw lake, explicitly distinguishing
**general elections** (federal + state offices) from **municipal elections**.

## Data flow

```text
TSE CKAN/CDN
    ↓
tse_ingest.py
    ↓
raw/election_type=general/year=2026/...
raw/election_type=municipal/year=2024/...
    ↓
_metadata/current_objects.jsonl
    ↓
dbt-duckdb
    ↓
tse_analytics.duckdb
```

The ingestion layer carries two related concepts:

- `election_type`: `general` or `municipal` — the election cycle.
- `election_scope`: `federal_state` or `municipal` — the broad scope covered by that cycle.

The candidate model adds `office_scope` at row level:

- `federal`: President, Senator, Federal Deputy and related federal offices.
- `state`: Governor, State/District Deputy and related state offices.
- `municipal`: Mayor, Vice-Mayor and Councilor.

These are intentionally distinct. A general election contains both federal and
state offices, so calling its row-level scope merely "general" would lose useful
information.

## Canonical election calendar

The project does **not** infer cycle type using `year % 4`. Regular cycles are
listed explicitly in `seeds/election_calendar.csv` and in the ingestion script.
For an exceptional or future year not yet listed, ingestion requires an explicit
`--election-type`.

## Setup

```bash
uv venv
uv pip install -r requirements.txt
uv run dbt deps --profiles-dir .
```

## Backfill both general and municipal cycles

```bash
uv run --with requests python ../tse_ingest.py \
  --mode backfill \
  --year 2022 --year 2024 --year 2026 \
  --root ../data/tse
```

The resulting raw hierarchy contains independent cycle partitions.

Build all three cycles:

```bash
uv run dbt build --profiles-dir . --full-refresh \
  --vars '{
    tse_raw_root: "../data/tse",
    election_years: [2022, 2024, 2026],
    election_types: ["general", "municipal"],
    incremental_years: [2022, 2024, 2026],
    incremental_election_types: ["general", "municipal"]
  }'
```

## Incremental refresh of the current general election

```bash
uv run --with requests python ../tse_ingest.py \
  --mode incremental --year 2026 --root ../data/tse

uv run dbt build --profiles-dir . \
  --vars '{
    tse_raw_root: "../data/tse",
    election_years: [2022, 2024, 2026],
    election_types: ["general", "municipal"],
    incremental_years: [2026],
    incremental_election_types: ["general"]
  }'
```

## Exceptional/future year

Unknown years are rejected rather than guessed:

```bash
python tse_ingest.py --year 2030 ...
# -> unknown election cycle ...
```

Until the canonical calendar is updated, classify it explicitly:

```bash
python tse_ingest.py \
  --year 2030 \
  --election-type general \
  --root ./data/tse
```

## Analytical contract

`dim_election` provides the canonical cycle dimension. Candidate and electorate
marts include `election_year`, `election_type`, and `election_scope`; candidate
models additionally include `office_scope`.

The dbt build contains consistency tests that fail if, for example, a municipal
cycle contains a federal/state candidate or a general cycle contains a mayor or
councilor.


## Loading semantics

Persistent warehouse models follow explicit loading contracts in
`model_contracts.yml` and `docs/adr/0001-warehouse-loading-semantics.md`.

- authoritative TSE snapshots: partition replacement
- canonical static dimensions: full replacement / Type 0
- SCD2: only persistent entities with independently changing attributes
- append-only: only genuine event/audit data

Incremental partition replacement is implemented with a pre-hook delete of the
selected `(election_year, election_type)` partition plus an exact current
snapshot insert.  Snapshot-completeness tests detect stale or missing rows.
