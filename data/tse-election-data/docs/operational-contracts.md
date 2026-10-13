# Operational contracts

The pipeline is production-defensible only when the raw control plane, warehouse
regressions, and dbt semantic tests agree.

## Raw contracts

`make raw-contracts` verifies:

1. every `source_object` and `object` referenced by `current_objects.jsonl`
   exists physically;
2. every active `domain=candidate/year=Y` object contains only
   `ANO_ELEICAO = Y`.

## Candidate domain ontology

- `consulta_cand_*` -> `candidate`
- `historico_candidatura_*` / "Histórico de candidaturas" -> `candidate_history`

The lean `analytics` profile excludes `candidate_history`; `extended` and
`mirror` may retain it as a distinct longitudinal domain.

Legacy state rows are normalized on load so a historical misclassification
cannot be resurrected as `candidate`.

## Cycle regressions

`make cycle-regressions` reads `contracts/election_cycles.json` and validates the
frozen warehouse baselines across 2018, 2020, 2022, 2024, and the provisional
2026 snapshot.

## Scope configuration

- `YEARS` selects the electoral cycles included in analytics.
- `UPDATE_YEARS` selects the years refreshed during this execution; by default,
  it inherits `YEARS` and must remain its subset.
- Election types are inferred from `tse_dbt/seeds/election_calendar.csv` and
  validated against `tse_ingest.ELECTION_CALENDAR`.
- dbt still receives its internal `election_types`, `incremental_years`, and
  `incremental_election_types` values automatically.

## Production gate

```bash
make production-gate YEARS=2026
```

To keep multiple cycles in analytical scope while refreshing only 2026:

```bash
make production-gate \
  YEARS="2018 2020 2022 2024 2026" UPDATE_YEARS=2026
```

The gate runs unit contracts, ingestion/rehydration, raw contracts, the normal
dbt build, and warehouse-wide cycle regressions.

A TSE republication may require an intentional update to a provisional baseline.
A silent baseline drift is a failure.
