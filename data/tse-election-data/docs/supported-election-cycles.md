# Supported election cycles

The analytical warehouse has been validated end to end for five election cycles.

| Year | Election type | Status | Full dbt build |
|---:|---|---|---:|
| 2018 | general | frozen | 151/151 |
| 2020 | municipal | frozen | 151/151 |
| 2022 | general | frozen | 151/151 |
| 2024 | municipal | frozen | 151/151 |
| 2026 | general | provisional | 151/151 |

`frozen` means the cycle is treated as historical and its regression baseline must
change only through an intentional contract update.

`provisional` means the currently published TSE snapshot is internally consistent
and supported, but its absolute row-count baseline may legitimately change when
TSE republishes or finalizes the cycle. Such a change must still be reviewed and
the baseline updated explicitly; it is never accepted silently.

The authoritative machine-readable baselines live in
`contracts/election_cycles.json`.
