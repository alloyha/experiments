# Quality gate: one command, concise output

Run from `data/tse-election-data`:

```bash
make check              # compact local and CI report
make check VERBOSE=1    # full logs for debugging
make help               # short command overview
make help-all           # detailed targets
```

The four CI stages still run, in order: Python unit tests, medallion
lineage, registered ODCS lint/version/consumer compatibility and disposable dbt
synchronization with drift checking, and coordinated fixture-only dbt compile.
The wrapper uses the same Make targets; it does not relax validation, run an
uncoordinated warehouse writer, modify production data, or issue certificates.

Only a stage's exit code determines success. ODCS CLI warnings remain visible
as a short warning count even when the gate passes; full details are available
with `VERBOSE=1`. Text such as `FAIL:` emitted by
successful *negative tests* is not treated as a CI failure. Failed stages print
the last 50 lines of their combined output; `VERBOSE=1` prints all logs. In
GitHub Actions, the **job summary** displays a small table with gate status and
duration instead of repeating each full command output in the channel.

For targeted investigation, individual targets such as
`make data-contracts-breaking-check` and `make data-contracts-dbt-check`
remain available. No changes to ODCS baseline trust anchors or GG physical
certification are made by this UX-only step.
