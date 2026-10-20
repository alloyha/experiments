# ADR: Canonical party identity before public ODCS promotion

## Decision

`dim_party.party_id` is **an election-local party snapshot identifier**:

```
party_id = election_id || ':' || party_number
         = year || ':' || election_type || ':' || election_code || ':' || party_number
```

This is not a permanent legal entity identifier and must not imply SCD2, succession,
or party continuity across elections. The four-component key is the existing
row uniqueness grain; `election_id` is not sufficient to identify a party.

## Defect and migration

The previous SQL repeated `election_code`:

```
year:type:election_code:election_code:party_number
```

That redundant token formed a deterministic but noncanonical value. The correction
**changes `party_id` values** even though the composite natural grain is unchanged.
Consumers joining or caching the old string must recompute their keys; never
silently alias the old ID to a different identity.

The in-repository `party_performance` and
`cube_party_performance_municipality` models project `dim_party.party_id` and
join using the existing natural keys. Tests verify no vote multiplier from
this change. Existing external consumers are not discoverable from repository
SQL alone; verify them before a release or materialized warehouse refresh.

## Runtime boundary

This patch changes SQL and static tests only. It does not rebuild a production
warehouse, issue GG certificates, alter physical Parquet, or publish a new ODCS
baseline. A separately coordinated, scope-aware dbt rebuild is necessary for
materialized historical rows before serving the corrected `party_id`. Do not
interpret `make check` / `dbt compile` as a warehouse migration.

## Acceptance

1. The executed dbt SQL returns exactly `election_id:party_number` for each
   observed election party, even when source rows have duplicate versions.
2. Party ID is unique across two elections with the same party number.
3. The municipal party cube preserves the ID and additive vote totals.
4. The domain identity policy declares the election-local semantics.
5. `make check` passes on the branch, independently of expensive GG certification.

Promote `dim_party` to immutable ODCS `v1.0.0` only after these conditions and
consumer migration implications have been reviewed.
