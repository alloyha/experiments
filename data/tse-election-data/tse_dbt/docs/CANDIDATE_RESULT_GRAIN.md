# Candidate result grain

For the 2018 general-election candidate result resource, the source is already
unique at the analytical grain:

- election_year
- election_type
- election_code
- round_number
- uf
- municipality_code
- zone
- office_code
- candidate_id
- is_transit_vote

The measured 2018 profile contains 8,680,108 rows and exactly 8,680,108 grains,
with zero duplicate grains in every UF.

Therefore `stg_candidate_votes_munzona` must not use `row_number()` or any
other deduplication sort for this source. It is a direct projection from
`stg_candidate_votes_raw`, restricted to selected incremental partitions.

If a future source year violates grain uniqueness, that must be treated as an
explicit source-semantic change and modeled separately rather than silently
reintroducing arbitrary row selection.


## Materialization policy

Because the prepared raw source is already unique at the candidate municipal-zone
grain, `stg_candidate_votes_munzona` is a view rather than another physical copy.

This avoids writing the same 8,680,108 rows twice:

prepared Parquet -> stg_candidate_votes_raw (view)
                 -> stg_candidate_votes_munzona (view)
                 -> int_candidate_votes (view)
                 -> fact_candidate_votes (first analytical materialization)

The heavy raw/munzona row-count equivalence assertion remains available under
the `integrity_heavy` gate.
