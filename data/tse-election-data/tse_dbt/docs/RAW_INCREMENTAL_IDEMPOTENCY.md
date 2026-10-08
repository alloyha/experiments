# Raw candidate incremental idempotency

`bronze_candidate_votes_raw` is an authoritative snapshot partitioned by
`(election_year, election_type)`.

The previous incremental behavior accumulated complete copies of the same
partition. For the 2018 general election the observed state was:

- 8,680,108 logical rows
- 26,040,324 physical rows
- every logical row present exactly 3 times

The model now uses dbt `delete+insert` with
`unique_key = [election_year, election_type]`, making the snapshot partition
itself the replacement key.

The regression test `assert_stg_candidate_votes_raw_no_physical_duplicates`
prevents this accumulation from returning.
