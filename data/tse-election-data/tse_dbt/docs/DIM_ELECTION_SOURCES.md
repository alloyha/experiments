# dim_election source policy

`dim_election` must not scan `stg_candidate_votes_raw`.

Election identity can be sourced from the much smaller metadata/result relations:

- `stg_candidates` for elections represented in candidate metadata;
- `stg_party_votes_raw` / party result staging;
- `stg_tally_munzona` for authoritative tally-only elections such as 2018 code 339.

This preserves the `(election_year, election_type, election_code)` grain without
forcing an 8.7M-row candidate result scan merely to discover election IDs.
