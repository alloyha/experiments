# Party performance analytical slice

Pipeline:

`Votação em partido por município e zona`
→ `stg_party_votes_raw`
→ `stg_party_votes_munzona`
→ `dim_party` + `fact_party_votes`
→ `party_performance`

The party fact separates:
- nominal votes attributed to party candidates;
- legend votes cast directly for the party;
- total party votes = nominal + legend.

`party_performance` performs a controlled drill-across against candidate votes at
the exact shared grain. Candidate votes are mapped to `party_number` through
`dim_candidate`, and `nominal_reconciliation_delta` exposes any discrepancy.

`dim_party` is intentionally an election snapshot for the current 2018 reference
year. It is marked as the first SCD2 candidate, but Type 2 history is not invented
until multiple cycles are loaded and a persistent business key is validated.

Vote share and turnout rates are deliberately not implemented here. They require
`tally / turnout` as a denominator and will be recomputed at the requested cube
grain rather than summed across lower-level ratios.
