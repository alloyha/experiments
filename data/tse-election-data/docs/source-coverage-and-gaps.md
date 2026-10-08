# Source coverage and reconciliation gaps

Reconciliation remains strict wherever the compared TSE sources have comparable
coverage. A gap is an explicit statement that the source coverage is not
comparable at that grain; it is not a device for changing facts or hiding a
mismatch.

## Candidate coverage

### `missing_candidate_result_coverage`

Tally nominal votes exist for a reconciliation grain for which the candidate
result source has no corresponding coverage.

The known 2018 regression is election `339`, PE, municipality `30015`, zone `4`,
office `25`, with 1,836 nominal valid votes.

## Party coverage

### `missing_party_source_coverage`

The tally has valid votes, but no party-source rows exist at the same
election/round/UF/municipality/zone/office/transit-vote grain.

### `missing_party_nominal_coverage`

Party rows exist and their legend component reconciles exactly with tally, but
the party source exposes no nominal component while tally nominal votes are
positive.

### `converted_vote_overlap`

The party source contains nominal votes converted to legend totals. The resulting
known overlap is explicitly classified rather than changing the canonical
`party_valid_votes = nominal_valid_votes + total_legend_valid_votes` formula.

## Invariant

Rows classified as coverage gaps are excluded from strict source-to-source
reconciliation. Every non-gap comparable grain must reconcile exactly.
