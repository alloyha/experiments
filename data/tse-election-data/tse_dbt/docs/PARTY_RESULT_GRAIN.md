# Party result grain

The 2018 TSE party-vote resource contains multiple representational rows at the
same analytical party grain.

Validated behavior:
- 607,022 duplicate grains;
- 74 nominal differences;
- every nominal difference is `0 + positive`;
- legend measures never differ;
- MAX-collapse reconciles nominal and legend totals to tally with zero mismatch.

Therefore `silver_party_votes_munzona` uses measure-wise MAX at the analytical grain.

The model retains source multiplicity diagnostics:
- source_row_count
- source_party_group_types
- source_coalitions
- source_federations
