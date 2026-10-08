# TSE result reconciliation semantics

## Electorate balance

The official 2018 TSE publication rules explicitly state that the election
electorate does not always equal turnout + abstention. Electors in urns/sections
that were not installed or not counted remain outside both turnout and
abstention.

For the 2018 municipality/zone tally resource we can directly observe:

`QT_ELEITORES_SECOES_NAO_INSTALADAS`

Therefore the implemented invariant is:

eligible_voters
= turnout
+ abstentions
+ voters_uninstalled_sections

`uncounted_voters = eligible_voters - turnout - abstentions` remains explicit in
the semantic participation model so later election schemas can add other
uncounted categories without redefining turnout or abstention.

## Party vs tally

A single total delta hides whether disagreement originates in nominal votes or
legend votes. The `party_tally_reconciliation` semantic model therefore exposes:

- nominal_valid_delta
- total_legend_valid_delta
- total_valid_delta

The strict test still requires all three to be zero. If the test fails, the
diagnostic model identifies the exact component and grain rather than weakening
the invariant.
