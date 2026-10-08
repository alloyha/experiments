# Valid-vote semantics

The TSE party-vote resource distinguishes valid nominal/list votes from broader
nominal vote totals. The analytical drill-across must compare like with like:

- candidate side: `QT_VOTOS_NOMINAIS_VALIDOS`
- party side: `QT_VOTOS_NOMINAIS_VALIDOS`
- party legend side: `QT_VOTOS_LEGENDA_VALIDOS`

`QT_VOTOS_NOMINAIS` remains available in the candidate fact as the broader
published nominal total, but it is not used for party reconciliation or
vote-share calculations.

This prevents annulled/sub-judice vote categories from silently contaminating
the candidate↔party reconciliation.
