# Exact 2018 TSE result schemas

This patch aligns the party and tally slices to the headers observed in the
actual 2018 extracted resources.

Important semantic distinctions preserved:

Party resource:
- QT_VOTOS_LEGENDA_VALIDOS
- QT_VOTOS_NOMINAIS_CONVR_LEG
- QT_TOTAL_VOTOS_LEG_VALIDOS
- QT_VOTOS_NOMINAIS_VALIDOS
- QT_VOTOS_LEGENDA_ANUL_SUBJUD
- QT_VOTOS_NOMINAIS_ANUL_SUBJUD

Tally resource:
- QT_TOTAL_VOTOS_VALIDOS
- QT_VOTOS_NOMINAIS_VALIDOS
- QT_TOTAL_VOTOS_LEG_VALIDOS
- QT_VOTOS_LEG_VALIDOS
- QT_VOTOS_NOM_CONVR_LEG_VALIDOS
- annulled and sub-judice categories
- QT_TOTAL_VOTOS_NULOS vs QT_VOTOS_NULOS
- ST_VOTO_EM_TRANSITO

`is_transit_vote` is therefore part of the tally grain and of drill-across joins.

The party cube uses:
  party_valid_votes = nominal_valid_votes + total_legend_valid_votes

This deliberately uses `QT_TOTAL_VOTOS_LEG_VALIDOS` because it is the total
legend-valid amount after considering converted nominal-to-legend votes.
