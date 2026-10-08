# Tally and turnout slice

`Detalhe da apuração por município e zona`
→ `stg_tally_munzona`
→ `fact_tally_munzona`
→ `fact_turnout`
→ `electoral_participation`

The tally fact supplies the denominators needed by analytical cubes:
- eligible voters
- turnout
- abstentions
- valid votes
- blank votes
- null votes

`fact_turnout` deliberately retains `office_code` in its grain. TSE tally metrics
are reported in an office context, so summing turnout across offices would
multiply the same electorate/attendance population.

Ratios are non-additive:
- vote_share
- turnout_rate
- abstention_rate

Every roll-up recomputes them from additive numerators and denominators rather
than summing or averaging lower-level ratios.
