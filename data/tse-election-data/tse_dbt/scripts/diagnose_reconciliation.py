#!/usr/bin/env python3
import duckdb
from pathlib import Path

db = Path("data/warehouse/tse_analytics.duckdb")
con = duckdb.connect(str(db), read_only=True)

print("=== tally electorate balance ===")
print(con.sql("""
select
    count(*) as mismatches,
    sum(eligible_voters - turnout - abstentions) as residual,
    sum(coalesce(voters_uninstalled_sections, 0)) as uninstalled_voters
from main.fact_tally_munzona
where eligible_voters <> turnout + abstentions + coalesce(voters_uninstalled_sections, 0)
"""))

print("\n=== party/tally reconciliation summary ===")
print(con.sql("""
select
    count(*) filter (where nominal_valid_delta <> 0) as nominal_mismatches,
    sum(nominal_valid_delta) as nominal_delta,
    count(*) filter (where total_legend_valid_delta <> 0) as legend_mismatches,
    sum(total_legend_valid_delta) as legend_delta,
    count(*) filter (where total_valid_delta <> 0) as total_mismatches,
    sum(total_valid_delta) as total_delta
from main.party_tally_reconciliation
"""))

print("\n=== largest party/tally mismatches ===")
print(con.sql("""
select *
from main.party_tally_reconciliation
where nominal_valid_delta <> 0
   or total_legend_valid_delta <> 0
   or total_valid_delta <> 0
order by abs(total_valid_delta) desc,
         abs(nominal_valid_delta) desc,
         abs(total_legend_valid_delta) desc
limit 25
"""))

con.close()
