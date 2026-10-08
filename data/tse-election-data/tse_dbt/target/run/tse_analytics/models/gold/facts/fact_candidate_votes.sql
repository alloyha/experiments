
  
  create view "tse_analytics"."main"."fact_candidate_votes__dbt_tmp" as (
    

select *
from read_parquet(
    '/home/pingu/github/experiments/data/tse-election-data/data/warehouse/fact_candidate_votes/election_type=*/year=*/data.parquet',
    hive_partitioning = true,
    union_by_name = true
)
  );
