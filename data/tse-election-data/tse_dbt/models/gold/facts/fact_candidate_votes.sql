{{ config(
    materialized='view',
    meta={
        'physical_source': 'candidate_fact_partition'
    }
) }}

select *
from read_parquet(
    '{{ var(
        "candidate_fact_root",
        var("tse_raw_root") ~ "/../warehouse/fact_candidate_votes"
    ) }}/election_type=*/year=*/data.parquet',
    hive_partitioning = true,
    union_by_name = true
)
