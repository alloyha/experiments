{% macro candidate_votes_prepared_relation() -%}
read_parquet(
    '{{ var("tse_raw_root") }}/raw/election_type=*/year=*/domain=*/dataset=*/resource=*/sha256=*/prepared/votacao_candidato_munzona_*.parquet',
    union_by_name=true,
    hive_partitioning=false
)
{%- endmacro %}
