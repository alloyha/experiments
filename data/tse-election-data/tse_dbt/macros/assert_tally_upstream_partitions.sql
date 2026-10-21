{#
  Fail closed before a tally partition-replace DELETE.  This gate checks that
  the selected upstream is not wholly empty, and that every selected cycle
  already represented by the target still exists upstream.

  It does NOT certify source-snapshot completeness or row-level parity.
  Those require separate publication checks against durable source metadata.
#}
{% macro assert_tally_upstream_partitions(upstream) %}
  {% if not is_incremental() %}
    {{ return('select 1') }}
  {% endif %}
  {% set predicate = incremental_partition_predicate('election_year', 'election_type') %}
  {% set target_predicate = incremental_partition_predicate('target.election_year', 'target.election_type') %}
  {% set source_predicate = incremental_partition_predicate('src.election_year', 'src.election_type') %}
  {{ return(
    'select case when ' ~
    'not exists (select 1 from ' ~ upstream ~ ' where ' ~ predicate ~ ') ' ~
    'or exists (select 1 from (select distinct target.election_year, target.election_type ' ~
      'from ' ~ this ~ ' as target where ' ~ target_predicate ~ ') as existing ' ~
      'where not exists (select 1 from ' ~ upstream ~ ' as src ' ~
      'where ' ~ source_predicate ~
      ' and src.election_year = existing.election_year ' ~
      'and src.election_type = existing.election_type)) ' ~
    "then error('C3 unsafe tally partition replacement: missing selected upstream cycle') " ~
    'else 1 end'
  ) }}
{% endmacro %}
