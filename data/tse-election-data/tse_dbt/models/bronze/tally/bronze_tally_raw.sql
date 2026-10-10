{{ config(
    pre_hook=[
      "{{ ensure_source_snapshot_column_pre_hook() }}",
      "{{ source_aware_partition_replace_pre_hook('vote_result', 'Detalhe da apuração por município e zona%') }}"
    ],
    materialized='incremental',
    incremental_strategy='append',
    on_schema_change='sync_all_columns',
    meta={
      'load_semantics':
        'source_aware_partition_replace',
      'history_semantics': 'source_versions',
      'partition_key': [
        'election_year',
        'election_type'
      ],
      'source_snapshot':
        'current_objects.source_sha256',
      'resource_class':
        'medium_source_aware'
    }
) }}


{% set should_refresh =
    source_snapshot_changed(
      'vote_result',
      'Detalhe da apuração por município e zona%'
    )
%}


{% if is_incremental() and not should_refresh %}

select *
from {{ this }}
where false

{% else %}

with current_snapshot as (

  {{
    source_snapshot_relation(
      'vote_result',
      'Detalhe da apuração por município e zona%',
      incremental_scope=is_incremental()
    )
  }}

),

src as (

  select
    raw.*,
    current_snapshot.source_snapshot_id

  from {{
    read_raw_csv(
      'vote_result',
      'Detalhe da apuração por município e zona%',
      incremental_scope=is_incremental()
    )
  }} as raw

  inner join current_snapshot
    on try_cast(
         raw."ANO_ELEICAO" as integer
       ) = current_snapshot.election_year

   and raw._election_type
       = current_snapshot.election_type

),

typed as (

  select
      try_cast(
        "ANO_ELEICAO" as integer
      ) as election_year,

      _election_type as election_type,
      _election_scope as election_scope,

      "CD_ELEICAO" as election_code,

      try_cast(
        "NR_TURNO" as integer
      ) as round_number,

      try_strptime(
        trim("DT_GERACAO")
        || ' '
        || trim("HH_GERACAO"),
        '%d/%m/%Y %H:%M:%S'
      ) as generated_at,

      "SG_UF" as uf,

      {{
        normalize_municipality_code(
          '"CD_MUNICIPIO"'
        )
      }} as municipality_code,

      try_cast(
        "NR_ZONA" as integer
      ) as zone,

      "CD_CARGO" as office_code,

      {{
        office_scope(
          '"DS_CARGO"'
        )
      }} as office_scope,

      case
        when upper(
          trim("ST_VOTO_EM_TRANSITO")
        ) = 'S'
        then true

        when upper(
          trim("ST_VOTO_EM_TRANSITO")
        ) = 'N'
        then false

        else null
      end as is_transit_vote,

      try_cast("QT_APTOS" as bigint)
        as eligible_voters,

      try_cast(
        "QT_SECOES_PRINCIPAIS" as bigint
      ) as main_sections,

      try_cast(
        "QT_SECOES_AGREGADAS" as bigint
      ) as aggregated_sections,

      try_cast(
        "QT_SECOES_NAO_INSTALADAS" as bigint
      ) as uninstalled_sections,

      try_cast(
        "QT_TOTAL_SECOES" as bigint
      ) as total_sections,

      try_cast(
        "QT_COMPARECIMENTO" as bigint
      ) as turnout,

      try_cast(
        "QT_ELEITORES_SECOES_NAO_INSTALADAS"
        as bigint
      ) as voters_uninstalled_sections,

      try_cast(
        "QT_ABSTENCOES" as bigint
      ) as abstentions,

      try_cast(
        "QT_VOTOS" as bigint
      ) as total_votes,

      try_cast(
        "QT_VOTOS_CONCORRENTES" as bigint
      ) as competing_votes,

      try_cast(
        "QT_TOTAL_VOTOS_VALIDOS" as bigint
      ) as valid_votes,

      try_cast(
        "QT_VOTOS_NOMINAIS_VALIDOS" as bigint
      ) as nominal_valid_votes,

      try_cast(
        "QT_TOTAL_VOTOS_LEG_VALIDOS" as bigint
      ) as total_legend_valid_votes,

      try_cast(
        "QT_VOTOS_LEG_VALIDOS" as bigint
      ) as legend_valid_votes,

      try_cast(
        "QT_VOTOS_NOM_CONVR_LEG_VALIDOS"
        as bigint
      ) as nominal_converted_to_legend_valid_votes,

      try_cast(
        "QT_TOTAL_VOTOS_ANULADOS" as bigint
      ) as annulled_votes,

      try_cast(
        "QT_VOTOS_NOMINAIS_ANULADOS" as bigint
      ) as nominal_annulled_votes,

      try_cast(
        "QT_VOTOS_LEGENDA_ANULADOS" as bigint
      ) as legend_annulled_votes,

      try_cast(
        "QT_TOTAL_VOTOS_ANUL_SUBJUD" as bigint
      ) as annulled_subjudice_votes,

      try_cast(
        "QT_VOTOS_NOMINAIS_ANUL_SUBJUD"
        as bigint
      ) as nominal_annulled_subjudice_votes,

      try_cast(
        "QT_VOTOS_LEGENDA_ANUL_SUBJUD"
        as bigint
      ) as legend_annulled_subjudice_votes,

      try_cast(
        "QT_VOTOS_BRANCOS" as bigint
      ) as blank_votes,

      try_cast(
        "QT_TOTAL_VOTOS_NULOS" as bigint
      ) as total_null_votes,

      try_cast(
        "QT_VOTOS_NULOS" as bigint
      ) as null_votes,

      try_cast(
        "QT_VOTOS_NULOS_TECNICOS" as bigint
      ) as technical_null_votes,

      try_cast(
        "QT_VOTOS_ANULADOS_APU_SEP" as bigint
      ) as separately_counted_annulled_votes,

      try_strptime(
        trim("DT_ULTIMA_TOTALIZACAO")
        || ' '
        || trim("HH_ULTIMA_TOTALIZACAO"),
        '%d/%m/%Y %H:%M:%S'
      ) as last_totalization_at,

      filename as source_file,

      source_snapshot_id

  from src

)

select *
from typed

{% if is_incremental() %}
where {{
  incremental_partition_predicate(
    'election_year',
    'election_type'
  )
}}
{% endif %}

{% endif %}
