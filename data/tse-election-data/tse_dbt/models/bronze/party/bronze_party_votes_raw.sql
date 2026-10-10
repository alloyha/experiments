{{ config(
    pre_hook=[
      "{{ ensure_source_snapshot_column_pre_hook() }}",
      "{{ source_aware_partition_replace_pre_hook('vote_result', 'Votação em partido por município e zona%') }}"
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
        'heavy_streaming_source_aware'
    }
) }}


{% set should_refresh =
    source_snapshot_changed(
      'vote_result',
      'Votação em partido por município e zona%'
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
      'Votação em partido por município e zona%',
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
      'Votação em partido por município e zona%',
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

      "TP_AGREMIACAO"
        as party_group_type,

      "NR_PARTIDO" as party_number,
      "SG_PARTIDO" as party,
      "NM_PARTIDO" as party_name,

      "NR_FEDERACAO"
        as federation_number,

      "NM_FEDERACAO"
        as federation_name,

      "SG_FEDERACAO"
        as federation,

      "DS_COMPOSICAO_FEDERACAO"
        as federation_composition,

      "SQ_COLIGACAO"
        as coalition_id,

      "NM_COLIGACAO"
        as coalition_name,

      "DS_COMPOSICAO_COLIGACAO"
        as coalition_composition,

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

      try_cast(
        "QT_VOTOS_LEGENDA_VALIDOS"
        as bigint
      ) as legend_valid_votes,

      {% if 2018 in var('election_years', []) %}

      try_cast(
        "QT_VOTOS_NOMINAIS_CONVR_LEG"
        as bigint
      )

      {% else %}

      try_cast(
        "QT_VOTOS_NOM_CONVR_LEG_VALIDOS"
        as bigint
      )

      {% endif %}
        as nominal_converted_to_legend_votes,

      try_cast(
        "QT_TOTAL_VOTOS_LEG_VALIDOS"
        as bigint
      ) as total_legend_valid_votes,

      try_cast(
        "QT_VOTOS_NOMINAIS_VALIDOS"
        as bigint
      ) as nominal_valid_votes,

      try_cast(
        "QT_VOTOS_LEGENDA_ANUL_SUBJUD"
        as bigint
      ) as legend_annulled_subjudice_votes,

      try_cast(
        "QT_VOTOS_NOMINAIS_ANUL_SUBJUD"
        as bigint
      ) as nominal_annulled_subjudice_votes,

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
