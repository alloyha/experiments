{{ config(enabled=false) }}

{#
  Disabled intentionally.

  domain='electorate_section' currently represents
  "Transferência temporária do eleitorado por seção", whose grain/schema is:
      origin municipality/zone/section
      destination municipality/zone/section
      transfer type
      QT_ELEITOR

  It is not a general electorate-by-section dataset and must not be modeled as
  QT_ELEITORES_PERFIL. Reintroduce this as a dedicated
  stg_temporary_transfer_section model when that subject area is needed.
#}

select 1 where false
