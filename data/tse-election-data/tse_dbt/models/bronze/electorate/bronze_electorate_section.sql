{{ config(
    enabled=false,
    meta={
        'architecture_status': 'orphan',
        'architecture_reason': 'Source retained for future section-level electorate products'
    }
) }}

{#
  Disabled intentionally.

  domain='electorate_section' currently represents
  "Transferência temporária do eleitorado por seção", whose grain/schema is:
      origin municipality/zone/section
      destination municipality/zone/section
      transfer type
      QT_ELEITOR

  It is not a general electorate-by-section dataset. Reintroduce this as a
  dedicated modeled subject area when section-level electorate products exist.
#}

select 1 where false
