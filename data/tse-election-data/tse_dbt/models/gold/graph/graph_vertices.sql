{{ config(materialized='view', tags=['graph', 'expensive'], meta={'graph_role': 'vertices', 'history_semantics': 'derived_live_gold'}) }}

-- One typed vertex per conformed natural identity; no numeric ID offsets.
-- Source facts are included when metadata is missing: a missing descriptive
-- member must never erase an observed candidate/party/municipality vertex.
-- This is a lazy Gold projection, NOT a persisted snapshot or certified result.
with candidate_keys as (
    select cycle_id, election_id, candidate_id from {{ ref('dim_candidate') }}
    union
    select cycle_id, election_id, candidate_id from {{ ref('fact_candidate_votes') }}
),
party_keys as (
    select cycle_id, election_id, party_number from {{ ref('dim_party') }}
    union
    select cycle_id, election_id, party_number from {{ ref('fact_party_votes') }}
    union
    select cycle_id, election_id, party_number from {{ ref('dim_candidate') }}
    where party_number is not null
),
municipality_keys as (
    select cycle_id, uf, municipality_code from {{ ref('dim_geography') }}
    union
    select cycle_id, uf, municipality_code from {{ ref('fact_candidate_votes') }}
    union
    select cycle_id, uf, municipality_code from {{ ref('fact_party_votes') }}
),
contest_keys as (
    select distinct cycle_id, election_id, round_number, uf, municipality_code,
        office_code, is_transit_vote
    from (
        select cycle_id, election_id, round_number, uf, municipality_code,
            office_code, is_transit_vote from {{ ref('fact_candidate_votes') }}
        union all
        select cycle_id, election_id, round_number, uf, municipality_code,
            office_code, is_transit_vote from {{ ref('fact_party_votes') }}
    ) source_contests
)
select
    to_json(list_value('cycle', cast(cycle_id as varchar))) as node_id, 'cycle' as node_type, cycle_id,
    cast(null as varchar) as election_id, cycle_id as label,
    true as metadata_matched, 'dim_election_cycle' as source_relation
from {{ ref('dim_election_cycle') }}
union all
select
    to_json(list_value('election', cast(election_id as varchar))) as node_id, 'election' as node_type, cycle_id,
    election_id, election_id as label,
    true as metadata_matched, 'dim_election' as source_relation
from {{ ref('dim_election') }}
union all
select
    to_json(list_value('municipality', cast(k.cycle_id as varchar), cast(k.uf as varchar), cast(k.municipality_code as varchar))) as node_id, 'municipality' as node_type,
    k.cycle_id, cast(null as varchar) as election_id,
    g.municipality as label, (g.municipality_code is not null) as metadata_matched,
    'dim_geography|fact_votes' as source_relation
from municipality_keys k
left join {{ ref('dim_geography') }} g
    on g.cycle_id = k.cycle_id and g.uf = k.uf
    and g.municipality_code = k.municipality_code
union all
select
    to_json(list_value('candidate', cast(k.election_id as varchar), cast(k.candidate_id as varchar))) as node_id, 'candidate' as node_type,
    k.cycle_id, k.election_id, d.ballot_name as label,
    (d.candidate_id is not null) as metadata_matched,
    'dim_candidate|fact_candidate_votes' as source_relation
from candidate_keys k
left join {{ ref('dim_candidate') }} d
    on d.election_id = k.election_id and d.candidate_id = k.candidate_id
union all
select
    to_json(list_value('party', cast(k.election_id as varchar), cast(k.party_number as varchar))) as node_id, 'party' as node_type,
    k.cycle_id, k.election_id, d.party_name as label,
    (d.party_number is not null) as metadata_matched,
    'dim_party|fact_party_votes|dim_candidate' as source_relation
from party_keys k
left join {{ ref('dim_party') }} d
    on d.election_id = k.election_id and d.party_number = k.party_number
union all
select
    to_json(list_value('contest', cast(election_id as varchar), cast(round_number as varchar), cast(uf as varchar), cast(municipality_code as varchar), cast(office_code as varchar), cast(is_transit_vote as varchar))) as node_id, 'contest' as node_type, cycle_id, election_id,
    cast(office_code as varchar) as label,
    true as metadata_matched, 'fact_candidate_votes|fact_party_votes' as source_relation
from contest_keys
