{{ config(tags=['graph', 'expensive']) }}
-- An edge must respect its TYPE and the full electoral context encoded in
-- its typed endpoints, not just the existence of endpoints within one cycle.
-- The separate edge_endpoints assertion checks that those vertices exist.
with scoped_edges as (
    select * from {{ ref('graph_edges') }}
    where {{ incremental_partition_predicate("cast(split_part(cycle_id, ':', 1) as integer)", "split_part(cycle_id, ':', 2)") }}
), decoded as (
    select *,
        json_extract_string(source_node_id, '$[0]') as source_kind,
        json_extract_string(target_node_id, '$[0]') as target_kind,
        json_extract_string(source_node_id, '$[1]') as source_scope,
        json_extract_string(target_node_id, '$[1]') as target_scope,
        -- Contest ID: ['contest', election_id, round, uf, municipality, office, transit].
        case when edge_type in ('LOCATED_IN', 'VOTED_FOR_CANDIDATE', 'VOTED_FOR_PARTY')
            then source_node_id else target_node_id end as contest_node_id
    from scoped_edges
), validated as (
    select *,
        -- Construct the expected endpoint types; any unknown type fails closed.
        case edge_type
            when 'CONTAINS_ELECTION' then 'cycle'
            when 'HAS_CONTEST' then 'election'
            when 'LOCATED_IN' then 'contest'
            when 'HAS_CANDIDATE' then 'election'
            when 'HAS_PARTY' then 'election'
            when 'AFFILIATED_WITH' then 'candidate'
            when 'VOTED_FOR_CANDIDATE' then 'contest'
            when 'VOTED_FOR_PARTY' then 'contest'
        end as expected_source_kind,
        case edge_type
            when 'CONTAINS_ELECTION' then 'election'
            when 'HAS_CONTEST' then 'contest'
            when 'LOCATED_IN' then 'municipality'
            when 'HAS_CANDIDATE' then 'candidate'
            when 'HAS_PARTY' then 'party'
            when 'AFFILIATED_WITH' then 'party'
            when 'VOTED_FOR_CANDIDATE' then 'candidate'
            when 'VOTED_FOR_PARTY' then 'party'
        end as expected_target_kind
    from decoded
)
select edge_id, edge_type, cycle_id, election_id
from validated
where
    -- NULL keys and unknown edge types are not exempt from validation.
    cycle_id is null or election_id is null or edge_id is null
    or expected_source_kind is null or expected_target_kind is null
    or source_kind is distinct from expected_source_kind
    or target_kind is distinct from expected_target_kind
    or cycle_id is distinct from
        (split_part(election_id, ':', 1) || ':' || split_part(election_id, ':', 2))
    or case
        when edge_type = 'CONTAINS_ELECTION' then
            source_node_id is distinct from to_json(list_value('cycle', cycle_id))
            or target_node_id is distinct from to_json(list_value('election', election_id))
            or round_number is not null or office_code is not null or is_transit_vote is not null
        when edge_type in ('HAS_CANDIDATE', 'HAS_PARTY', 'AFFILIATED_WITH') then
            -- Entity-to-entity links are scoped to one election, not one person across time.
            source_scope is distinct from election_id
            or target_scope is distinct from election_id
            or round_number is not null or office_code is not null or is_transit_vote is not null
        when edge_type in ('HAS_CONTEST', 'LOCATED_IN',
                           'VOTED_FOR_CANDIDATE', 'VOTED_FOR_PARTY') then
            -- No cross-election endpoint within the same cycle; annotations
            -- must match the exact contest key embedded in the vertex ID.
            json_extract_string(contest_node_id, '$[1]') is distinct from election_id
            or json_extract_string(contest_node_id, '$[2]') is distinct from cast(round_number as varchar)
            or json_extract_string(contest_node_id, '$[5]') is distinct from office_code
            or json_extract_string(contest_node_id, '$[6]') is distinct from cast(is_transit_vote as varchar)
            or round_number is null or office_code is null or is_transit_vote is null
            or (edge_type = 'HAS_CONTEST' and source_scope is distinct from election_id)
            or (edge_type in ('VOTED_FOR_CANDIDATE', 'VOTED_FOR_PARTY')
                and target_scope is distinct from election_id)
            or (edge_type = 'LOCATED_IN' and (
                target_scope is distinct from cycle_id
                or json_extract_string(target_node_id, '$[2]') is distinct from
                    json_extract_string(contest_node_id, '$[3]')
                or json_extract_string(target_node_id, '$[3]') is distinct from
                    json_extract_string(contest_node_id, '$[4]')
            ))
        else true
    end
