
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  select *
from "tse_analytics"."main"."fact_tally_munzona"
where total_votes <> valid_votes
                   + blank_votes
                   + total_null_votes
                   + annulled_votes
                   + annulled_subjudice_votes
                   + separately_counted_annulled_votes
  
  
      
    ) dbt_internal_test