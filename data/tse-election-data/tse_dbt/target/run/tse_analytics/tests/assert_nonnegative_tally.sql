
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  select *
from "tse_analytics"."main"."fact_tally_munzona"
where eligible_voters < 0
   or turnout < 0
   or abstentions < 0
   or total_votes < 0
   or valid_votes < 0
   or nominal_valid_votes < 0
   or total_legend_valid_votes < 0
   or blank_votes < 0
   or total_null_votes < 0
   or annulled_votes < 0
   or annulled_subjudice_votes < 0
  
  
      
    ) dbt_internal_test