
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  select *
from "tse_analytics"."main"."fact_party_votes"
where nominal_valid_votes < 0
   or legend_valid_votes < 0
   or total_legend_valid_votes < 0
   or party_valid_votes < 0
   or nominal_annulled_subjudice_votes < 0
   or legend_annulled_subjudice_votes < 0
  
  
      
    ) dbt_internal_test