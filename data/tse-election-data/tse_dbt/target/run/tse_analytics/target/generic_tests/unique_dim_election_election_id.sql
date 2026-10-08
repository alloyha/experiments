
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    

select
    election_id as unique_field,
    count(*) as n_records

from "tse_analytics"."main"."dim_election"
where election_id is not null
group by election_id
having count(*) > 1



  
  
      
    ) dbt_internal_test