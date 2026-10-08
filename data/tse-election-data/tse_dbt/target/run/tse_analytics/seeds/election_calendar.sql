 -- noqa: Should accept a string instead of a integer
    
    
    truncate table "tse_analytics"."main"."election_calendar";
    -- dbt seed --
    
          COPY "tse_analytics"."main"."election_calendar" FROM '/home/pingu/github/experiments/data/tse-election-data/tse_dbt/seeds/election_calendar.csv' (FORMAT CSV, HEADER TRUE, DELIMITER ',')
        

;
  