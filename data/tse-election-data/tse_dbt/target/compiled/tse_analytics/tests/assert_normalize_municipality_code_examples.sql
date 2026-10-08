with cases(input_value, expected_value) as (
    values
        ('35',    '00035'),
        ('1392',  '01392'),
        ('2550',  '02550'),
        ('4154',  '04154'),
        ('30015', '30015'),
        (' 35 ',  '00035'),
        ('',      null),
        (null,    null)
),
evaluated as (
    select
        input_value,
        expected_value,
        
case
    when input_value is null then null
    when trim(cast(input_value as varchar)) = '' then null
    else lpad(trim(cast(input_value as varchar)), 5, '0')
end
 as actual_value
    from cases
)
select *
from evaluated
where actual_value is distinct from expected_value