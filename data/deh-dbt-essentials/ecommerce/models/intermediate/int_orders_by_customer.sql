select
    customer_id,
    count(*) as total_orders,
    sum(is_paid) as paid_orders,
    sum(is_cancelled) as cancelled_orders,
    sum(is_refunded) as refunded_orders,
    sum(recognized_revenue) as revenue,
    sum(refunded_amount) as refunded_amount,
    avg(
        case
            when is_paid = 1
            then amount
        end
    ) as average_paid_order_value,
    min(order_date) as first_order_date,
    max(order_date) as last_order_date
from {{ ref('fct_orders') }}
group by customer_id

