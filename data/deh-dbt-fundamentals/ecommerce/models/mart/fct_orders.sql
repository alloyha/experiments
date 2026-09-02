select
    order_id,
    customer_id,
    order_date,
    status,
    amount,

    case when status = 'paid' then 1 else 0 end as is_paid,
    case when status = 'cancelled' then 1 else 0 end as is_cancelled,
    case when status = 'refunded' then 1 else 0 end as is_refunded,

    case
        when status = 'paid'
        then amount
        else 0
    end as recognized_revenue,

    case
        when status = 'refunded'
        then amount
        else 0
    end as refunded_amount,

    created_at,
    updated_at

from {{ ref('stg_orders') }}

