# scripts/mutate.py

import argparse
from datetime import datetime, timedelta
import random

import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

POSTGRES_HOST=os.getenv('POSTGRES_HOST')
POSTGRES_PORT=os.getenv('POSTGRES_PORT')
POSTGRES_USER=os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD=os.getenv('POSTGRES_PASSWORD')
POSTGRES_DB=os.getenv('POSTGRES_DB')


DSN = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
print(DSN)

# ---------------------------------------------------------------------------
# Normal mutations
# ---------------------------------------------------------------------------


def insert_customer(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into customers (
                name,
                email,
                country_code
            )
            values (%s, %s, %s)
            returning customer_id
            """,
            (
                "João Santos",
                f"joao.{datetime.now().timestamp()}@example.com",
                "BR",
            ),
        )

        customer_id = cur.fetchone()[0]

    print(f"Created customer {customer_id}")


def insert_order(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into orders (
                customer_id,
                amount,
                status
            )
            values (%s, %s, %s)
            returning order_id
            """,
            (1, 299.90, "paid"),
        )

        order_id = cur.fetchone()[0]

    print(f"Created order {order_id}")


def update_customer(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            update customers
            set
                country_code = 'DE',
                updated_at = now()
            where customer_id = 1
            """
        )

    print("Customer 1 moved from BR to DE")


def update_order(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            update orders
            set
                status = 'refunded',
                updated_at = now()
            where order_id = 1
            """
        )

    print("Order 1 changed to refunded")


def delete_order(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            delete from orders
            where order_id = (
                select max(order_id)
                from orders
            )
            returning order_id
            """
        )

        row = cur.fetchone()

    if row:
        print(f"Deleted order {row[0]}")
    else:
        print("No order found")


# ---------------------------------------------------------------------------
# Bad data mutations
# ---------------------------------------------------------------------------


def bug_invalid_country(conn):
    """
    Creates a customer whose country code does not exist
    in the country_codes seed.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into customers (
                name,
                email,
                country_code
            )
            values (%s, %s, %s)
            returning customer_id
            """,
            (
                "Invalid Country",
                f"invalid-country.{datetime.now().timestamp()}@example.com",
                "XX",
            ),
        )

        customer_id = cur.fetchone()[0]

    print(
        f"BUG injected: customer {customer_id} "
        "has unknown country_code='XX'"
    )


def bug_negative_amount(conn):
    """
    Inserts an economically invalid order.
    PostgreSQL accepts it because the schema does not impose
    amount > 0.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into orders (
                customer_id,
                amount,
                status
            )
            values (%s, %s, %s)
            returning order_id
            """,
            (1, -150.00, "paid"),
        )

        order_id = cur.fetchone()[0]

    print(
        f"BUG injected: order {order_id} "
        "has negative amount=-150.00"
    )


def bug_zero_amount(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into orders (
                customer_id,
                amount,
                status
            )
            values (%s, %s, %s)
            returning order_id
            """,
            (2, 0, "paid"),
        )

        order_id = cur.fetchone()[0]

    print(
        f"BUG injected: order {order_id} "
        "has amount=0"
    )


def bug_future_order(conn):
    future_date = datetime.now() + timedelta(days=30)

    with conn.cursor() as cur:
        cur.execute(
            """
            insert into orders (
                customer_id,
                order_date,
                amount,
                status
            )
            values (%s, %s, %s, %s)
            returning order_id
            """,
            (
                3,
                future_date,
                199.90,
                "paid",
            ),
        )

        order_id = cur.fetchone()[0]

    print(
        f"BUG injected: order {order_id} "
        f"has future date {future_date.date()}"
    )


def bug_dirty_customer_name(conn):
    """
    Creates data that is structurally valid but poorly normalized.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into customers (
                name,
                email,
                country_code
            )
            values (%s, %s, %s)
            returning customer_id
            """,
            (
                "   MARIA DA SILVA   ",
                f"dirty-name.{datetime.now().timestamp()}@example.com",
                "BR",
            ),
        )

        customer_id = cur.fetchone()[0]

    print(
        f"BUG injected: customer {customer_id} "
        "has badly formatted name"
    )


def bug_duplicate_logical_email(conn):
    """
    PostgreSQL VARCHAR uniqueness is case-sensitive.

    Therefore these can coexist:

        ana@example.com
        ANA@EXAMPLE.COM

    A dbt model that normalizes emails with lower()
    can expose the logical duplicate.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            select email
            from customers
            order by customer_id
            limit 1
            """
        )

        email = cur.fetchone()[0]
        duplicate_email = email.upper()

        cur.execute(
            """
            insert into customers (
                name,
                email,
                country_code
            )
            values (%s, %s, %s)
            returning customer_id
            """,
            (
                "Duplicate Customer",
                duplicate_email,
                "BR",
            ),
        )

        customer_id = cur.fetchone()[0]

    print(
        f"BUG injected: customer {customer_id} "
        f"has logical duplicate email '{duplicate_email}'"
    )


def bug_stale_record(conn):
    stale_date = datetime.now() - timedelta(days=365)

    with conn.cursor() as cur:
        cur.execute(
            """
            update customers
            set updated_at = %s
            where customer_id = 1
            """,
            (stale_date,),
        )

    print(
        "BUG injected: customer 1 has "
        f"stale updated_at={stale_date.date()}"
    )


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


NORMAL_ACTIONS = [
    insert_order,
    update_order,
    insert_customer,
    update_customer,
]


BUG_ACTIONS = [
    bug_invalid_country,
    bug_negative_amount,
    bug_zero_amount,
    bug_future_order,
    bug_dirty_customer_name,
    bug_duplicate_logical_email,
    bug_stale_record,
]


def simulate(conn):
    action = random.choice(NORMAL_ACTIONS)

    print(f"Executing: {action.__name__}")

    action(conn)


def simulate_bug(conn):
    action = random.choice(BUG_ACTIONS)

    print(f"Injecting: {action.__name__}")

    action(conn)


ACTIONS = {
    "insert-customer": insert_customer,
    "insert-order": insert_order,
    "update-customer": update_customer,
    "update-order": update_order,
    "delete-order": delete_order,

    "bug-invalid-country": bug_invalid_country,
    "bug-negative-amount": bug_negative_amount,
    "bug-zero-amount": bug_zero_amount,
    "bug-future-order": bug_future_order,
    "bug-dirty-name": bug_dirty_customer_name,
    "bug-duplicate-email": bug_duplicate_logical_email,
    "bug-stale-record": bug_stale_record,

    "simulate": simulate,
    "simulate-bug": simulate_bug,
}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "action",
        choices=ACTIONS,
    )

    args = parser.parse_args()

    with psycopg.connect(DSN) as conn:
        ACTIONS[args.action](conn)
        conn.commit()


if __name__ == "__main__":
    main()

