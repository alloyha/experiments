import time
from database import get_db_connection


def time_query_execution(cur, query, params):
    start = time.time()
    cur.execute(query, params)
    rows = cur.fetchall()
    elapsed = time.time() - start
    return rows, elapsed


def run_keyset_pagination(limit=100, total=100_000):
    results = []
    last_id = 0

    with get_db_connection() as cur:
        for _ in range(total // limit):
            query = "SELECT id, name, created_at FROM users WHERE id > %s ORDER BY id ASC LIMIT %s;"
            rows, elapsed = time_query_execution(cur, query, (last_id, limit))
            if not rows:
                break
            last_id = rows[-1][0]
            results.append((last_id, elapsed))

    return results

def run_range_pagination(limit=100, total=100_000):
    results = []

    with get_db_connection() as cur:
        for start_range in range(0, total, limit):
            end_range = start_range + limit
            query = "SELECT id, name, created_at FROM users WHERE id BETWEEN %s AND %s ORDER BY id;"
            _, elapsed = time_query_execution(cur, query, (start_range, end_range))
            results.append((start_range, elapsed))

    return results

def run_offset_pagination(limit=100, total=100_000):
    results = []

    with get_db_connection() as cur:
        for offset in range(0, total, limit):
            query = "SELECT id, name, created_at FROM users ORDER BY id OFFSET %s LIMIT %s;"
            _, elapsed = time_query_execution(cur, query, (offset, limit))
            results.append((offset, elapsed))

    return results


def run_timestamp_pagination(limit=100, total=100_000):
    results = []
    last_timestamp = '1970-01-01 00:00:00'

    with get_db_connection() as cur:
        for _ in range(total // limit):
            query = """
                SELECT id, name, created_at FROM users
                WHERE created_at > %s
                ORDER BY created_at ASC
                LIMIT %s;
            """
            rows, elapsed = time_query_execution(cur, query, (last_timestamp, limit))
            if not rows:
                break
            last_timestamp = rows[-1][2]  # Assuming created_at is at position 2
            results.append((last_timestamp, elapsed))

    return results

def run_window_pagination(limit=100, total=100_000):
    results = []
    offset = 0

    with get_db_connection() as cur:
        while offset < total:
            query = """
                SELECT id, name, created_at FROM users
                ORDER BY id
                LIMIT %s OFFSET %s;
            """
            rows, elapsed = time_query_execution(cur, query, (limit, offset))
            if not rows:
                break
            offset += limit
            results.append((offset, elapsed))

    return results

def run_hybrid_pagination(limit=100, total=100_000):
    results = []
    last_created_at, last_id = '1970-01-01 00:00:00', 0
    offset = 0

    with get_db_connection() as cur:
        for _ in range(total // limit):
            if offset % 2 == 0:
                query = """
                    SELECT id, name, created_at FROM users
                    WHERE (created_at, id) > (%s, %s)
                    ORDER BY created_at ASC, id ASC
                    LIMIT %s;
                """
                rows, elapsed = time_query_execution(cur, query, (last_created_at, last_id, limit))
                if not rows:
                    break
                last_created_at, last_id = rows[-1][2], rows[-1][0]
            else:
                query = """
                    SELECT id, name, created_at FROM users
                    ORDER BY id
                    LIMIT %s OFFSET %s;
                """
                rows, elapsed = time_query_execution(cur, query, (limit, offset))
                if not rows:
                    break
                offset += limit
            results.append((offset, elapsed))

    return results


benchmarks = [
    ("offset", run_offset_pagination),
    ("keyset", run_keyset_pagination),
    ("timestamp", run_timestamp_pagination),
    ("range", run_range_pagination),
    ("window", run_window_pagination),
    ("hybrid", run_hybrid_pagination),
]
