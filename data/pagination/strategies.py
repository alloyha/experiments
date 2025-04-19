import time
from database import get_db_connection


def time_query(cur, query, params):
    start = time.time()
    cur.execute(query, params)
    return cur.fetchall(), time.time() - start


def run_paginated(limit, total, fetch_fn):
    results = []
    total_fetched = 0
    context = {}  # shared mutable state between iterations

    with get_db_connection() as cur:
        while total_fetched < total:
            query, params, context = fetch_fn(limit, context)
            rows, elapsed = time_query(cur, query, params)
            if not rows:
                break

            total_fetched += len(rows)
            context = fetch_fn.post_process_context(rows, context)
            results.append({"elapsed_seconds": elapsed, "row_count": len(rows)})

    return results


# === Strategies ===

def offset_strategy(limit, ctx):
    offset = ctx.get("offset", 0)
    query = "SELECT id, name, created_at FROM users ORDER BY id OFFSET %s LIMIT %s;"
    return query, (offset, limit), ctx

def offset_post(rows, ctx):
    ctx["offset"] = ctx.get("offset", 0) + len(rows)
    return ctx

offset_strategy.post_process_context = offset_post


def keyset_strategy(limit, ctx):
    last_id = ctx.get("last_id", 0)
    query = "SELECT id, name, created_at FROM users WHERE id > %s ORDER BY id ASC LIMIT %s;"
    return query, (last_id, limit), ctx

def keyset_post(rows, ctx):
    if rows:
        ctx["last_id"] = rows[-1][0]  # assumes `id` is first column
    return ctx

keyset_strategy.post_process_context = keyset_post


def range_strategy(limit, ctx):
    start = ctx.get("start", 0)
    end = start + limit
    query = "SELECT id, name, created_at FROM users WHERE id BETWEEN %s AND %s ORDER BY id;"
    return query, (start, end), ctx

def range_post(rows, ctx):
    ctx["start"] = ctx.get("start", 0) + len(rows)
    return ctx

range_strategy.post_process_context = range_post


def timestamp_strategy(limit, ctx):
    ts = ctx.get("ts", '1970-01-01 00:00:00')
    query = "SELECT id, name, created_at FROM users WHERE created_at > %s ORDER BY created_at ASC LIMIT %s;"
    return query, (ts, limit), ctx

def timestamp_post(rows, ctx):
    if rows:
        ctx["ts"] = rows[-1][2]  # assumes `created_at` is third column
    return ctx

timestamp_strategy.post_process_context = timestamp_post


def hybrid_strategy(limit, ctx):
    offset = ctx.get("offset", 0)
    ts = ctx.get("ts", '1970-01-01 00:00:00')
    last_id = ctx.get("last_id", 0)

    if offset % (2 * limit) == 0:
        query = """SELECT id, name, created_at FROM users
                   WHERE (created_at, id) > (%s, %s)
                   ORDER BY created_at ASC, id ASC LIMIT %s;"""
        params = (ts, last_id, limit)
    else:
        query = "SELECT id, name, created_at FROM users ORDER BY id LIMIT %s OFFSET %s;"
        params = (limit, offset)

    return query, params, ctx

def hybrid_post(rows, ctx):
    ctx["offset"] = ctx.get("offset", 0) + len(rows)
    if rows:
        ctx["last_id"] = rows[-1][0]
        ctx["ts"] = rows[-1][2]
    return ctx

hybrid_strategy.post_process_context = hybrid_post


# === Benchmark registry ===
benchmarks = [
    ("offset", offset_strategy),
    ("keyset", keyset_strategy),
    ("timestamp", timestamp_strategy),
    ("range", range_strategy),
    ("hybrid", hybrid_strategy),
]
