#!/bin/bash
# Health check for dbt loop containers (dbt-flink-ingestao, dbt-duckdb-medallhao).
# The main process (PID 1) is the long-running bash loop invoking dbt run/test
# on a cycle; if it has died or been replaced, cmdline will no longer show it.
grep -q "dbt" /proc/1/cmdline 2>/dev/null
