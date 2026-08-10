#!/usr/bin/env python3
"""
Validate Sidecars
Verifica se a automação via cron (data generator + dbt runner) está
configurada e funcionando corretamente.
"""

import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

CHECKS = []


def check(label, fn):
    try:
        ok = fn()
    except Exception:
        ok = False
    CHECKS.append((label, ok))
    status = "PASS" if ok else "FAIL"
    icon = "✓" if ok else "✗"
    print(f"{icon} {label:<24} {status}")
    return ok


def check_python_packages():
    for module in ("psycopg2", "names", "dotenv"):
        __import__(module)
    return True


def check_postgres():
    import psycopg2
    conn = psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
        dbname=os.environ.get("POSTGRES_DB", "cdc_db"),
        connect_timeout=5,
    )
    conn.close()
    return True


def check_cron_jobs():
    result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    if result.returncode != 0:
        return False
    return "CDC-DATA-GENERATOR" in result.stdout and "CDC-DBT-RUNNER" in result.stdout


def check_log_files():
    expected = ["data_generator.log", "dbt_runner.log"]
    return all(os.path.exists(os.path.join(LOG_DIR, name)) for name in expected)


def _log_recently_updated(filename, max_age_seconds):
    path = os.path.join(LOG_DIR, filename)
    if not os.path.exists(path):
        return False
    age = time.time() - os.path.getmtime(path)
    return age <= max_age_seconds


def check_data_generator():
    return _log_recently_updated("data_generator.log", max_age_seconds=120)


def check_dbt_runner():
    return _log_recently_updated("dbt_runner.log", max_age_seconds=300)


def main():
    check("Python packages", check_python_packages)
    check("PostgreSQL", check_postgres)
    check("Cron jobs", check_cron_jobs)
    check("Log files", check_log_files)
    check("Data generator", check_data_generator)
    check("dbt runner", check_dbt_runner)

    print()
    if all(ok for _, ok in CHECKS):
        print("✓ ALL CHECKS PASSED - CDC sidecars are ready!")
        sys.exit(0)
    else:
        failed = [label for label, ok in CHECKS if not ok]
        print(f"✗ SOME CHECKS FAILED: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
