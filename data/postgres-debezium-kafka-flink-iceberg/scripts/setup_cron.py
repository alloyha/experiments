#!/usr/bin/env python3
"""
Setup Cron
Instala/remove/lista os cron jobs que rodam o data generator (a cada 10s)
e o dbt runner (a cada 60s) como alternativa aos sidecars via Docker.
"""

import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

MARKER_GENERATOR = "# CDC-DATA-GENERATOR"
MARKER_DBT = "# CDC-DBT-RUNNER"

PYTHON = sys.executable or "python3"


def current_crontab():
    result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else ""


def write_crontab(content):
    subprocess.run(["crontab", "-"], input=content, text=True)


def build_generator_lines():
    """6 entradas por minuto (offsets de 0,10,20,30,40,50s) simulam 10s de cadência"""
    lines = []
    script = os.path.join(SCRIPT_DIR, "data_generator_once.py")
    log = os.path.join(LOG_DIR, "data_generator.log")
    for offset in (0, 10, 20, 30, 40, 50):
        lines.append(
            f"* * * * * sleep {offset}; {PYTHON} {script} >> {log} 2>&1  {MARKER_GENERATOR}"
        )
    return lines


def build_dbt_line():
    script = os.path.join(SCRIPT_DIR, "dbt_runner_once.py")
    log = os.path.join(LOG_DIR, "dbt_runner.log")
    return f"* * * * * {PYTHON} {script} >> {log} 2>&1  {MARKER_DBT}"


def install():
    os.makedirs(LOG_DIR, exist_ok=True)

    existing = current_crontab()
    existing_lines = [
        line for line in existing.splitlines()
        if MARKER_GENERATOR not in line and MARKER_DBT not in line
    ]

    new_lines = existing_lines + build_generator_lines() + [build_dbt_line()]
    write_crontab("\n".join(new_lines) + "\n")

    print("✓ Cron jobs installed successfully!\n")
    print("Installed jobs:")
    print("  - Data generator: every 10 seconds (6 entries per minute)")
    print("  - dbt runner: every minute")


def remove():
    existing = current_crontab()
    remaining = [
        line for line in existing.splitlines()
        if MARKER_GENERATOR not in line and MARKER_DBT not in line
    ]
    write_crontab("\n".join(remaining) + ("\n" if remaining else ""))
    print("✓ Cron jobs removidos")


def list_jobs():
    existing = current_crontab()
    jobs = [
        line for line in existing.splitlines()
        if MARKER_GENERATOR in line or MARKER_DBT in line
    ]
    if not jobs:
        print("Nenhum cron job do CDC Stack instalado.")
        return
    print("Cron jobs instalados:")
    for job in jobs:
        print(f"  {job}")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("install", "remove", "list"):
        print("Uso: setup_cron.py [install|remove|list]")
        sys.exit(1)

    command = sys.argv[1]
    if command == "install":
        install()
    elif command == "remove":
        remove()
    elif command == "list":
        list_jobs()


if __name__ == "__main__":
    main()
