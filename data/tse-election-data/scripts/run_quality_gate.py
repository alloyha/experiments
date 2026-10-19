#!/usr/bin/env python3
"""Compact, evidence-preserving presentation of the existing TSE CI targets.

Only the output changes: each check remains an existing Makefile target, with
its existing failure code and operational fences. VERBOSE=1 prints its output.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TextIO

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Stage:
    title: str
    make_target: str


STAGES = (
    Stage("Testes Python", "unit"),
    Stage("DAG e linhagem", "medallion-contracts"),
    Stage("Contratos e sincronização ODCS", "data-contracts-check"),
    Stage("Compilação dbt", "compile"),
)


def _tail(output: str, limit: int = 50) -> str:
    lines = output.splitlines()
    return "\n".join(lines[-limit:])


def _summary(rows: list[tuple[Stage, str, float]], stages: tuple[Stage, ...], *, success: bool) -> str:
    result = "Aprovado" if success else "Reprovado"
    lines = ["## TSE · Verificações", "", "| Etapa | Estado | Tempo |", "|---|---|---:|"]
    for stage, state, duration in rows:
        lines.append(f"| {stage.title} | {state} | {duration:.1f}s |")
    for stage in stages[len(rows):]:
        lines.append(f"| {stage.title} | Não executado | — |")
    lines.extend(["", f"**{result}: {sum(state == 'PASS' for _, state, _ in rows)}/{len(stages)} etapas aprovadas.**", ""])
    return "\n".join(lines)


def run_checks(
    *,
    stages: tuple[Stage, ...] = STAGES,
    verbose: bool = False,
    output: TextIO = sys.stdout,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    clock: Callable[[], float] = time.monotonic,
    summary_path: Path | None = None,
    workdir: Path = ROOT,
) -> int:
    print("TSE · Verificação de qualidade", file=output, flush=True)
    rows: list[tuple[Stage, str, float]] = []
    failed = False
    for index, stage in enumerate(stages, 1):
        print(f"  [{index}/{len(stages)}] {stage.title}...", file=output, flush=True)
        start = clock()
        try:
            proc = runner(
                ["make", "--no-print-directory", "--silent", stage.make_target],
                cwd=workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                check=False,
            )
            content = proc.stdout or ""
            passed = proc.returncode == 0
            status = "PASS" if passed else "FAIL"
        except OSError as exc:
            content = str(exc)
            passed, status = False, "FAIL"
        duration = max(0.0, clock() - start)
        rows.append((stage, status, duration))
        count = re.search(r"\bRan (\d+) tests? in\b", content) if stage.make_target == "unit" else None
        suffix = f" · {count.group(1)} testes" if count and passed else ""
        print(f"    {'✓' if passed else '✗'} {status}{suffix} · {duration:.1f}s", file=output, flush=True)
        # Consumer-compatibility warnings can exit zero; show their presence
        # without dumping successful CLI transcripts into the CI channel.
        if passed and stage.make_target == "data-contracts-check":
            warnings = [line.strip() for line in content.splitlines()
                        if re.search(r"\b(?:WARN|WARNING)\b", line, re.I)]
            if warnings:
                print(f"    ⚠ {len(warnings)} aviso(s) de compatibilidade; VERBOSE=1 para detalhes", file=output)
        if verbose and content:
            print(content.rstrip(), file=output)
        if not passed:
            failed = True
            if not verbose:
                print("    Diagnóstico (últimas 50 linhas):", file=output)
                print(_tail(content) or "(sem saída; verificar dependências)", file=output)
                print("    Detalhes: make check VERBOSE=1", file=output)
            break
    passed_count = sum(status == "PASS" for _, status, _ in rows)
    print(f"{'PASS' if not failed else 'FAIL'} · {passed_count}/{len(stages)} etapas aprovadas", file=output, flush=True)
    if summary_path is not None:
        # Reporting is best-effort: failures creating a GitHub UI summary must
        # never turn a successful data-quality check into a failed check.
        try:
            with summary_path.open("a", encoding="utf-8") as handle:
                handle.write(_summary(rows, stages, success=not failed))
        except OSError as exc:
            print(f"Aviso: resumo GitHub indisponível: {exc}", file=output)
    return int(failed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="Show full command output")
    args = parser.parse_args(argv)
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    return run_checks(verbose=args.verbose, summary_path=Path(path) if path else None)


if __name__ == "__main__":
    raise SystemExit(main())
