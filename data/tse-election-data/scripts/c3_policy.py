"""C3 protection policy: reject ambiguous dbt publication selectors.

No parsing of dbt's full selector language is attempted: the caller must
provide an exact model allowlist whenever the publish policy is enabled.
"""
from __future__ import annotations

C3_MODELS = frozenset({'silver_tally_munzona', 'fact_tally_munzona'})
NON_WRITERS = frozenset({'deps', 'parse', 'compile', 'debug', 'ls', 'list'})


def selection(args: list[str]) -> frozenset[str]:
    if not args or args[0] not in {'run', 'build'}:
        raise ValueError('C3 controlled publication supports only dbt run/build')
    if '--full-refresh' in args or '-f' in args:
        raise ValueError('C3 full refresh requires separate certification')
    selects = []
    for flag in ('--select', '-s'):
        if flag in args:
            idx = args.index(flag)
            if idx + 1 >= len(args):
                raise ValueError('dbt selector value missing')
            selects.extend(args[idx+1].split())
    if len(selects) != 1 or selects[0] not in C3_MODELS:
        raise ValueError('C3 protected publication requires exactly one explicitly named C3 model; selectors, tags, graph operators and mixed models are denied')
    if any(token in args for token in ('--exclude', '--selector', '--indirect-selection')):
        raise ValueError('C3 policy denies alternate selector semantics')
    return frozenset(selects)


def require_protected_run(args: list[str], *, enabled: bool, configured: bool) -> None:
    if not args:
        raise ValueError('dbt command missing')
    if args[0] in NON_WRITERS:
        if enabled or configured:
            raise ValueError('C3 preflight is for dbt writes only')
        return
    # A generic run/build could transitively include C3; reject non-explicit
    # publication selection until we can resolve the dbt manifest exactly.
    if args[0] in ('run', 'build'):
        if not enabled:
            raise ValueError('Fail-closed: run/build requires an explicit C3 policy or a manifest-certified non-C3 scope')
        selection(args)
        if not configured:
            raise ValueError('C3 publication requires complete reference configuration')
    elif enabled or configured:
        raise ValueError('C3 preflight cannot certify this dbt subcommand')
