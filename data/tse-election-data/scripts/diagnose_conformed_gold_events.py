#!/usr/bin/env python3
"""Read-only coverage audit of persisted Gold event keys vs dim_election.

No Bronze/Silver CSV access, no writes to DuckDB, no invented event codes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from migrate_conformed_gold import observed_event_evidence


def diagnose(conn):
    existing = set(conn.execute('''select cast(election_year as integer),
        cast(election_type as varchar), cast(election_code as varchar)
        from main.dim_election''').fetchall())
    sources = observed_event_evidence(conn)
    missing = []
    for year, typ, code in sorted(k for k in sources if k not in existing):
        entry = sources[(year, typ, code)]
        scopes = sorted(entry['scopes'])
        expected = {'general': 'federal_state', 'municipal': 'municipal'}.get(typ)
        missing.append({
            'election_id': f'{year}:{typ}:{code}',
            'election_scope_values': scopes,
            'observed_in': sorted(entry['sources']),
            'safe_to_reconcile': len(scopes) == 1 and scopes[0] == expected,
        })
    return {
        'event_dim_count': len(existing),
        'distinct_gold_observed_event_keys': len(sources),
        'missing_event_count': len(missing),
        'safe_to_reconcile_count': sum(m['safe_to_reconcile'] for m in missing),
        'missing_events': missing,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', required=True, type=Path)
    args = parser.parse_args()
    import duckdb
    with duckdb.connect(str(args.db.expanduser().resolve(strict=True)), read_only=True) as conn:
        print(json.dumps(diagnose(conn), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
