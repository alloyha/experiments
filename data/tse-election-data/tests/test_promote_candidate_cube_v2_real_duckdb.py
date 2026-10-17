"""Real-DuckDB V2 activation/recovery rehearsals on disposable local fixtures.

This test intentionally reuses the preflight's *synthetic provenance* fixture;
only the DuckDB bytes, persisted state, locking, live smoke, and SIGKILL are real.
It does not attest real election data and NEVER opens the operator's main Gold.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import signal
import unittest

from scripts import consolidate_candidate_cube_transfer as evidence
from scripts import promote_candidate_cube_v2 as v2
from scripts.warehouse_lifecycle import lifecycle_epoch
import test_preflight_candidate_cube_promotion_v2 as fixture_support

try:
    import duckdb
except ImportError:
    duckdb = None


CYCLES = ('2018:general', '2020:municipal', '2022:general',
          '2024:municipal', '2026:general')
BOUNDARIES = ('after_epoch_rotation_intent', 'after_epoch_rotation',
              'after_epoch_rotated_event', 'after_swap_intent', 'after_swap',
              'before_live_smoke', 'after_live_smoke')


def dump(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding='utf8')


def build_gold(main: Path) -> None:
    """Construct a tiny, valid baseline DuckDB with the V1 historical registry."""
    main.unlink()
    with duckdb.connect(str(main)) as con:
        con.execute('CREATE TABLE dim_election (election_id BIGINT NOT NULL)')
        con.execute('INSERT INTO dim_election SELECT range FROM range(198)')
        con.execute('CREATE TABLE dim_election_cycle (cycle_id VARCHAR NOT NULL)')
        con.executemany('INSERT INTO dim_election_cycle VALUES (?)', [(c,) for c in CYCLES])
    assert not Path(str(main) + '.wal').exists()


def build_clone(main: Path, clone: Path) -> None:
    clone.unlink()
    shutil.copyfile(main, clone)
    with duckdb.connect(str(clone)) as con:
        con.execute('CREATE TABLE cube_candidate_performance_municipality ('
                    'cycle_id VARCHAR NOT NULL, nominal_valid_votes BIGINT NOT NULL)')
        con.executemany('INSERT INTO cube_candidate_performance_municipality VALUES (?, ?)',
                        [(c, n) for c in CYCLES for n in (1, 3, 7)])
    assert not Path(str(clone) + '.wal').exists()


def verify_gold(path: Path, *, cube: bool) -> None:
    with duckdb.connect(str(path), read_only=True) as con:
        assert con.execute('SELECT count(*) FROM dim_election').fetchone()[0] == 198
        assert con.execute('SELECT count(*) FROM dim_election_cycle').fetchone()[0] == 5
        found = con.execute("SELECT count(*) FROM information_schema.tables "
                            "WHERE table_schema='main' AND table_name="
                            "'cube_candidate_performance_municipality'").fetchone()[0]
        assert found == int(cube)
        if cube:
            rows = con.execute('SELECT cycle_id, count(*), sum(nominal_valid_votes) '
                               'FROM cube_candidate_performance_municipality '
                               'GROUP BY cycle_id ORDER BY cycle_id').fetchall()
            assert rows == sorted((c, 3, 11) for c in CYCLES)


@contextmanager
def real_v2_fixture():
    """Produce a fully pinned synthetic proof chain around two real DuckDB files."""
    f = fixture_support.CubePromotionPreflightTests(
        'test_full_legacy_anchor_and_candidate_provenance_pass_readonly')
    f.setUp()
    try:
        build_gold(f.main)
        build_clone(f.main, f.clone)

        # Rewrite every synthetic evidence link affected by replacing the DB bytes.
        journal = json.loads(f.journal.read_text())
        journal['candidate_sha256'] = evidence.digest(f.main)
        dump(f.journal, journal)

        f.manifest_data['main_sha256'] = evidence.digest(f.main)
        dump(f.manifest, f.manifest_data)

        f.stage_data['main_sha256'] = evidence.digest(f.main)
        f.stage_data['output_sha256'] = evidence.digest(f.clone)
        f.stage_data['manifest_sha256'] = evidence.digest(f.manifest)
        dump(f.stage, f.stage_data)

        f.contract_data['main_sha256'] = evidence.digest(f.main)
        f.contract_data['clone_sha256'] = evidence.digest(f.clone)
        f.contract_data['manifest_sha256'] = evidence.digest(f.manifest)
        f.contract_data['stage_report_sha256'] = evidence.digest(f.stage)
        dump(f.contracts, f.contract_data)

        approval = f.dir / 'verified-real-duckdb-approval.json'
        evidence.write_create_only(approval, f.run_gate())
        yield f, approval
    finally:
        f.doCleanups()


def die_at(expected: str):
    def hook(label: str) -> None:
        if label == expected:
            os.kill(os.getpid(), signal.SIGKILL)
    return hook


def killed_activation(main: str, boundary: str) -> None:
    v2.activate(main=Path(main), timeout=3, hook=die_at(boundary))


@unittest.skipUnless(duckdb is not None, 'requires real DuckDB Python installation')
class RealDuckDBV2RehearsalTests(unittest.TestCase):
    def test_real_file_prepare_and_activate_with_live_smoke(self):
        with real_v2_fixture() as (f, approval):
            old_hash = evidence.digest(f.main)
            old_epoch = lifecycle_epoch(f.main)
            old_journal = evidence.digest(f.journal)
            old_backup = evidence.digest(f.backup)
            candidate_hash = evidence.digest(f.clone)
            verify_gold(f.main, cube=False)
            prepared = v2.prepare(main=f.main, approval_path=approval,
                                  approval_sha=evidence.digest(approval), timeout=3)
            self.assertEqual(prepared['phase'], 'PREPARED')
            self.assertEqual(evidence.digest(f.main), old_hash)
            self.assertEqual(lifecycle_epoch(f.main), old_epoch)
            self.assertEqual(v2.status(f.main)['phase'], 'PREPARED')
            self.assertEqual(evidence.digest(Path(prepared['incoming'])), candidate_hash)
            self.assertEqual(evidence.digest(Path(prepared['backup'])), old_hash)
            activated = v2.activate(main=f.main, timeout=3)  # real DuckDB smoke
            self.assertEqual(activated['phase'], 'ACTIVE_UNCERTIFIED')
            self.assertFalse(activated['certified'])
            self.assertEqual(evidence.digest(f.main), candidate_hash)
            self.assertNotEqual(lifecycle_epoch(f.main), old_epoch)
            self.assertEqual(v2.status(f.main)['phase'], 'ACTIVE_UNCERTIFIED')
            self.assertFalse(v2.status(f.main)['recovery_required'])
            self.assertEqual(evidence.digest(f.journal), old_journal)
            self.assertEqual(evidence.digest(f.backup), old_backup)
            verify_gold(f.main, cube=True)

    def test_real_file_recovery_after_success(self):
        with real_v2_fixture() as (f, approval):
            baseline = evidence.digest(f.main)
            v2.prepare(main=f.main, approval_path=approval,
                       approval_sha=evidence.digest(approval), timeout=3)
            v2.activate(main=f.main, timeout=3)
            restored = v2.recover(main=f.main, timeout=3)
            self.assertEqual(restored['phase'], 'ROLLED_BACK')
            self.assertEqual(evidence.digest(f.main), baseline)
            self.assertEqual(v2.status(f.main)['phase'], 'ROLLED_BACK')
            self.assertFalse(v2.status(f.main)['recovery_required'])
            verify_gold(f.main, cube=False)

    def test_real_file_smoke_failure_rolls_back(self):
        with real_v2_fixture() as (f, approval):
            baseline = evidence.digest(f.main)
            v2.prepare(main=f.main, approval_path=approval,
                       approval_sha=evidence.digest(approval), timeout=3)

            def fail_after_live_smoke(db: Path, anchor: dict) -> None:
                v2.default_smoke(db, anchor)
                verify_gold(db, cube=True)
                raise ValueError('intentional failure after real live-path smoke')

            with self.assertRaisesRegex(ValueError, 'intentional failure'):
                v2.activate(main=f.main, timeout=3, validator=fail_after_live_smoke)
            self.assertEqual(evidence.digest(f.main), baseline)
            self.assertEqual(v2.status(f.main)['phase'], 'ROLLED_BACK')
            verify_gold(f.main, cube=False)

    @unittest.skipUnless(os.name == 'posix', 'requires POSIX SIGKILL and flock')
    def test_real_file_sigkill_at_each_durable_boundary(self):
        for boundary in BOUNDARIES:
            with self.subTest(boundary=boundary), real_v2_fixture() as (f, approval):
                baseline = evidence.digest(f.main)
                old_journal = evidence.digest(f.journal)
                old_backup = evidence.digest(f.backup)
                old_epoch = lifecycle_epoch(f.main)
                v2.prepare(main=f.main, approval_path=approval,
                           approval_sha=evidence.digest(approval), timeout=3)
                ctx = multiprocessing.get_context('fork')
                child = ctx.Process(target=killed_activation,
                                    args=(str(f.main), boundary))
                child.start()
                child.join(timeout=25)
                if child.is_alive():
                    child.kill()
                    child.join(timeout=5)
                    self.fail('child hung before SIGKILL injection at ' + boundary)
                self.assertEqual(child.exitcode, -signal.SIGKILL)
                self.assertTrue(v2.status(f.main)['recovery_required'])
                result = v2.recover(main=f.main, timeout=3)
                self.assertEqual(result['phase'], 'ROLLED_BACK')
                self.assertEqual(evidence.digest(f.main), baseline)
                self.assertNotEqual(lifecycle_epoch(f.main), old_epoch)
                self.assertEqual(evidence.digest(f.journal), old_journal)
                self.assertEqual(evidence.digest(f.backup), old_backup)
                verify_gold(f.main, cube=False)


if __name__ == '__main__':
    unittest.main()
