"""Tests for the resource lifecycle state machine and its decision helpers."""

from __future__ import annotations

import unittest

from statemachine.exceptions import TransitionNotAllowed

from resource_lifecycle import (
    ResourceLifecycle,
    drive,
    initial_plan_event,
    post_download_event,
)


class ResourceLifecycleTransitionTests(unittest.TestCase):
    def test_metadata_skip_is_terminal(self):
        m = drive(ResourceLifecycle(), "plan_metadata_skip")
        self.assertTrue(m.metadata_skipped.is_active)
        with self.assertRaises(TransitionNotAllowed):
            drive(m, "plan_download")

    def test_rehydrate_path_reaches_published(self):
        m = drive(ResourceLifecycle(), "plan_rehydrate")
        drive(m, "rehydrated")
        drive(m, "publish")
        self.assertTrue(m.published.is_active)

    def test_download_path_materializes_then_publishes(self):
        m = drive(ResourceLifecycle(), "plan_download")
        drive(m, "materialize")
        drive(m, "publish")
        self.assertTrue(m.published.is_active)

    def test_content_skip_cannot_publish(self):
        m = drive(ResourceLifecycle(), "plan_download")
        drive(m, "same_content")
        with self.assertRaises(TransitionNotAllowed):
            drive(m, "publish")

    def test_cannot_publish_straight_from_planned(self):
        with self.assertRaises(TransitionNotAllowed):
            drive(ResourceLifecycle(), "publish")

    def test_failure_allowed_from_every_active_state(self):
        for setup in (
            [],
            ["plan_rehydrate"],
            ["plan_download"],
            ["plan_download", "materialize"],
        ):
            m = ResourceLifecycle()
            for event in setup:
                drive(m, event)
            drive(m, "fail")
            self.assertTrue(m.failed.is_active, msg=str(setup))

    def test_final_states_reject_failure(self):
        m = drive(ResourceLifecycle(), "plan_metadata_skip")
        with self.assertRaises(TransitionNotAllowed):
            drive(m, "fail")


class DecisionHelperTests(unittest.TestCase):
    def test_complete_local_state_skips_metadata(self):
        event = initial_plan_event(
            force=False, fingerprint_matches=True,
            local_complete=True, local_source_present=True,
        )
        self.assertEqual(event, "plan_metadata_skip")

    def test_incomplete_local_source_rehydrates(self):
        event = initial_plan_event(
            force=False, fingerprint_matches=True,
            local_complete=False, local_source_present=True,
        )
        self.assertEqual(event, "plan_rehydrate")

    def test_fingerprint_change_downloads(self):
        event = initial_plan_event(
            force=False, fingerprint_matches=False,
            local_complete=True, local_source_present=True,
        )
        self.assertEqual(event, "plan_download")

    def test_force_always_downloads(self):
        event = initial_plan_event(
            force=True, fingerprint_matches=True,
            local_complete=True, local_source_present=True,
        )
        self.assertEqual(event, "plan_download")

    def test_same_sha_after_download_is_content_skip(self):
        self.assertEqual(
            post_download_event(force=False, same_sha256_as_previous=True),
            "same_content",
        )

    def test_force_never_content_skips(self):
        self.assertEqual(
            post_download_event(force=True, same_sha256_as_previous=True),
            "materialize",
        )


if __name__ == "__main__":
    unittest.main()
