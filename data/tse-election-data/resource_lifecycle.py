"""Explicit lifecycle of one CKAN resource during an ingestion run.

The state machine replaces the implicit control flow that was spread across
download_resource_worker (metadata skip, rehydrate, download, content skip,
materialize) and the Phase-2 publish path. Illegal transitions raise
TransitionNotAllowed instead of silently corrupting the active state.

Decision inputs are plain booleans computed by the caller; the machine only
enforces which transitions are legal from which state.
"""

from __future__ import annotations

from statemachine import State, StateMachine
from statemachine.exceptions import TransitionNotAllowed


class ResourceLifecycle(StateMachine):
    planned = State(initial=True)
    metadata_skipped = State(final=True)
    rehydrating = State()
    downloading = State()
    content_skipped = State(final=True)
    materialized = State()
    published = State(final=True)
    failed = State(final=True)

    plan_metadata_skip = planned.to(metadata_skipped)
    plan_rehydrate = planned.to(rehydrating)
    plan_download = planned.to(downloading)

    rehydrated = rehydrating.to(materialized)
    same_content = downloading.to(content_skipped)
    materialize = downloading.to(materialized)

    publish = materialized.to(published)

    fail = (
        planned.to(failed)
        | rehydrating.to(failed)
        | downloading.to(failed)
        | materialized.to(failed)
    )


def initial_plan_event(
    *,
    force: bool,
    fingerprint_matches: bool,
    local_complete: bool,
    local_source_present: bool,
) -> str:
    """Name of the first lifecycle event for a resource.

    Order matters: a complete local state wins over rehydration, and any forced
    run or fingerprint change falls through to a download.
    """
    if not force and fingerprint_matches:
        if local_complete:
            return "plan_metadata_skip"
        if local_source_present:
            return "plan_rehydrate"
    return "plan_download"


def post_download_event(*, force: bool, same_sha256_as_previous: bool) -> str:
    if not force and same_sha256_as_previous:
        return "same_content"
    return "materialize"


def drive(machine: ResourceLifecycle, event: str) -> ResourceLifecycle:
    """Send one event by name; the library raises on an illegal transition."""
    machine.send(event)
    return machine


def fail_lifecycle(machine: ResourceLifecycle) -> str:
    """Record failure if the resource is still active; return the state it stopped in.

    Idempotent: a resource already in a final state (for example published, or
    failed by an earlier handler) keeps its state and no transition is attempted.
    """
    from_state = state_name(machine)
    try:
        machine.send("fail")
    except TransitionNotAllowed:
        pass
    return from_state


def state_name(machine: "ResourceLifecycle") -> str:
    """Name of the single active state, via the non-deprecated `configuration`."""
    return next(iter(machine.configuration)).id
