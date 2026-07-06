"""Trigger -> ack -> async vetting -> deliver surface (issue #9).

Slack slash commands (and simple web forms) must get a response within
Slack's hard ~3s window, or the caller sees a timeout. This module's
contract is exactly that: `handle_trigger` returns an ack IMMEDIATELY,
having only enqueued the vetting work — never having waited for it — and
the vetting result reaches the caller's own destination later,
asynchronously, via a `deliver` callback (see notify.py).

The dashboard/Slack-app registration that turns a real slash command into a
call to this module's entrypoint (and back) is a supervised, live-demo-time
setup concern — not something built or exercised autonomously here (same
standing mock-first discipline as issues #1/#4/#5/#8).
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class TriggerPayload:
    """One caller's own request — never shared/merged with another
    caller's (per-caller result legibility, issue #9 acceptance)."""

    caller_id: str
    site_name: str
    destination_channel: str


@dataclass(frozen=True)
class AckResponse:
    text: str


class VettingDispatcher(Protocol):
    """Contract: `dispatch` must return WITHOUT waiting for the vetting to
    finish — that's what makes `handle_trigger` safe to call from a
    3-second-timeout surface."""

    def dispatch(self, payload: TriggerPayload) -> str:
        """Start the vetting for `payload` in the background; return a
        request id immediately."""
        ...


def ack_text(payload: TriggerPayload) -> str:
    return f"🔍 Vetting {payload.site_name} for you — I'll post the result here shortly."


def handle_trigger(payload: TriggerPayload, dispatcher: VettingDispatcher) -> AckResponse:
    """The 3-second-safe entrypoint: validate, dispatch (fire-and-forget),
    ack. Do nothing else here — any additional work is a latent timeout bug."""
    if not payload.site_name.strip():
        return AckResponse(text="Please provide a site name.")
    dispatcher.dispatch(payload)
    return AckResponse(text=ack_text(payload))


class ThreadedVettingDispatcher:
    """Runs the vetting + delivery in a background thread so `dispatch`
    never blocks the caller.

    `run_vetting` (sync) and `deliver` (async — matches notify.py's
    `Agent.kickoff`-based delivery) are injected so tests can substitute a
    `ScriptedLLM`-backed crew run and a `FakeConnection`-backed delivery —
    see tests/test_surface.py.
    """

    def __init__(
        self,
        run_vetting: Callable[[TriggerPayload], str],
        deliver: Callable[[TriggerPayload, str], Awaitable[None]],
        max_workers: int = 4,
    ) -> None:
        self._run_vetting = run_vetting
        self._deliver = deliver
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self.dispatched: list[TriggerPayload] = []
        self._futures: list = []

    def dispatch(self, payload: TriggerPayload) -> str:
        request_id = str(uuid.uuid4())
        self.dispatched.append(payload)
        self._futures.append(self._executor.submit(self._run_and_deliver, payload))
        return request_id

    def _run_and_deliver(self, payload: TriggerPayload) -> None:
        result = self._run_vetting(payload)
        asyncio.run(self._deliver(payload, result))

    def wait_for_all(self, timeout: float | None = None) -> None:
        """Block until every dispatched request has finished — a test hook
        for deterministically asserting on post-delivery state; the real
        entrypoint (main.py) never calls this, since waiting would defeat
        the entire point of dispatching in the background."""
        for future in self._futures:
            future.result(timeout=timeout)

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)
