"""AMP Flow entrypoint (issue #9) — mirrors the proven Autodesk shape
(a `Flow` subclass + a no-argument `kickoff()`), fixing the live-deploy bug
where `type="flow"` requires a real Flow, not a bare function (amp-showcase
#29). The trigger payload is read from env (with synthetic defaults) so the
deployed crew is runnable for a smoke kickoff; a real Slack slash-command or
web-form surface supplies these fields live (a supervised setup concern).

Vetting and delivery run synchronously inside the Flow step: AMP already
runs a kickoff asynchronously and exposes the result via its own status
endpoint, so the 3-second-ack + background-thread pattern used by the
non-deployed surface (surface.py) isn't needed here.
"""

from __future__ import annotations

import asyncio
import os

from crewai import LLM
from crewai.flow.flow import Flow, start

from pharma_no_code_trigger.crew import build_vetting_crew
from pharma_no_code_trigger.notify import deliver_via_slack_connection
from pharma_no_code_trigger.surface import TriggerPayload, ack_text

DEFAULT_MODEL = "anthropic/claude-haiku-4-5-20251001"


def _payload_from_env() -> TriggerPayload:
    return TriggerPayload(
        caller_id=os.environ.get("SMOKE_CALLER_ID", "live-smoke-test"),
        site_name=os.environ.get("SMOKE_SITE_NAME", "Fictia Clinical Research Center"),
        destination_channel=os.environ.get("SMOKE_SLACK_CHANNEL", ""),
    )


class TestDriveFlow(Flow):
    @start()
    def vet_and_deliver(self) -> dict:
        # SYNC step: AMP runs the kickoff in an event loop, so calling the
        # synchronous crew.kickoff() from an `async` step raises "invoked
        # synchronously from within a running event loop". Keep the step sync
        # and run the async Slack delivery in its own loop via asyncio.run().
        payload = _payload_from_env()
        result = build_vetting_crew(payload.site_name, LLM(model=DEFAULT_MODEL)).kickoff()
        result_text = result.raw
        delivered = False
        # Only attempt Slack delivery when a channel is configured; guarded so a
        # delivery failure never fails the run (the vetting + trace is the payload).
        if payload.destination_channel:
            try:
                delivered = asyncio.run(
                    deliver_via_slack_connection(payload, result_text, LLM(model=DEFAULT_MODEL))
                )
            except Exception:
                delivered = False
        return {
            "ack": ack_text(payload),
            "site": payload.site_name,
            "result": result_text,
            "delivered": delivered,
        }


def kickoff():
    """AMP deployment entrypoint (no args; mirrors Autodesk)."""
    return TestDriveFlow().kickoff()


if __name__ == "__main__":
    import json

    print(json.dumps(kickoff(), indent=2, default=str))
