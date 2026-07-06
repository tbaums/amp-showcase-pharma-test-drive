"""Deliver a vetting result back to the caller's own destination — modeled
on `~/crewai-autodesk/src/autodesk_triage/notify.py`'s
`send_via_slack_connection`: a single-purpose relay agent calls the AMP
Slack Connection with the exact text, instructed never to edit it. The
verification checks the agent's message trace for a real tool call rather
than trusting its closing prose — an agent with no Connection attached will
happily explain why it *can't* act, which reads exactly like normal text
unless you check whether a tool actually ran.

`_tool_call_succeeded` is inlined here (not imported from `scenarios._shared`)
so the deployed artifact is self-contained — the deploy tree is only this
package, and `scenarios/_shared/` is a test-only helper that is NOT shipped
(amp-showcase #30). The shared module still backs the unit tests.
"""

from __future__ import annotations

import json
from typing import Any

from crewai import Agent

from pharma_no_code_trigger.surface import TriggerPayload


def _tool_call_succeeded(result: Any) -> tuple[bool, str]:
    """True only if a real `role="tool"` message is present AND at least one
    parsed JSON object in it has no `"error"` key. Don't trust the agent's
    prose — check whether a tool actually ran. (Inlined from the shared
    test helper so production code ships without a repo-relative import.)"""
    messages = getattr(result, "messages", None) or []
    tool_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"]
    if not tool_msgs:
        return False, f"no tool call was made — agent replied instead: {str(result)[:300]}"

    content = " ".join(str(m.get("content") or "") for m in tool_msgs)
    decoder = json.JSONDecoder()
    objects: list[Any] = []
    i, n = 0, len(content)
    while i < n:
        while i < n and content[i] in " \t\r\n":
            i += 1
        if i >= n:
            break
        try:
            obj, end = decoder.raw_decode(content, i)
            objects.append(obj)
            i = end
        except json.JSONDecodeError:
            break

    successes = [o for o in objects if isinstance(o, dict) and "error" not in o]
    if successes:
        return True, json.dumps(successes[-1])[:500]
    return False, content[:500]


def _relay_agent(llm) -> Agent:
    return Agent(
        role="Vetting Result Relay",
        goal="Deliver the exact vetting result to the caller's own channel, with zero edits.",
        backstory="A mechanical relay; never paraphrases or edits the result it's given.",
        apps=["slack/send_message"],
        llm=llm,
    )


async def deliver_via_slack_connection(payload: TriggerPayload, result_text: str, llm) -> bool:
    """Post `result_text` — this caller's own vetting result, addressed by
    name — back to `payload.destination_channel`. Returns True only if the
    platform action actually ran; never raises, so a delivery failure can't
    crash the background dispatch thread."""
    agent = _relay_agent(llm)
    message = f"<@{payload.caller_id}> {result_text}"
    try:
        agent_result = await agent.kickoff(
            messages=(
                f"Call slack/send_message with channel={payload.destination_channel!r} and "
                f"message set to EXACTLY the text below — verbatim, no edits, no added "
                f"commentary:\n\n{message}"
            )
        )
    except Exception:
        return False
    ok, _detail = _tool_call_succeeded(agent_result)
    return ok
