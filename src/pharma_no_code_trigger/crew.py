"""The clinical-trial site vetting crew (issue #9).

One agent, one task: look up the caller's site in the synthetic vetting
data and produce a concise site-readiness summary. `llm` is injected by
the caller (`main.py` for a real deployment, `ScriptedLLM` for offline
tests — see scenarios/_shared/llm_mock.py) so this module never hard-codes
a provider.
"""

from __future__ import annotations

from crewai import Agent, Crew, Task
from crewai.tools import tool

from pharma_no_code_trigger.lookup import lookup_site


@tool("vet_site")
def vet_site_tool(site_name: str) -> str:
    """Look up a clinical-trial site in the synthetic vetting database by
    its exact name. Returns a short description of what's on file, or that
    nothing is on file for that name."""
    record = lookup_site(site_name)
    if record is None:
        return f"No vetting record on file for {site_name!r}."
    flags = record["flags"]
    flag_text = "; ".join(flags) if flags else "no flags"
    return (
        f"{site_name} — PI {record['principal_investigator']}, "
        f"{record['location']}, {record['prior_trials_completed']} prior trials completed. "
        f"Flags: {flag_text}. {record['notes']}"
    )


def build_vetting_crew(site_name: str, llm) -> Crew:
    """A crew scoped to vetting exactly one site. Built fresh per request
    (never shared/mutated across callers) so two callers' runs can never
    leak state into each other — see surface.py's per-caller isolation."""
    coordinator = Agent(
        role="Clinical Site Vetting Coordinator",
        goal=(
            f"Vet {site_name!r} using the vetting tool and write a concise, factual "
            "site-readiness summary a clinical-ops coordinator can act on."
        ),
        backstory=(
            "A meticulous site-qualification coordinator who checks candidate sites against "
            "internal records before enrollment proceeds. Reports facts plainly — never "
            "invents details the vetting tool didn't return."
        ),
        tools=[vet_site_tool],
        llm=llm,
    )
    task = Task(
        description=(
            f"Vet the site {site_name!r} using the vet_site tool, then write a 2-3 sentence "
            "readiness summary naming the site and stating whether it is clean or flagged, "
            "and why."
        ),
        expected_output=(
            "A short readiness summary that names the site and states its vetting result."
        ),
        agent=coordinator,
    )
    return Crew(agents=[coordinator], tasks=[task])
