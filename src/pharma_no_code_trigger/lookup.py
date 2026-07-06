"""Synthetic clinical-trial site vetting lookup (issue #9).

All sites, investigators, and sponsors are invented and unmistakably
fictional (CONTRIBUTING.md) — no real trials, sites, investigators, drugs,
or patients. The data ships as **package data** inside this package
(`data/sites.json`) and is loaded via `importlib.resources`, so it resolves
identically whether the module runs from source, from the compiler's copied
demo-checkpoint tree, or from an installed wheel on AMP (amp-showcase #31 —
a source-relative path breaks once the package is installed).
"""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any


def _load_sites() -> dict[str, Any]:
    data = files("pharma_no_code_trigger").joinpath("data/sites.json").read_text()
    return json.loads(data)


def lookup_site(name: str) -> dict[str, Any] | None:
    """The synthetic vetting record for `name`, or None if it's not on file
    — never raises on an unknown name, since "no record found" is a normal,
    expected outcome for a caller-supplied site name."""
    return _load_sites().get(name)


def known_site_names() -> list[str]:
    return sorted(_load_sites())
