"""Deliberate faithfulness-degrading edits, for building 'rejected' candidates.

Each perturbation takes a faithful answer and introduces exactly one of the
failure modes this project targets (brief §3). Perturbations are *labelled* so
Phase 3 can measure the judge's detection rate per type.

Implementation lands in Phase 3; the catalogue below is the contract.
"""

from __future__ import annotations

from enum import Enum


class PerturbationType(str, Enum):
    INFLATE_NUMBER = "inflate_number"  # 97.2% -> "over 99%"
    UPGRADE_VERB = "upgrade_verb"  # contributed -> led; prototype -> production
    INVENT_DEMO_URL = "invent_demo_url"  # add a plausible but non-existent live link
    ADD_UNSUPPORTED_TECH = "add_unsupported_tech"  # claim a tech not in the cited chunk
    DROP_LIMITATION = "drop_limitation"  # remove a stated caveat / known limitation
    FIRST_PERSON = "first_person"  # rewrite into Kai's first-person voice


VERB_UPGRADES: dict[str, str] = {
    "contributed to": "led",
    "helped build": "built",
    "prototype": "production system",
    "explored": "shipped",
    "assisted with": "owned",
    "worked on": "drove",
}
