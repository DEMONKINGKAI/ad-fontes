"""Deliberate faithfulness-degrading edits, for building 'rejected' candidates
(brief §3).

Each perturbation takes a *faithful* answer (``prose`` + ``claims``) and injects
exactly one of the failure modes this project targets. Perturbations are labelled
so Phase 3 can measure the judge's detection rate per type, and they are
deterministic (seeded) so the dataset is reproducible.

``apply(kind, prose, claims, rng)`` returns ``(prose', claims', applied)`` —
``applied`` is False when the edit had nothing to bite on (e.g. INFLATE_NUMBER on
a numberless answer), and the caller skips it.
"""

from __future__ import annotations

import random
import re
from enum import Enum


class PerturbationType(str, Enum):
    INFLATE_NUMBER = "inflate_number"
    UPGRADE_VERB = "upgrade_verb"
    INVENT_DEMO_URL = "invent_demo_url"
    ADD_UNSUPPORTED_TECH = "add_unsupported_tech"
    DROP_LIMITATION = "drop_limitation"
    FIRST_PERSON = "first_person"


VERB_UPGRADES: dict[str, str] = {
    r"\bcontributed to\b": "led",
    r"\bcollaborated on\b": "led",
    r"\bhelped build\b": "built",
    r"\bhelped\b": "single-handedly built",
    r"\bassisted with\b": "owned",
    r"\bworked on\b": "drove",
    r"\bprototype\b": "production system",
    r"\bexplored\b": "shipped",
    r"\ba dependency\b": "the core engine",
    r"\brecommended\b": "the industry standard",
}

_INVENT_URLS = [
    "https://{p}.kai-sharma.dev",
    "https://{p}.demo.archit.ai",
    "https://huggingface.co/spaces/DEMONKINGKAI/{p}-live",
]
_UNSUPPORTED_TECH = [
    "Kubernetes",
    "Apache Kafka",
    "Ray",
    "Kubeflow",
    "Terraform",
    "gRPC",
    "Redis",
    "PostgreSQL with pgvector",
    "Snowflake",
]
_NUM = re.compile(r"(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(%|k|K|M| tests| nodes| edges| users)?")
_LIMIT_HINT = re.compile(
    r"\b(however|but |although|limitation|caveat|not (a|fully|yet)|only|does not|didn't|"
    r"cannot|isn't a clean|partly reflects|known open)\b",
    re.I,
)


def _first_sentence_span(text: str, pattern: re.Pattern[str]) -> tuple[int, int] | None:
    for m in re.finditer(r"[^.!?]*[.!?]", text):
        if pattern.search(m.group(0)):
            return m.span()
    return None


def apply(  # noqa: PLR0911 - one branch per perturbation type is clearest
    kind: PerturbationType,
    prose: str,
    claims: list[dict],
    rng: random.Random,
    *,
    project_slug: str = "project",
) -> tuple[str, list[dict], bool]:
    claims = [dict(c) for c in claims]

    if kind is PerturbationType.INFLATE_NUMBER:
        m = _NUM.search(prose)
        if not m:
            return prose, claims, False
        raw = m.group(1).replace(",", "")
        try:
            val = float(raw)
        except ValueError:
            return prose, claims, False
        suffix = m.group(2) or ""
        if suffix == "%":
            new = f"over {min(99.9, val * 1.2):.0f}%"
        else:
            new = f"more than {int(val * 2)}{suffix}"
        new_prose = prose[: m.start()] + new + prose[m.end() :]
        return new_prose, claims, new_prose != prose

    if kind is PerturbationType.UPGRADE_VERB:
        for pat, repl in VERB_UPGRADES.items():
            if re.search(pat, prose, re.I):
                new_prose = re.sub(pat, repl, prose, count=1, flags=re.I)
                for c in claims:
                    c["text"] = re.sub(pat, repl, c["text"], count=1, flags=re.I)
                return new_prose, claims, True
        return prose, claims, False

    if kind is PerturbationType.INVENT_DEMO_URL:
        url = rng.choice(_INVENT_URLS).format(p=project_slug)
        add = f" A live demo is available at {url}."
        return prose.rstrip() + add, claims, True

    if kind is PerturbationType.ADD_UNSUPPORTED_TECH:
        tech = rng.choice(_UNSUPPORTED_TECH)
        add = f" The system also uses {tech} in production."
        return prose.rstrip() + add, claims, True

    if kind is PerturbationType.DROP_LIMITATION:
        span = _first_sentence_span(prose, _LIMIT_HINT)
        if span is None:
            return prose, claims, False
        new_prose = (prose[: span[0]] + prose[span[1] :]).strip()
        new_prose = re.sub(r"\s{2,}", " ", new_prose)
        return new_prose, claims, new_prose != prose

    if kind is PerturbationType.FIRST_PERSON:
        new_prose = re.sub(
            r"\bKai(?:'s)?\b", lambda m: "my" if m.group(0).endswith("'s") else "I", prose
        )
        new_prose = new_prose.replace("I is ", "I am ").replace("I has ", "I have ")
        new_prose = re.sub(r"\bI (built|led|drove|shipped|owned)\b", r"I \1", new_prose)
        if new_prose == prose:
            return prose, claims, False
        for c in claims:
            c["text"] = re.sub(r"\bKai\b", "I", c["text"])
        return new_prose, claims, True

    return prose, claims, False
