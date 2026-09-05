"""Resolve ``audience="auto"`` to ``recruiter`` or ``engineer``.

Phase 0 heuristic: a small keyword vote. Engineer-leaning questions name methods,
architectures, metrics, or ask "how"; recruiter-leaning questions ask about fit,
availability, seniority, "best example of". Ties default to ``recruiter`` (the
primary audience for this widget).

Phase 2 may replace this with an embedding classifier if the eval shows the
heuristic mislabels; the interface (``resolve_audience``) stays the same.
"""

from __future__ import annotations

import re

from app.api.schemas import Audience

_ENGINEER_HINTS = re.compile(
    r"\b(how does|how did|architecture|implement|pipeline|latency|throughput|"
    r"embedding|vector|inference|quantiz|fine-?tun|hyperparam|benchmark|"
    r"precision|recall|f1|ndcg|hit ?rate|token|gguf|qlora|dpo|"
    r"cross-?encoder|nli|chunk|retriev|reranker|calibration|cpt|do-?calculus)\b",
    re.I,
)
_RECRUITER_HINTS = re.compile(
    r"\b(hire|hiring|fit|role|seniority|team|available|start date|notice period|"
    r"years of experience|strongest|best (example|project)|culture|salary|"
    r"relocat|remote|onsite|visa|leadership|ownership)\b",
    re.I,
)


def resolve_audience(requested: Audience, question: str) -> Audience:
    if requested is not Audience.auto:
        return requested
    eng = len(_ENGINEER_HINTS.findall(question))
    rec = len(_RECRUITER_HINTS.findall(question))
    if eng > rec:
        return Audience.engineer
    return Audience.recruiter
