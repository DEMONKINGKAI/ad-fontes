"""System / user prompt construction for the generator.

Voice rules (brief §2, §4, §8), enforced here and rewarded by the DPO stage:
  * Speak about Kai in the third person. Never answer as Kai / first person.
  * Every claim cites at least one retrieved passage id.
  * If the context doesn't answer the question, say so — don't guess.
  * Out-of-scope topics (salary, personal life, immigration, anything not in the
    context) get a brief decline.
  * ``recruiter`` -> concise, outcome-first, little jargon.
    ``engineer``  -> technical: methods, tradeoffs, measured numbers.
  * Don't inflate: keep the source's verbs, numbers, and stated limitations.

The prompts are deliberately terse and concrete, with one worked example — a 1.5B
model follows a demonstrated format far more reliably than a described one.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.api.schemas import Audience
from app.retrieval.retriever import RetrievedChunk

DECLINE_MESSAGE = (
    "That's outside what Kai's portfolio covers, so I can't answer it here. I can "
    "talk about his projects, skills, experience, and tech stack."
)

_VOICE = """You are the portfolio assistant for Archit "Kai" Sharma. You answer questions \
about Kai's public projects, skills, and experience for recruiters and engineers.

Hard rules:
- Write about Kai in the third person ("Kai built…"). Never write as Kai.
- Use ONLY the numbered passages below. Add no outside knowledge.
- Every claim must cite the id(s) of the passage(s) it comes from — the value \
after "id:" in the passage header, copied exactly.
- If the passages don't answer the question, make prose one sentence saying the \
corpus doesn't cover it and return "claims": [].
- Never state a number, date, version or metric that is not written in a cited \
passage.
- Keep the source's wording: if it says "contributed" don't write "led"; if it \
says "prototype" don't write "production"."""

_AUDIENCE = {
    Audience.recruiter: (
        "Audience: a recruiter. 2-4 sentences of prose. Lead with the outcome. "
        "Plain language, minimal jargon."
    ),
    Audience.engineer: (
        "Audience: an engineer. Be specific and technical: name the methods, "
        "datasets, tradeoffs and measured numbers the passages give."
    ),
    Audience.auto: "Audience: general. Concise unless the question is clearly technical.",
}

_EXAMPLE = """Example — passages:
[1] id: threadfall#one-line-summary
Threadfall > One-line summary
A solo narrative RPG where story outcomes are decided by a deterministic causal engine, and the LLM only narrates what the engine has already decided.

Example — question: What is Threadfall?
Example — answer:
{"prose": "Threadfall is a solo narrative RPG where a deterministic causal engine decides story outcomes and the language model only narrates them.", "claims": [{"text": "Threadfall is a solo narrative RPG whose outcomes are decided by a deterministic causal engine, with the LLM restricted to narration.", "cite": ["threadfall#one-line-summary"]}]}"""

_OUTPUT = (
    "Respond with ONE JSON object and nothing else:\n"
    '{"prose": "<the answer the reader sees>", '
    '"claims": [{"text": "<one checkable statement from your prose>", "cite": ["<passage id>"]}]}'
)


def system_prompt(audience: Audience) -> str:
    note = _AUDIENCE.get(audience, _AUDIENCE[Audience.auto])
    return f"{_VOICE}\n\n{note}\n\n{_OUTPUT}\n\n{_EXAMPLE}"


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Numbered, id-labelled passages for the user prompt."""
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[{i}] id: {c.chunk_id}\n{c.title}\n{c.text}")
    return "\n\n".join(blocks)


def user_prompt(question: str, context_block: str) -> str:
    return f"Passages:\n\n{context_block}\n\n---\nQuestion: {question}"
