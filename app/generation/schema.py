"""The JSON contract the generator is forced to emit (brief §4).

The model must return::

    {
      "prose": "<the human-readable answer>",
      "claims": [
        {"text": "<one atomic assertion>", "cite": ["<chunk_id>", ...]},
        ...
      ]
    }

``prose`` is streamed to the client; ``claims`` is parsed at the end and fed to
the verification layers. Constraining generation this way (rather than parsing
``[1]`` markers out of free text) is what makes layer-1 structural grounding
deterministic — exactly as fons-iuris does it.

This module holds the schema in three forms so Phase 2 can pick per backend:
  * ``ANSWER_JSON_SCHEMA``  — JSON Schema for llama.cpp ``json_schema`` / for
    HF ``response_format={"type": "json_schema", ...}``;
  * ``ANSWER_GBNF``          — a GBNF grammar for llama.cpp when a raw grammar is
    preferred over schema conversion;
  * ``AnswerDraft``          — the pydantic model used to validate whatever comes
    back before verification.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

ANSWER_JSON_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["prose", "claims"],
    "properties": {
        "prose": {"type": "string", "minLength": 1, "maxLength": 4000},
        "claims": {
            "type": "array",
            "minItems": 0,
            "maxItems": 12,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "cite"],
                "properties": {
                    "text": {"type": "string", "minLength": 1, "maxLength": 500},
                    "cite": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 4,
                        "items": {"type": "string", "minLength": 1, "maxLength": 120},
                    },
                },
            },
        },
    },
}

# GBNF mirror of the schema for llama-cpp-python's ``grammar=`` path. Kept in sync
# with ANSWER_JSON_SCHEMA by tests in Phase 2.
ANSWER_GBNF = r"""
root   ::= "{" ws "\"prose\"" ws ":" ws string ws "," ws "\"claims\"" ws ":" ws claims ws "}"
claims ::= "[" ws (claim (ws "," ws claim)*)? ws "]"
claim  ::= "{" ws "\"text\"" ws ":" ws string ws "," ws "\"cite\"" ws ":" ws cites ws "}"
cites  ::= "[" ws string (ws "," ws string)* ws "]"
string ::= "\"" ([^"\\] | "\\" .)* "\""
ws     ::= [ \t\n]*
"""


class ClaimDraft(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)
    cite: list[str] = Field(..., min_length=1, max_length=4)


class AnswerDraft(BaseModel):
    """Validated generator output, pre-verification."""

    prose: str = Field(..., min_length=1, max_length=4000)
    claims: list[ClaimDraft] = Field(default_factory=list, max_length=12)
