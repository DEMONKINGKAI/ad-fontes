"""Generate a stratified question set from the corpus (brief §3).

Templates × personas over every project/section, plus profile / experience /
skills / cross-project / adversarial buckets. ~600 questions, deduped, stratified
by (bucket, persona); a deterministic 20% holdout (by hash of the id) is reserved
for eval and never used to build training pairs.

    python -m app.rlhf.gen_questions              # -> data/rlhf/questions.jsonl
    python -m app.rlhf.gen_questions --target 600

Deterministic given the corpus + seed.
"""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

from app.config import get_settings
from app.ingestion.loader import load_corpus
from app.rlhf._io import RLHF_DIR, write_jsonl

PERSONAS = {
    "recruiter": "{q}",
    "hr_screener": "For an initial screening call: {q}",
    "ml_lead": "As an ML lead considering Kai for a role — {q} What were the tradeoffs?",
    "skeptical_cto": "Be straight with me, no overselling: {q}",
}

# section slug -> question phrasing(s)
_SECTION_TEMPLATES = {
    "one-line-summary": ["What is {name}?", "Give me the one-line pitch for {name}."],
    "the-problem-it-solves": ["What problem does {name} solve?", "Why does {name} exist?"],
    "how-it-works": ["How does {name} work?"],
    "pipeline": ["Walk me through {name}'s pipeline."],
    "results": ["What results did {name} produce?", "How did {name} do on its evaluation?"],
    "evaluation-methodology-and-results": ["What does {name}'s evaluation show?"],
    "calibration-evaluation-full-held-out-test-split-134-529-patients": [
        "How well calibrated is {name}?"
    ],
    "key-decisions-and-why": ["What were the key design decisions in {name}, and why?"],
    "stage-by-stage-design-and-the-reasoning-behind-each-choice": [
        "How is {name} designed, stage by stage?",
        "What did {name} try and then revert?",
    ],
    "skills-demonstrated": ["What skills does {name} demonstrate?"],
    "architecture": ["What does {name}'s architecture look like?"],
    "data": ["What data does {name} use?"],
    "data-realities-faers": ["What are the data limitations in {name}?"],
    "discovery-assumptions-stated-plainly": [
        "What assumptions does {name} make, and are they met?"
    ],
    "validation-against-sider-not-drugbank": ["How was {name} validated?"],
    "multimodal-support": ["How does {name} handle multiple data modalities?"],
    "engineering-highlight-the-memory-fix": ["What was the notable engineering fix in {name}?"],
    "llm-narrator": ["What LLM does {name} use, and how is it constrained?"],
    "rag-inside-threadfall": ["How does {name} use retrieval?"],
    "the-campaign-graph": ["How big is {name}'s graph?"],
    "how-it-works-the-action-pipeline-9-steps-per-player-input": [
        "What are the steps in {name}'s per-action pipeline?"
    ],
    "model": ["What model architecture does {name} use?"],
    "nlp-intent-system-serving-time": ["How does {name} handle natural-language queries?"],
    "engineering-and-reliability-work-from-the-changelog-v0-4-v1-1": [
        "What engineering and reliability work went into {name}?"
    ],
}

_PROFILE_Q = [
    "Who is Kai and what's his background?",
    "What is Kai's core technical identity?",
    "What is Kai's design philosophy?",
    "What kind of role is Kai looking for?",
    "Where is Kai based and what's his availability?",
    "What is Kai studying?",
    "Tell me about Kai's portfolio site.",
    "What is Kai's strongest area?",
]
_EXPERIENCE_Q = [
    "What did Kai do at EffiGO Global?",
    "What did Kai build during his EffiGO internship?",
    "What did Kai work on at Axisray?",
    "Has Kai shipped ML to production, or is it all side projects?",
    "Which generative-AI models has Kai used in production?",
    "What cloud platforms has Kai used professionally?",
    "Has Kai worked with OCR or document extraction?",
]
_SKILLS = [
    (
        "causal inference and probabilistic graphical models",
        "skills#causal-inference-probabilistic-graphical-models",
    ),
    ("generative AI and NLP", "skills#generative-ai-nlp"),
    ("reinforcement learning", "skills#reinforcement-learning"),
    ("computer vision", "skills#computer-vision"),
    ("recommender systems", "skills#recommender-systems"),
    ("MLOps and deployment", "skills#mlops-engineering"),
    ("data science and analytics", "skills#data-science-analytics"),
]
_CROSS = [
    ("Which of Kai's projects use Bayesian networks?", "tech-stack-map#technology-project-map"),
    ("Which projects use a vector database?", "tech-stack-map#technology-project-map"),
    ("Where has Kai used FastAPI?", "tech-stack-map#technology-project-map"),
    ("Which projects use ChromaDB?", "tech-stack-map#technology-project-map"),
    ("Which projects involve RAG?", "tech-stack-map#technology-project-map"),
    (
        "What recurring architectural patterns show up across Kai's work?",
        "tech-stack-map#recurring-architectural-patterns",
    ),
    ("What datasets has Kai worked with?", "tech-stack-map#datasets-and-external-sources-used"),
    ("Which projects use causal-learn?", "tech-stack-map#technology-project-map"),
    ("Which of Kai's projects have a React frontend?", "tech-stack-map#technology-project-map"),
    (
        "Is there a live demo of any project I can try?",
        "recruiter-faq#is-there-a-live-demo-i-can-try",
    ),
]
# Adversarial — invite overclaiming or ask the unanswerable. `answerable` False
# means the ideal answer declines / says the corpus doesn't cover it.
_ADVERSARIAL = [
    ("Did Kai deploy {name} to production for real users?", False),
    ("How many users does {name} have?", False),
    ("Is {name} generating revenue?", False),
    ("Did Kai lead the team that built {name}?", False),
    ("Is Kai a world expert in {domain}?", False),
]
_NEGATIVE = [
    "What is Kai's expected salary?",
    "Does Kai have a partner?",
    "What is Kai's visa status?",
    "Does Kai know Rust?",
    "Has Kai worked with Kubernetes?",
    "Where did Kai grow up?",
    "What is Kai's GPA?",
    "Is Kai fluent in German?",
    "What are Kai's politics?",
    "Is Kai the best engineer you've worked with?",
]


def _slugify_name(name: str) -> str:
    return name.strip().strip('"').split(" — ")[0].split(":")[0].strip()


def _holdout(qid: str) -> bool:
    h = int(hashlib.blake2b(qid.encode(), digest_size=4).hexdigest(), 16)
    return (h % 100) < 20


def build_questions(corpus_dir: Path, target: int, seed: int = 0) -> list[dict]:
    chunks = load_corpus(corpus_dir)
    projects: dict[str, dict] = {}
    for c in chunks:
        if c.doc_type == "project" and c.project_id:
            p = projects.setdefault(
                c.project_id,
                {
                    "name": _slugify_name(c.name or c.project_id),
                    "sections": [],
                    "domain": list(c.domain),
                },
            )
            p["sections"].append(c.chunk_id.split("#", 1)[1])

    rows: list[dict] = []
    seen: set[str] = set()

    def add(
        bucket: str,
        question: str,
        persona: str,
        *,
        answerable: bool = True,
        project_id: str | None = None,
    ):
        text = PERSONAS[persona].format(q=question)
        key = text.lower()
        if key in seen:
            return
        seen.add(key)
        qid = f"{bucket}-{len(rows):04d}"
        rows.append(
            {
                "id": qid,
                "bucket": bucket,
                "persona": persona,
                "question": text,
                "answerable": answerable,
                "project_id": project_id,
                "holdout": _holdout(qid),
            }
        )

    personas = list(PERSONAS)
    rng = random.Random(seed)

    # project overview + per-section detail — each phrasing gets 2 personas
    for pid, p in sorted(projects.items()):
        for tmpl in _SECTION_TEMPLATES["one-line-summary"]:
            for persona in personas[:2]:
                add("project_overview", tmpl.format(name=p["name"]), persona, project_id=pid)
        for sec in p["sections"]:
            for tmpl in _SECTION_TEMPLATES.get(sec, []):
                for persona in rng.sample(personas, 2):
                    add("project_detail", tmpl.format(name=p["name"]), persona, project_id=pid)

    for q in _PROFILE_Q:
        for persona in personas:
            add("profile", q, persona)
    for q in _EXPERIENCE_Q:
        for persona in personas:
            add("experience", q, persona)
    for skill, _cid in _SKILLS:
        for phrasing in (f"How strong is Kai at {skill}?", f"Does Kai have {skill} experience?"):
            for persona in personas:
                add("skills", phrasing, persona)
    for q, _cid in _CROSS:
        for persona in personas:
            add("cross_project", q, persona)

    for tmpl, answerable in _ADVERSARIAL:
        for pid, p in sorted(projects.items()):
            q = tmpl.format(name=p["name"], domain=(p["domain"] or ["his field"])[0])
            for persona in rng.sample(personas, 2):
                add("adversarial", q, persona, answerable=answerable, project_id=pid)
    for q in _NEGATIVE:
        for persona in ("recruiter", "skeptical_cto"):
            add("negative_control", q, persona, answerable=False)

    rng.shuffle(rows)
    if target and len(rows) > target:
        # keep stratification: round-robin by bucket until we hit target
        by_bucket: dict[str, list] = {}
        for r in rows:
            by_bucket.setdefault(r["bucket"], []).append(r)
        picked: list[dict] = []
        while len(picked) < target and any(by_bucket.values()):
            for b in list(by_bucket):
                if by_bucket[b]:
                    picked.append(by_bucket[b].pop())
                if len(picked) >= target:
                    break
        rows = picked

    for i, r in enumerate(rows):
        r["id"] = f"{r['bucket']}-{i:04d}"
        r["holdout"] = _holdout(r["id"])
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=RLHF_DIR / "questions.jsonl")
    args = p.parse_args(argv)

    rows = build_questions(get_settings().corpus_dir, args.target, args.seed)
    write_jsonl(args.out, rows)

    from collections import Counter

    print(f"wrote {len(rows)} questions -> {args.out}")
    print("by bucket:", dict(Counter(r["bucket"] for r in rows)))
    print(
        "holdout:", sum(r["holdout"] for r in rows), "/ train:", sum(not r["holdout"] for r in rows)
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
