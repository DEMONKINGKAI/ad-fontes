"""Parse corpus Markdown into citable chunks.

Chunking rules (corpus README + brief §3):

  * **Project / profile / skills / experience files** — split on ``##`` headings.
    Prose that appears before the first ``##`` (e.g. the intro paragraph under
    ``# Who Kai is``) becomes its own chunk keyed to the ``#`` title.
  * **``faq/recruiter-faq.md``** — one chunk per ``**Q: ...**`` block. The FAQ has
    no ``##`` sections; each Q&A is already answer-shaped and independently
    citable.
  * **``stack/tech-stack-map.md``** — one chunk per ``#`` section (it uses single
    ``#`` for its three parts, not ``##``).

Every chunk gets:

  * ``chunk_id`` — ``<file-stem>#<section-slug>``, deterministic. Slug collisions
    inside a file get a ``-2`` / ``-3`` suffix.
  * ``text`` — the pure citable body, no heading, no breadcrumb. This is what a
    citation quotes and what the NLI layer uses as its premise.
  * ``embed_text`` — ``"<name> › <section>\\n<text>"``. The breadcrumb goes only
    into the vector, never into the citable text (mirrors fons-iuris: operative
    text alone has little lexical overlap with how questions are phrased).
  * frontmatter (``doc_type``, ``project_id``, ``repo``, ``stack``, ``domain`` …)
    carried as chunk metadata.

No torch / chromadb imports here — determinism is unit-tested in CI without them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import frontmatter

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_H1_RE = re.compile(r"^#\s+(?P<title>.+?)\s*$", re.M)
_H2_RE = re.compile(r"^##\s+(?P<title>.+?)\s*$", re.M)
_H1_SPLIT_RE = re.compile(r"^#\s+(?P<title>.+?)\s*$", re.M)
_FAQ_Q_RE = re.compile(r"^\*\*Q:\s*(?P<q>.+?)\*\*\s*$", re.M)

_MAX_SLUG = 80

# Friendly display names for the breadcrumb when a file has no frontmatter ``name``.
_DISPLAY_NAME: dict[str, str] = {
    "kai-profile": "Kai Sharma",
    "experience": "Experience",
    "skills": "Skills & expertise",
    "recruiter-faq": "Recruiter FAQ",
    "tech-stack-map": "Tech stack map",
}

# Files that are ingestion notes / indexes, not portfolio content.
_SKIP_NAMES = {"readme.md"}


@dataclass(slots=True, frozen=True)
class Chunk:
    """One citable unit of the corpus."""

    chunk_id: str
    text: str
    """Pure citable text — no breadcrumb, no heading."""
    embed_text: str
    """What actually gets embedded: ``breadcrumb + '\\n' + text``."""
    title: str
    """Breadcrumb: ``<name> › <section>``."""
    section: str
    doc_type: str
    source_path: str
    project_id: str | None = None
    repo_url: str | None = None
    name: str | None = None
    stack: tuple[str, ...] = ()
    domain: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)

    def to_chroma_metadata(self) -> dict[str, str]:
        """Chroma metadata must be scalar — lists are pipe-joined and pipe-fenced
        (``|python|fastapi|``) so a substring test can match a whole token."""
        return {
            "chunk_id": self.chunk_id,
            "doc_type": self.doc_type,
            "project_id": self.project_id or "",
            "name": self.name or "",
            "section": self.section,
            "title": self.title,
            "source_path": self.source_path,
            "repo_url": self.repo_url or "",
            "stack": _fence(self.stack),
            "domain": _fence(self.domain),
        }


def slugify(text: str, *, max_len: int = _MAX_SLUG) -> str:
    """Lowercase, non-alphanumerics to single hyphens, trimmed, length-capped."""
    slug = _SLUG_RE.sub("-", text.strip().lower()).strip("-")
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("-")
    return slug


def _fence(items: tuple[str, ...]) -> str:
    if not items:
        return ""
    return "|" + "|".join(i.strip().lower() for i in items) + "|"


def _clean_name(raw: object) -> str | None:
    if not raw:
        return None
    return str(raw).strip().strip('"').strip() or None


def _as_tuple(raw: object) -> tuple[str, ...]:
    if isinstance(raw, list | tuple):
        return tuple(str(x).strip() for x in raw if str(x).strip())
    if isinstance(raw, str) and raw.strip():
        return (raw.strip(),)
    return ()


class _IdFactory:
    """Hands out ``<stem>#<slug>`` ids, de-duplicating within a file."""

    def __init__(self, stem: str) -> None:
        self._stem = stem
        self._seen: dict[str, int] = {}

    def make(self, section: str) -> str:
        base = slugify(section) or "section"
        n = self._seen.get(base, 0) + 1
        self._seen[base] = n
        suffix = "" if n == 1 else f"-{n}"
        return f"{self._stem}#{base}{suffix}"


def _strip_h1(text: str) -> str:
    """Drop a leading ``# Title`` line so it doesn't leak into the preamble chunk."""
    return _H1_RE.sub("", text, count=1).strip()


def _split_sections(body: str) -> list[tuple[str, str]]:
    """Return ``[(heading, text), ...]`` splitting on ``##``. Prose before the first
    ``##`` (minus the document ``# H1`` line) is returned under the empty heading."""
    matches = list(_H2_RE.finditer(body))
    out: list[tuple[str, str]] = []
    preamble = _strip_h1(body[: matches[0].start()] if matches else body)
    if preamble:
        out.append(("", preamble))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        out.append((m.group("title").strip(), body[start:end].strip()))
    return out


def _split_hash_sections(body: str) -> list[tuple[str, str]]:
    """Like ``_split_sections`` but on single ``#`` — for the stack map."""
    matches = list(_H1_SPLIT_RE.finditer(body))
    out: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        out.append((m.group("title").strip(), body[start:end].strip()))
    return out


def _chunk_faq(post: frontmatter.Post, stem: str, rel: str, name: str) -> list[Chunk]:
    body = post.content
    marks = list(_FAQ_Q_RE.finditer(body))
    ids = _IdFactory(stem)
    chunks: list[Chunk] = []
    for i, m in enumerate(marks):
        question = m.group("q").strip()
        start = m.end()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        answer = body[start:end].strip()
        text = f"Q: {question}\n{answer}".strip()
        section = question.rstrip("?")
        title = f"{name} › {section}"
        chunks.append(
            Chunk(
                chunk_id=ids.make(question),
                text=text,
                embed_text=f"{title}\n{text}",
                title=title,
                section=section,
                doc_type="faq",
                source_path=rel,
                name=name,
            )
        )
    return chunks


def _chunk_standard(
    post: frontmatter.Post,
    stem: str,
    rel: str,
    name: str,
    *,
    hash_level: int,
) -> list[Chunk]:
    meta = post.metadata
    doc_type = str(meta.get("doc_type", "")) or "document"
    project_id = _clean_name(meta.get("project_id"))
    repo = _clean_name(meta.get("repo"))
    if repo and repo.lower().startswith("null"):
        repo = None
    stack = _as_tuple(meta.get("stack"))
    domain = _as_tuple(meta.get("domain"))

    body = post.content
    h1 = _H1_RE.search(body)
    h1_title = h1.group("title").strip() if h1 else name

    sections = _split_hash_sections(body) if hash_level == 1 else _split_sections(body)
    ids = _IdFactory(stem)
    chunks: list[Chunk] = []
    for heading, text in sections:
        if not text.strip():
            continue
        section = heading or h1_title
        title = f"{name} › {section}"
        chunks.append(
            Chunk(
                chunk_id=ids.make(section),
                text=text.strip(),
                embed_text=f"{title}\n{text.strip()}",
                title=title,
                section=section,
                doc_type=doc_type,
                source_path=rel,
                project_id=project_id,
                repo_url=repo,
                name=name,
                stack=stack,
                domain=domain,
            )
        )
    return chunks


def chunk_markdown(path: Path, corpus_dir: Path) -> list[Chunk]:
    """Chunk a single corpus Markdown file into an ordered list of chunks."""
    post = frontmatter.loads(path.read_text(encoding="utf-8"))
    stem = path.stem
    rel = path.relative_to(corpus_dir).as_posix()
    name = _clean_name(post.metadata.get("name")) or _DISPLAY_NAME.get(
        stem, stem.replace("-", " ").title()
    )
    doc_type = str(post.metadata.get("doc_type", ""))

    if doc_type == "faq" or stem == "recruiter-faq":
        return _chunk_faq(post, stem, rel, name)
    if doc_type == "stack_map" or stem == "tech-stack-map":
        return _chunk_standard(post, stem, rel, name, hash_level=1)
    return _chunk_standard(post, stem, rel, name, hash_level=2)


def load_corpus(corpus_dir: Path) -> list[Chunk]:
    """Walk ``corpus_dir`` and return every chunk in a stable, sorted order."""
    corpus_dir = Path(corpus_dir)
    files = sorted(p for p in corpus_dir.rglob("*.md") if p.name.lower() not in _SKIP_NAMES)
    chunks: list[Chunk] = []
    for path in files:
        chunks.extend(chunk_markdown(path, corpus_dir))
    return chunks
