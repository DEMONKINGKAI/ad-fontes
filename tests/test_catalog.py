from __future__ import annotations

from pathlib import Path

from app.catalog import load_projects

_CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus"


def test_load_projects_from_real_manifest():
    projects = load_projects(_CORPUS)
    assert len(projects) == 7
    by_id = {p.id: p for p in projects}

    assert by_id["threadfall"].name == "Threadfall: The Shattered Pact"
    assert by_id["threadfall"].repo == "https://github.com/DEMONKINGKAI/Threadfall"
    assert by_id["threadfall"].summary  # non-empty one-liner extracted from the file
    assert "interactive fiction" in by_id["threadfall"].domain

    # manifest repo field is 'null (private code; public Kaggle artifacts)'
    assert by_id["loan-approval"].repo is None


def test_projects_sorted_by_id():
    ids = [p.id for p in load_projects(_CORPUS)]
    assert ids == sorted(ids)
