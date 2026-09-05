#!/usr/bin/env bash
# Hit a running instance the way the Vercel widget will: SSE over plain curl.
# Usage: scripts/smoke_test.sh [BASE_URL]   (default http://localhost:8000)
set -euo pipefail

BASE="${1:-http://localhost:8000}"

echo "# GET $BASE/api/health"
curl -fsS "$BASE/api/health" | python -m json.tool

echo
echo "# GET $BASE/api/projects (ids only)"
curl -fsS "$BASE/api/projects" | python -c "import sys,json;print([p['id'] for p in json.load(sys.stdin)])"

echo
echo "# POST $BASE/api/ask/sync"
curl -fsS -X POST "$BASE/api/ask/sync" \
  -H 'content-type: application/json' \
  -d '{"question":"Which project best shows Kai'\''s RAG skills?","audience":"recruiter"}' \
  | python -m json.tool || echo "(expected 503 until Phase 2)"

echo
echo "# POST $BASE/api/ask  (SSE stream)"
curl -fsS -N -X POST "$BASE/api/ask" \
  -H 'content-type: application/json' \
  -d '{"question":"What did Kai build at EffiGO?","audience":"auto"}' \
  | sed -n '1,40p'
