#!/usr/bin/env bash
# End-to-end smoke test against a running instance — hits every §5 endpoint the
# way the Vercel widget will, including a real SSE parse of POST /api/ask.
#
#   scripts/smoke_test.sh [BASE_URL]      default: http://localhost:8000
#
# Exit code is non-zero on the first hard failure. A 503 from the model paths is
# tolerated only while --allow-cold is passed (models still loading).
set -uo pipefail

BASE="${1:-http://localhost:8000}"
BASE="${BASE%/}"
ALLOW_COLD="${ALLOW_COLD:-0}"
[[ "${2:-}" == "--allow-cold" ]] && ALLOW_COLD=1

pass=0 fail=0
ok()   { printf '  \033[32mok\033[0m   %s\n' "$1"; pass=$((pass + 1)); }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; fail=$((fail + 1)); }
note() { printf '  --   %s\n' "$1"; }

need() { command -v "$1" >/dev/null || { echo "need '$1' on PATH"; exit 2; }; }
need curl
need python

# --------------------------------------------------------------------------
echo "# GET $BASE/api/health/live"
if curl -fsS --max-time 10 "$BASE/api/health/live" | grep -q '"status"'; then
  ok "liveness responds"
else
  bad "liveness probe unreachable — is the server up?"
  echo; echo "$pass passed, $fail failed"; exit 1
fi

# --------------------------------------------------------------------------
echo "# GET $BASE/api/health"
health="$(curl -fsS --max-time 15 "$BASE/api/health" || true)"
if [[ -z "$health" ]]; then
  bad "/api/health returned nothing"
else
  status="$(printf '%s' "$health" | python -c 'import sys,json;print(json.load(sys.stdin)["status"])' 2>/dev/null || echo '?')"
  case "$status" in
    ok)                 ok "models loaded (status=ok)";;
    starting|degraded)  [[ "$ALLOW_COLD" == 1 ]] && note "status=$status (cold start tolerated)" || bad "status=$status — models not ready";;
    *)                  bad "unexpected health status: $status";;
  esac
  printf '%s' "$health" | python -m json.tool
fi

# --------------------------------------------------------------------------
echo "# GET $BASE/api/projects"
if curl -fsS --max-time 15 "$BASE/api/projects" \
    | python -c 'import sys,json; d=json.load(sys.stdin); assert isinstance(d,list) and d; print("  ids:", [p["id"] for p in d])'; then
  ok "project list non-empty"
else
  bad "/api/projects malformed or empty"
fi

# --------------------------------------------------------------------------
echo "# POST $BASE/api/ask/sync"
sync_body='{"question":"What did Kai build at EffiGO?","audience":"recruiter"}'
sync_resp="$(curl -sS --max-time 120 -o /dev/null -w '%{http_code}' -X POST "$BASE/api/ask/sync" \
  -H 'content-type: application/json' -d "$sync_body" || true)"
if [[ "$sync_resp" == "200" ]]; then
  curl -fsS --max-time 120 -X POST "$BASE/api/ask/sync" -H 'content-type: application/json' -d "$sync_body" \
    | python -c 'import sys,json; d=json.load(sys.stdin); assert d.get("prose") and d.get("meta") and d.get("answer_id"); print("  claims:", len(d.get("claims",[])), "generator:", d["meta"]["generator"], "declined:", d.get("declined"))'
  ok "/api/ask/sync answered"
elif [[ "$sync_resp" == "503" && "$ALLOW_COLD" == 1 ]]; then
  note "/api/ask/sync 503 (cold start tolerated)"
else
  bad "/api/ask/sync HTTP $sync_resp"
fi

# --------------------------------------------------------------------------
echo "# POST $BASE/api/ask  (SSE)"
raw="$(curl -sS --max-time 120 -N -X POST "$BASE/api/ask" \
  -H 'content-type: application/json' \
  -d '{"question":"Which project best shows Kai'\''s RAG skills?","audience":"auto"}' || true)"

# Parse the event stream: collect the `event:` names and the final `data:` JSON.
# The raw stream comes in on argv (a heredoc would collide with a stdin pipe).
_parser="$(cat <<'PY'
import json, sys
events, last_meta, tokens, err, cur = [], None, 0, None, None
for line in sys.argv[1].splitlines():
    if line.startswith("event:"):
        cur = line.split(":", 1)[1].strip(); events.append(cur)
    elif line.startswith("data:"):
        payload = line.split(":", 1)[1].strip()
        if cur == "token":
            tokens += 1
        elif cur == "error":
            err = payload
        elif cur == "meta":
            try: last_meta = json.loads(payload)
            except Exception: pass
uniq = []
for e in events:
    if e not in uniq: uniq.append(e)
print(json.dumps({"events": uniq, "tokens": tokens, "meta": last_meta, "error": err}))
PY
)"
parsed="$(python -c "$_parser" "$raw")"
if [[ -z "$parsed" ]]; then
  bad "SSE stream produced no parseable output"
else
  echo "  $parsed"
  py_get() { printf '%s' "$parsed" | python -c "import sys,json;print(json.load(sys.stdin).get('$1'))"; }
  sse_err="$(py_get error)"
  has_done="$(printf '%s' "$parsed" | python -c "import sys,json;print('done' in json.load(sys.stdin)['events'])")"
  if [[ "$sse_err" != "None" ]]; then
    if [[ "$ALLOW_COLD" == 1 ]]; then note "SSE error event (cold start tolerated): $sse_err"; else bad "SSE error event: $sse_err"; fi
  elif [[ "$has_done" == "True" ]]; then
    ok "SSE stream reached 'done'"
  else
    bad "SSE stream never emitted 'done'"
  fi
fi

# --------------------------------------------------------------------------
echo "# POST $BASE/api/feedback"
fb='{"session_id":"smoke","question":"What did Kai build at EffiGO?","answer_id":"smoke-test","rating":"up","note":"smoke_test.sh"}'
fb_code="$(curl -sS --max-time 15 -o /dev/null -w '%{http_code}' -X POST "$BASE/api/feedback" \
  -H 'content-type: application/json' -d "$fb" || true)"
[[ "$fb_code" == "200" ]] && ok "/api/feedback accepted a rating" || bad "/api/feedback HTTP $fb_code"

# --------------------------------------------------------------------------
echo "# POST $BASE/api/ask  length cap (expect 422)"
long="$(python -c 'print("a"*5000)')"
code="$(curl -sS --max-time 15 -o /dev/null -w '%{http_code}' -X POST "$BASE/api/ask" \
  -H 'content-type: application/json' -d "{\"question\":\"$long\"}" || true)"
[[ "$code" == "422" ]] && ok "over-long question rejected (422)" || bad "length cap: expected 422, got $code"

# --------------------------------------------------------------------------
echo
echo "$pass passed, $fail failed"
[[ "$fail" == 0 ]]
