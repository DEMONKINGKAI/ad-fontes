#!/usr/bin/env bash
# Build the tree a Hugging Face Space expects and wire up a 'space' git remote.
#
#   scripts/prepare_space.sh <hf-user>/<space-name>
#
# The Space's root README.md must carry the `sdk: docker` front-matter, so this
# stages deploy/README.md as ./README.md on a dedicated 'space' branch (leaving
# the repo's own README.md untouched on main). Then:
#
#   git push space space:main
set -euo pipefail

SLUG="${1:?usage: scripts/prepare_space.sh <hf-user>/<space-name>}"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

REMOTE_URL="https://huggingface.co/spaces/${SLUG}"
if git remote get-url space >/dev/null 2>&1; then
  git remote set-url space "$REMOTE_URL"
else
  git remote add space "$REMOTE_URL"
fi
echo "remote 'space' -> $REMOTE_URL"

git branch -f space HEAD
git switch space >/dev/null 2>&1 || git checkout space

cp deploy/README.md README.md
git add README.md
git commit -m "Space card (deploy/README.md as root README)" >/dev/null 2>&1 || true

cat <<EOF

'space' branch ready. Its README.md is the Space card; everything else is the
repo as-is (Dockerfile at the root is what the Space builds).

Next:
  git push space space:main          # first push
  # set secrets in the Space UI: HF_TOKEN, CORS_ORIGINS, AD_FONTES_FEEDBACK_DATASET
  # then: scripts/smoke_test.sh https://$(echo "$SLUG" | tr '/' '-').hf.space --allow-cold

Back to work:  git switch main
EOF
