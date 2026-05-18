#!/usr/bin/env sh
set -eu

# POSIX-compatible test runner: finds every Lisp file with 'test' in its name
# (case-insensitive) under numpy-speedrun-poc/, runs unit/module tests first
# and integration/mini-array tests last.
# Usage:
#   sh numpy-speedrun-poc/run-all-tests.sh
#   or: ./numpy-speedrun-poc/run-all-tests.sh

SCRIPT="$0"
# Resolve root dir (directory containing this script)
ROOT_DIR=$(cd "$(dirname "$SCRIPT")" && pwd)
SBCL="sbcl --noinform --non-interactive"

TMP_ALL="$(mktemp)"
TMP_FIRST="$(mktemp)"
TMP_LAST="$(mktemp)"
cleanup() { rm -f "$TMP_ALL" "$TMP_FIRST" "$TMP_LAST"; }
trap cleanup EXIT HUP INT TERM

# Discover tests (case-insensitive) and sort case-insensitively for stability
( cd "$ROOT_DIR" && find . -type f -iname '*test*.lisp' | sort -f ) > "$TMP_ALL"

if [ ! -s "$TMP_ALL" ]; then
  echo "No test files found under $ROOT_DIR matching *test*.lisp" >&2
  exit 1
fi

# Split into first (unit/module) and last (integration/mini-array)
# Keep original order by filtering with grep while preserving lines.
# 'last' are any paths containing 'integration' or 'mini-array'.
grep -iE 'integration|mini-array' "$TMP_ALL" > "$TMP_LAST" || true
grep -viE 'integration|mini-array' "$TMP_ALL" > "$TMP_FIRST" || true

run_one() {
  rel="$1"
  # strip leading ./ if present
  case "$rel" in ./*) rel=${rel#./} ;; esac
  abs="$ROOT_DIR/$rel"
  echo "== $rel =="
  # Load via --eval so both load-style and script-style tests work
  $SBCL \
    --eval "(load \"$abs\")" \
    --eval "(sb-ext:quit :unix-status 0)"
}

# First pass
if [ -s "$TMP_FIRST" ]; then
  while IFS= read -r t; do [ -n "$t" ] || continue; run_one "$t"; done < "$TMP_FIRST"
fi
# Last pass
if [ -s "$TMP_LAST" ]; then
  while IFS= read -r t; do [ -n "$t" ] || continue; run_one "$t"; done < "$TMP_LAST"
fi

echo "All tests passed."