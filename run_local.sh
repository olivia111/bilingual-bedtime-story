#!/usr/bin/env bash
# Run the bilingual bedtime story app locally.
#   ./run_local.sh          -> http://127.0.0.1:8000
set -euo pipefail
cd "$(dirname "$0")"

# Render builds on 3.10; prefer a matching interpreter, but 3.9 also works.
pick_python() {
  for c in python3.13 python3.12 python3.11 python3.10 \
           /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12 \
           /opt/homebrew/bin/python3.11 /opt/homebrew/bin/python3.10 \
           /usr/local/bin/python3.13 /usr/local/bin/python3.12 \
           /usr/local/bin/python3.11 /usr/local/bin/python3.10 \
           python3; do
    if command -v "$c" >/dev/null 2>&1; then command -v "$c"; return 0; fi
    if [ -x "$c" ]; then echo "$c"; return 0; fi
  done
  return 1
}

PY="${PYTHON:-$(pick_python)}"
echo "Using $("$PY" -V) at $PY"

# Rebuild the venv if it is missing, broken, or built from a different interpreter.
NEED_VENV=1
if [ -x .venv/bin/python ] && ./.venv/bin/python -c "" 2>/dev/null; then
  have=$(./.venv/bin/python -c "import sys;print('%d.%d'%sys.version_info[:2])")
  want=$("$PY" -c "import sys;print('%d.%d'%sys.version_info[:2])")
  [ "$have" = "$want" ] && NEED_VENV=0
fi
if [ "$NEED_VENV" = 1 ]; then
  echo "Creating .venv ..."
  rm -rf .venv
  "$PY" -m venv .venv
fi

./.venv/bin/python -m pip install -q --upgrade pip
./.venv/bin/python -m pip install -q -r requirements.txt

if [ ! -f .env ]; then
  echo "No .env found. Copy .env.example to .env and add your LITHOSAI_API_KEY." >&2
  exit 1
fi

echo "Serving on http://127.0.0.1:8000  (Ctrl-C to stop)"
exec ./.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
