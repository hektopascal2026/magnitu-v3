#!/bin/bash
# ─────────────────────────────────────────────
#  Magnitu — Start in a native desktop window (pywebview)
#  Usage: ./start_desktop.sh
#
#  One-time: pip install -r requirements-desktop.txt
#  (Or this script will offer to install pywebview if missing.)
# ─────────────────────────────────────────────

DIR="$(cd "$(dirname "$0")" && pwd)"
PY="$DIR/.venv/bin/python"
REQ="$DIR/requirements-desktop.txt"

cd "$DIR" || exit 1

if [ ! -f "$PY" ]; then
    echo "  No .venv found. Run install/bootstrap.sh or start.sh once to create the environment."
    exit 1
fi

if [ ! -f magnitu_config.json ]; then
    echo "  No magnitu_config.json found. Run install/bootstrap.sh or start.sh once for first-time setup."
    exit 1
fi

if ! "$PY" -c "import webview" 2>/dev/null; then
    echo "  Installing desktop dependencies (pywebview)..."
    if ! "$PY" -m pip install -q -r "$REQ"; then
        echo "  ERROR: pip install -r $REQ failed."
        exit 1
    fi
    echo "  Done."
    echo ""
fi

exec "$PY" "$DIR/desktop.py"
