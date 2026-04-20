#!/bin/bash
# ─────────────────────────────────────────────
#  Magnitu — Start server
#  Usage:  ~/magnitu/start.sh
# ─────────────────────────────────────────────

DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=8000
HOST="127.0.0.1"
URL="http://$HOST:$PORT"
PY="$DIR/.venv/bin/python"

clear 2>/dev/null || true
echo ""
echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   Magnitu 2"
echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check setup
cd "$DIR" || exit 1

# Check if already running — always pull and restart to pick up any updates
if lsof -ti:$PORT > /dev/null 2>&1; then
    NEED_RESTART=0

    if [ -d "$DIR/.git" ]; then
        echo "  Checking for updates..."
        BEFORE=$(git -C "$DIR" rev-parse HEAD 2>/dev/null)
        git -C "$DIR" pull origin main 2>&1 | sed 's/^/  /'
        AFTER=$(git -C "$DIR" rev-parse HEAD 2>/dev/null)
        if [ "$BEFORE" != "$AFTER" ]; then
            echo "  New version pulled — restarting server..."
            NEED_RESTART=1
        else
            echo "  Already on latest version."
        fi
        echo ""
    fi

    if [ "$NEED_RESTART" = "1" ]; then
        # Kill the old server and fall through to a fresh start
        echo "  Stopping old server..."
        lsof -ti:$PORT | xargs kill 2>/dev/null
        sleep 1
        # Force kill if still alive
        lsof -ti:$PORT | xargs kill -9 2>/dev/null 2>&1 || true
        sleep 1
    else
        echo "  Already running at $URL"
        echo "  Opening browser..."
        open "$URL" 2>/dev/null || echo "  Open $URL in your browser."
        echo ""
        echo "  To stop: close this window or press Ctrl+C"
        read -r -p "  Press Enter to stop the server... "
        lsof -ti:$PORT | xargs kill 2>/dev/null
        exit 0
    fi
fi
if [ ! -f "$PY" ]; then
    echo "  Not set up yet. Running installer..."
    echo ""
    /bin/bash "$DIR/install/bootstrap.sh"
    exit $?
fi

if [ ! -f magnitu_config.json ]; then
    echo "  No config found. Running installer..."
    echo ""
    /bin/bash "$DIR/install/bootstrap.sh"
    exit $?
fi

# ── Auto-update ──
DEPS_CHANGED=0
if [ -d "$DIR/.git" ]; then
    echo "  Checking for updates..."
    BEFORE=$(git -C "$DIR" rev-parse HEAD 2>/dev/null)
    git -C "$DIR" pull -q origin main 2>/dev/null || true
    AFTER=$(git -C "$DIR" rev-parse HEAD 2>/dev/null)
    if [ "$BEFORE" != "$AFTER" ]; then
        echo "  Updated to latest version."
        DEPS_CHANGED=1
    else
        echo "  Already up to date."
    fi
    echo ""
fi

# ── Dependency check ──
# Use a stamp file to skip the slow import check (~5s for torch) when
# requirements.txt hasn't changed since the last successful check.
STAMP="$DIR/.venv/.deps_ok"
REQ="$DIR/requirements.txt"
NEED_CHECK=0

if [ "$DEPS_CHANGED" = "1" ]; then
    NEED_CHECK=1
elif [ ! -f "$STAMP" ]; then
    NEED_CHECK=1
elif [ "$REQ" -nt "$STAMP" ]; then
    NEED_CHECK=1
fi

if [ "$NEED_CHECK" = "1" ]; then
    echo "  Checking dependencies..."
    MISSING=$("$PY" -c "
missing = []
for mod in ['uvicorn', 'fastapi', 'httpx', 'sklearn', 'torch', 'transformers']:
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)
print(','.join(missing))
" 2>/dev/null)

    if [ -n "$MISSING" ] || [ "$DEPS_CHANGED" = "1" ]; then
        if [ -n "$MISSING" ]; then
            echo "  Missing packages: $MISSING"
        fi
        echo "  Installing dependencies (this may take a few minutes)..."
        if ! "$PY" -m pip install -q -r "$REQ" 2>&1; then
            echo ""
            echo "  ERROR: pip install failed. Check your internet connection"
            echo "  and try running manually:"
            echo "    $PY -m pip install -r $REQ"
            echo ""
            exit 1
        fi

        # Verify again after install
        STILL_MISSING=$("$PY" -c "
missing = []
for mod in ['uvicorn', 'fastapi', 'httpx', 'sklearn', 'torch', 'transformers']:
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)
print(','.join(missing))
" 2>/dev/null)

        if [ -n "$STILL_MISSING" ]; then
            echo ""
            echo "  ERROR: These packages failed to install: $STILL_MISSING"
            echo "  Try installing manually:"
            echo "    $PY -m pip install $STILL_MISSING"
            echo ""
            exit 1
        fi
    fi
    # Mark deps as verified
    touch "$STAMP"
    echo "  All dependencies OK."
else
    echo "  Dependencies OK (cached)."
fi
echo ""

echo "  Starting on $URL ..."
echo "  Press Ctrl+C to stop."
echo ""

# Wait for server to be ready, then open browser (background)
(
    for i in $(seq 1 30); do
        sleep 1
        if curl -s -o /dev/null -w '' "http://$HOST:$PORT" 2>/dev/null; then
            open "http://$HOST:$PORT" 2>/dev/null
            break
        fi
    done
) &

# Run server (foreground — Ctrl+C stops it)
"$PY" -m uvicorn main:app --host "$HOST" --port $PORT

echo ""
echo "  Magnitu stopped."
