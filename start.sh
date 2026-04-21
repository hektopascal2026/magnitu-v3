#!/bin/bash
# ─────────────────────────────────────────────
#  Magnitu — Start server
#  Usage:  ~/magnitu/start.sh   (or path to this repo)
#
#  Updates: always runs `git fetch origin main`. If your checkout is on
#  branch `main`, fast-forwards with `git merge --ff-only origin/main`
#  so you get exactly what is on GitHub (no silent merge commits).
#  Fetch failures only warn (offline); merge failures on `main` abort
#  cold start so you notice divergent history.
# ─────────────────────────────────────────────

DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=8000
HOST="127.0.0.1"
URL="http://$HOST:$PORT"
PY="$DIR/.venv/bin/python"

# Out: MAG_GIT_UPDATED=1 if HEAD moved after fast-forward
mag_sync_github() {
    MAG_GIT_UPDATED=0
    if [ ! -d "$DIR/.git" ]; then
        echo "  (No .git — not a clone; skipping GitHub sync.)"
        return 0
    fi
    echo "  Syncing with GitHub (origin main)..."
    if ! git -C "$DIR" fetch origin main; then
        echo "  WARNING: git fetch failed (offline or no network). Using local tree as-is."
        return 0
    fi
    local br
    br=$(git -C "$DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
    if [ "$br" != "main" ]; then
        echo "  NOTE: Current branch is '$br', not main — fetched origin/main but did not merge."
        echo "        To run the same code as GitHub: git checkout main && git merge --ff-only origin/main"
        return 0
    fi
    local before after merge_out merge_rc
    before=$(git -C "$DIR" rev-parse HEAD)
    merge_out=$(git -C "$DIR" merge --ff-only origin/main 2>&1) || merge_rc=$?
    if [ -n "$merge_out" ]; then
        echo "$merge_out" | sed 's/^/  /'
    fi
    if [ "${merge_rc:-0}" != "0" ]; then
        echo "  ERROR: Could not fast-forward main to origin/main (local commits or conflicts)."
        echo "        Fix: cd \"$DIR\" && git status"
        return 1
    fi
    after=$(git -C "$DIR" rev-parse HEAD)
    if [ "$before" != "$after" ]; then
        MAG_GIT_UPDATED=1
        echo "  Updated to: $(git -C "$DIR" log -1 --oneline)"
    else
        echo "  Already at origin/main: $(git -C "$DIR" log -1 --oneline)"
    fi
    return 0
}

clear 2>/dev/null || true
echo ""
echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   Magnitu v3"
echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check setup
cd "$DIR" || exit 1

# Check if already running — pull and restart when new commits land
if lsof -ti:$PORT > /dev/null 2>&1; then
    NEED_RESTART=0

    if mag_sync_github; then
        if [ "$MAG_GIT_UPDATED" = "1" ]; then
            echo "  New commits pulled — restarting server..."
            NEED_RESTART=1
        fi
    else
        echo "  Git fast-forward failed — not restarting; server still on port $PORT"
    fi
    echo ""

    if [ "$NEED_RESTART" = "1" ]; then
        echo "  Stopping old server..."
        lsof -ti:$PORT | xargs kill 2>/dev/null
        sleep 1
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

# ── Auto-update (cold start) ──
DEPS_CHANGED=0
if ! mag_sync_github; then
    echo ""
    exit 1
fi
if [ "$MAG_GIT_UPDATED" = "1" ]; then
    DEPS_CHANGED=1
fi
echo ""

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
