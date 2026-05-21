#!/bin/bash
# ─────────────────────────────────────────────
#  Magnitu — Complete uninstall (macOS)
#
#  Removes app shortcuts, common clone locations, user data, and logs.
#  Does not touch Seismo or remote labels on the server.
#
#  Run:
#    bash install/uninstall_mac.sh
#
#  Or from anywhere (after downloading this file):
#    bash uninstall_mac.sh
#
#  Non-interactive (e.g. IT script):
#    bash install/uninstall_mac.sh --yes
# ─────────────────────────────────────────────

set -euo pipefail

YES=0
for arg in "$@"; do
    case "$arg" in
        --yes|-y) YES=1 ;;
        -h|--help)
            echo "Usage: bash install/uninstall_mac.sh [--yes]"
            echo "  --yes   Skip confirmation prompt"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg (try --help)" >&2
            exit 1
            ;;
    esac
done

if [[ "${OSTYPE:-}" != darwin* ]]; then
    echo "This script is for macOS only." >&2
    exit 1
fi

HOME="${HOME:-$(eval echo ~$(whoami))}"

# If run from inside a clone, remove that path too (even if non-canonical).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd)" || SCRIPT_DIR=""
REPO_FROM_SCRIPT=""
if [ -n "$SCRIPT_DIR" ] && [ -f "$SCRIPT_DIR/../main.py" ]; then
    REPO_FROM_SCRIPT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

SHORTCUTS=(
    "$HOME/Applications/Magnitu.app"
    "$HOME/Desktop/Magnitu.app"
)

CLONES=(
    "$HOME/Applications/magnitu3"
    "$HOME/magnitu"
    "$HOME/magnitu-v3"
)

DATA_DIRS=(
    "$HOME/Library/Application Support/Magnitu"
    "$HOME/Library/Logs/Magnitu"
)

_confirm_done() {
    local msg="$1"
    echo ""
    echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "   $msg"
    echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    if command -v osascript &>/dev/null; then
        osascript -e "display alert \"Magnitu\" message \"$msg\"" 2>/dev/null || true
    fi
}

echo ""
echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   Magnitu — Complete uninstall (macOS)"
echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  This will permanently delete Magnitu from this Mac:"
echo "    • App shortcuts (Applications / Desktop)"
echo "    • Local git clones"
echo "    • User data (config, database, trained models)"
echo "    • Local log files"
echo ""
echo "  Seismo (remote) labels and scores are NOT removed."
echo ""

if [ "$YES" != 1 ]; then
    read -r -p "  Type yes to start uninstall, or anything else to cancel: " CONFIRM_START
    _confirm_lc=$(printf '%s' "$CONFIRM_START" | tr '[:upper:]' '[:lower:]')
    if [ "$_confirm_lc" != "yes" ]; then
        echo ""
        echo "  Cancelled — nothing was removed."
        echo ""
        exit 0
    fi
fi

echo ""
echo "  Checking what is installed…"
echo ""

_paths_to_remove() {
    local p
    for p in "${SHORTCUTS[@]}"; do
        [ -e "$p" ] || [ -L "$p" ] && echo "    $p"
    done
    for p in "${CLONES[@]}"; do
        [ -d "$p" ] && echo "    $p/"
    done
    if [ -n "$REPO_FROM_SCRIPT" ]; then
        _seen=0
        for p in "${CLONES[@]}"; do
            [ "$(cd "$p" 2>/dev/null && pwd)" = "$REPO_FROM_SCRIPT" ] && _seen=1 && break
        done
        [ "$_seen" = 0 ] && [ -d "$REPO_FROM_SCRIPT" ] && echo "    $REPO_FROM_SCRIPT/  (this repo)"
    fi
    for p in "${DATA_DIRS[@]}"; do
        [ -e "$p" ] && echo "    $p/"
    done
}

FOUND=0
while IFS= read -r line; do
    [ -n "$line" ] && FOUND=1 && echo "$line"
done < <(_paths_to_remove)

if [ "$FOUND" = 0 ]; then
    _confirm_done "Uninstall finished. No Magnitu files were found on this Mac."
    exit 0
fi

echo ""
echo "  Stopping Magnitu if it is running…"

_stop_magnitu_processes() {
    local pat
    for pat in \
        "magnitu3/.venv" \
        "/magnitu3/main.py" \
        "/magnitu3/desktop.py" \
        "/magnitu/.venv" \
        "/magnitu/main.py" \
        "/magnitu/desktop.py" \
        "magnitu-v3/.venv" \
        "magnitu v3/.venv"
    do
        pkill -f "$pat" 2>/dev/null || true
    done
}

_stop_magnitu_processes
sleep 1

_rm_path() {
    local target="$1"
    if [ -e "$target" ] || [ -L "$target" ]; then
        rm -rf "$target"
        echo "         removed: $target"
    fi
}

echo "  Removing files…"
echo ""

for p in "${SHORTCUTS[@]}"; do
    _rm_path "$p"
done

for p in "${CLONES[@]}"; do
    _rm_path "$p"
done

if [ -n "$REPO_FROM_SCRIPT" ] && [ -d "$REPO_FROM_SCRIPT" ]; then
    _still=0
    for p in "${CLONES[@]}"; do
        [ -d "$p" ] && _still=1 && break
    done
    if [ "$_still" = 0 ]; then
        _rm_path "$REPO_FROM_SCRIPT"
    fi
fi

for p in "${DATA_DIRS[@]}"; do
    _rm_path "$p"
done

# Desktop log next to clone (harmless if clone already gone)
_rm_path "$HOME/Applications/magnitu3/.magnitu_desktop_last.log"

_confirm_done "Uninstall complete. Magnitu has been removed from this Mac."

echo "  To reinstall later:"
echo "    mkdir -p ~/Applications"
echo "    git clone https://github.com/hektopascal2026/magnitu-v3.git ~/Applications/magnitu3"
echo "    cd ~/Applications/magnitu3"
echo "    bash install/bootstrap.sh"
echo ""
