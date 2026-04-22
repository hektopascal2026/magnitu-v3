# Shared: git fetch origin main + fast-forward merge when on main.
# Source after setting DIR to the repository root. Defines mag_sync_github
# and sets MAG_GIT_UPDATED=1 when HEAD moved.
# Shellcheck: use from start.sh and mac_open_desktop.sh (bash).

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
