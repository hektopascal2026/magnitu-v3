# Run the Magnitu venv's Python, forcing the arm64 slice on Apple Silicon.
# A universal2 .venv/bin/python can run as x86_64 (e.g. from Finder) while
# site-packages are arm64 from a native Terminal — use arch -arm64 to match.
# Set MAGNITU_VENV_PY to the path, or fall back to PY. Sourced on macOS only.

magnitu_venv() {
    local _py="${MAGNITU_VENV_PY:-${PY:-}}"
    if [ -z "$_py" ] || [ ! -x "$_py" ]; then
        return 127
    fi
    if [ "$(uname -m)" = "arm64" ]; then
        arch -arm64 "$_py" "$@"
    else
        "$_py" "$@"
    fi
}
