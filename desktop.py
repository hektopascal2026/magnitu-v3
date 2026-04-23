"""
Magnitu — open the app in a native desktop window (WebView) instead of a separate browser tab.

Requires: pip install -r requirements-desktop.txt

Environment (optional):
  MAGNITU_HOST   bind address (default 127.0.0.1)
  MAGNITU_PORT   preferred port (default 8000); if in use, tries the next free port
"""
import atexit
import os
import platform
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Callable, List, Optional

Proc = Any


def _python_cmd() -> List[str]:
    """Run the venv Python as arm64 on Apple Silicon (universal2 can pick x86 from Finder)."""
    exe = os.path.normpath(os.path.realpath(sys.executable))
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return ["arch", "-arm64", exe]
    return [exe]


def _repair_venv_native_wheels_if_macos(cwd: str) -> None:
    """Reinstall from requirements if pydantic/numpy (etc.) are wrong arch for this Python (mixed arm64/x86_64 venv)."""
    if sys.platform != "darwin":
        return
    probe = r"""
import importlib, sys
for name in ("pydantic_core", "numpy"):
    try:
        importlib.import_module(name)
    except Exception as e:
        if "incompatible architecture" in str(e):
            sys.exit(3)
        raise
"""
    r = subprocess.run(
        _python_cmd() + ["-c", probe],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if r.returncode == 0:
        return
    err = (r.stderr or "") + (r.stdout or "")
    if r.returncode != 3 and "incompatible architecture" not in err:
        return
    req = os.path.join(cwd, "requirements.txt")
    if os.path.isfile(req):
        print(
            "  Reinstalling Python dependencies for this Mac's architecture (one-time, may take a few minutes)...",
            file=sys.stderr,
        )
        subprocess.run(
            _python_cmd()
            + [
                "-m",
                "pip",
                "install",
                "-q",
                "--no-cache-dir",
                "--force-reinstall",
                "-r",
                req,
            ],
            cwd=cwd,
            check=False,
        )
    else:
        subprocess.run(
            _python_cmd()
            + [
                "-m",
                "pip",
                "install",
                "-q",
                "--no-cache-dir",
                "--force-reinstall",
                "numpy",
                "pydantic",
                "pydantic-core",
            ],
            cwd=cwd,
            check=False,
        )


def _server_listening(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.35):
            return True
    except OSError:
        return False


def _pick_listening_port(host: str, preferred: int) -> int:
    """Return a port that is free on host right now (best-effort)."""
    for port in range(preferred, preferred + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
            except OSError:
                continue
            return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def _wait_for_http(url: str, timeout: float = 90.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1.0)
            return True
        except (urllib.error.URLError, OSError, TimeoutError):
            time.sleep(0.25)
    return False


def main() -> None:
    try:
        import webview
    except ImportError:
        print(
            "pywebview is not installed. For a desktop window, run:\n"
            "  pip install -r requirements-desktop.txt\n",
            file=sys.stderr,
        )
        sys.exit(1)

    from config import BASE_DIR

    _repair_venv_native_wheels_if_macos(str(BASE_DIR))

    host = os.getenv("MAGNITU_HOST", "127.0.0.1")
    preferred = int(os.getenv("MAGNITU_PORT", "8000"))

    proc: Optional[Proc] = None

    def cleanup() -> None:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    if _server_listening(host, preferred):
        print(
            f"  Server already on http://{host}:{preferred}/ — opening window only.",
            file=sys.stderr,
        )
        port = preferred
    else:
        port = _pick_listening_port(host, preferred)
        if port != preferred:
            print(f"  Port {preferred} was busy — using {port}", file=sys.stderr)
        cmd: List[str] = _python_cmd() + [
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            host,
            "--port",
            str(port),
        ]
        proc = subprocess.Popen(cmd, cwd=str(BASE_DIR))
        atexit.register(cleanup)

        def _stop_and_exit(signum: int, _frame: Any) -> None:
            cleanup()
            # 128 + signal is a common convention; SIGINT -> 130
            raise SystemExit(128 + signum if signum > 0 else 0)

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _stop_and_exit)
            except (OSError, ValueError):
                pass

        if not _wait_for_http(f"http://{host}:{port}/"):
            print("  Server did not become ready in time.", file=sys.stderr)
            cleanup()
            sys.exit(1)

    url = f"http://{host}:{port}/"

    webview.create_window(
        "Magnitu",
        url,
        width=1280,
        height=840,
        min_size=(800, 600),
    )
    webview.start()
    cleanup()


if __name__ == "__main__":
    import traceback

    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        sys.exit(1)
