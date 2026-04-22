Magnitu v3 – Installation
━━━━━━━━━━━━━━━━━━━━━━

Magnitu is a machine-learning relevance engine for Seismo.
It learns which news entries matter to you and highlights investigation leads.

Requirements:
  • macOS or Linux
  • Python 3.9+ and git (on macOS, Xcode Command Line Tools)

Install (from the project README):

  Recommended path (macOS double-click app, see below):
    git clone https://github.com/hektopascal2026/magnitu-v3.git ~/Applications/magnitu3
    cd ~/Applications/magnitu3

  Or any directory name (browser / Terminal only):
    git clone https://github.com/hektopascal2026/magnitu-v3.git
    cd magnitu-v3

  bash install/bootstrap.sh
  ./start.sh

  The clone directory can be any name. bootstrap.sh will:
  - create .venv and install dependencies
  - ask for your Seismo URL and Magnitu API key (magnitu_config.json)
  - test the Seismo connection
  - optionally help create a model profile or import a .magnitu file

  Full detail: see README.md → "Install from scratch (native)".

Start: ./start.sh in your clone  →  http://127.0.0.1:8000

  If you used the remote bootstrap that clones to ~/magnitu, use:
  ~/magnitu/start.sh

  macOS — Magnitu.app (desktop window, git pull on each launch)
  - Put the clone at:  ~/Applications/magnitu3
  - Run:  bash install/bootstrap.sh
  - Bootstrap places symlinks:  ~/Applications/Magnitu.app  and  ~/Desktop/Magnitu.app
    (both point at install/Magnitu.app inside the repo; updates come from git pull.)
  - Double-click Magnitu: fetches and fast-forwards main, then opens the app in a
    native window (same as start_desktop.sh / desktop.py).
  - If macOS says the app is from an unidentified developer, right-click Magnitu.app
    → Open once, or allow in System Settings → Privacy & Security.
  - If something goes wrong, an alert may appear and the last run log can open in TextEdit.
    Log file:  ~/Applications/magnitu3/.magnitu_desktop_last.log
    (or, before the clone exists:  ~/Library/Logs/Magnitu/magnitu_desktop_last.log)
  - App icon: Magnitu.icns is bundled under install/Magnitu.app/Contents/Resources.
    Edit install/magnitu_app_icon_source.svg if needed, then run:
      bash install/build_mac_icon.sh
    Finder may cache the old icon; move the app or log out/in to refresh.
  - If the clone lives elsewhere, run manually:  bash install/post_bootstrap_mac_app.sh
    only after moving the repo to ~/Applications/magnitu3, or use Terminal only.

Update:

  cd /path/to/your/magnitu-v3 && git pull

Uninstall:

  rm -rf /path/to/your/clone
