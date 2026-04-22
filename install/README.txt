Magnitu v3 – Installation
━━━━━━━━━━━━━━━━━━━━━━

Magnitu is a machine-learning relevance engine for Seismo.
It learns which news entries matter to you and highlights investigation leads.

Requirements:
  • macOS or Linux
  • Python 3.9+ and git (on macOS, Xcode Command Line Tools)

Install (from the project README):

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

Update:

  cd /path/to/your/magnitu-v3 && git pull

Uninstall:

  rm -rf /path/to/your/clone
