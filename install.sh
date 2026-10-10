#!/bin/sh
# Install web_express into its own folder, without touching the system Python packages.
#
#   sh install.sh [SOURCE]
#
# SOURCE is what pip should install: a folder containing this project (the
# default is the folder this script is in), or a git address such as
# git+https://example.org/you/web_express.git
set -eu

SOURCE="${1:-$(cd "$(dirname "$0")" && pwd)}"
HOME_DIR="${WEB_EXPRESS_HOME:-$HOME/.web_express}"
APP="$HOME_DIR/app"

PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 &&
        "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done
if [ -z "$PYTHON" ]; then
    echo "web_express needs Python 3.11 or newer. Install it from https://www.python.org/downloads/ and run this again." >&2
    exit 1
fi

echo "Installing web_express into $APP using $($PYTHON --version)"
mkdir -p "$HOME_DIR"
"$PYTHON" -m venv "$APP"
"$APP/bin/python" -m pip install --quiet --upgrade pip
"$APP/bin/python" -m pip install --quiet "$SOURCE"

"$APP/bin/web-express" setup || echo "The ranking model will be downloaded on first search instead."

cat <<DONE

web_express is installed: $APP/bin/web-express

Next, register it with your AI agents:
    $APP/bin/web-express init --dry-run     (shows what would change)
    $APP/bin/web-express init
DONE
