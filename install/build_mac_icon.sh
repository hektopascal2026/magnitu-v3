#!/bin/bash
# Regenerate install/Magnitu.app/Contents/Resources/Magnitu.icns from
# install/magnitu_app_icon_source.svg (macOS only; uses qlmanage, sips, iconutil).
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SVG="$SCRIPT_DIR/magnitu_app_icon_source.svg"
OUT_DIR="$SCRIPT_DIR/Magnitu.app/Contents/Resources"
OUT_ICNS="$OUT_DIR/Magnitu.icns"

if [[ "$OSTYPE" != darwin* ]]; then
  echo "This script only runs on macOS." >&2
  exit 1
fi
if [ ! -f "$SVG" ]; then
  echo "Missing: $SVG" >&2
  exit 1
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/magnitu-icns.XXXXXX")"
# iconutil requires the folder name to end with .iconset
ICONSET="$WORK/Magnitu.iconset"
mkdir -p "$ICONSET" "$OUT_DIR"

qlmanage -t -s 1024 -o "$WORK" "$SVG" >/dev/null
MASTER="$WORK/$(basename "$SVG").png"
if [ ! -f "$MASTER" ]; then
  echo "Could not rasterize SVG (qlmanage)." >&2
  rm -rf "$WORK"
  exit 1
fi

sips -z 16 16 "$MASTER" --out "$ICONSET/icon_16x16.png" >/dev/null
sips -z 32 32 "$MASTER" --out "$ICONSET/icon_16x16@2x.png" >/dev/null
sips -z 32 32 "$MASTER" --out "$ICONSET/icon_32x32.png" >/dev/null
sips -z 64 64 "$MASTER" --out "$ICONSET/icon_32x32@2x.png" >/dev/null
sips -z 128 128 "$MASTER" --out "$ICONSET/icon_128x128.png" >/dev/null
sips -z 256 256 "$MASTER" --out "$ICONSET/icon_128x128@2x.png" >/dev/null
sips -z 256 256 "$MASTER" --out "$ICONSET/icon_256x256.png" >/dev/null
sips -z 512 512 "$MASTER" --out "$ICONSET/icon_256x256@2x.png" >/dev/null
sips -z 512 512 "$MASTER" --out "$ICONSET/icon_512x512.png" >/dev/null
sips -z 1024 1024 "$MASTER" --out "$ICONSET/icon_512x512@2x.png" >/dev/null

iconutil -c icns "$ICONSET" -o "$OUT_ICNS"
rm -rf "$WORK"
echo "Wrote $OUT_ICNS"
