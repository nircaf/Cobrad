#!/usr/bin/env bash
# Portable install of Blender 5.2 LTS (no sudo/root needed) — extracts the
# official tarball to utils/blender-5.2/ so make_topomap.py's headless
# `blender -b -P eeg_brain_blender.py` can find a build with the
# mesh.color_attributes API (needs Blender >= 3.2; system blender here is 3.0.1).
set -euo pipefail

VERSION="${1:-5.2.0}"
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$INSTALL_DIR/blender-${VERSION}"
URL="https://download.blender.org/release/Blender${VERSION%.*}/blender-${VERSION}-linux-x64.tar.xz"
TARBALL="$INSTALL_DIR/blender-${VERSION}.tar.xz"

if [ -x "$DEST/blender" ]; then
    echo "Already installed: $DEST/blender"
    "$DEST/blender" --version
    exit 0
fi

echo "Downloading $URL"
curl -fL --progress-bar -o "$TARBALL" "$URL"

echo "Extracting to $DEST"
mkdir -p "$DEST"
tar -xf "$TARBALL" -C "$DEST" --strip-components=1
rm -f "$TARBALL"

echo "Installed. Verify:"
"$DEST/blender" --version

echo
echo "Point make_topomap.py at it with:"
echo "  export BLENDER_BIN=\"$DEST/blender\""
