#!/usr/bin/env bash
# Download the open Metrica Sports sample games (three other matches with event data) at the
# exact revision the model was trained on.  https://github.com/metrica-sports/sample-data
set -euo pipefail
DEST="${1:-external/sample-data}"
REV="e706dd506b360d69d9d123d5b8026e7294b13996"
if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/metrica-sports/sample-data.git "$DEST"
fi
git -C "$DEST" fetch --all --quiet || true
git -C "$DEST" checkout --quiet "$REV"
echo "sample games ready in $DEST/data"
