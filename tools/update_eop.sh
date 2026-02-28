#!/usr/bin/env bash
# update_eop.sh — Download latest Earth Orientation Parameters from IERS
# Based on TEMPO2's T2runtime/earth/update_eop.sh
# Usage: ./update_eop.sh [target_dir]
#   target_dir defaults to data/earth/ relative to this script's location

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EOP_DIR="${1:-$SCRIPT_DIR/../data/earth}"
EOP_FILE="eopc04_IAU2000.62-now"
URL="https://hpiers.obspm.fr/iers/eop/eopc04/$EOP_FILE"

mkdir -p "$EOP_DIR"
cd "$EOP_DIR"

echo "Updating Earth Orientation Parameters in $EOP_DIR"

if [ -f "$EOP_FILE" ]; then
  echo ""
  echo "Before:"
  tail -n 3 "$EOP_FILE"
fi

echo ""
echo "Downloading from $URL ..."
if curl -fsSL -o "$EOP_FILE" "$URL"; then
  echo ""
  echo "After:"
  tail -n 3 "$EOP_FILE"
  echo ""
  echo "EOP file updated successfully."
else
  echo "FAILED to download EOP file."
  exit 1
fi
