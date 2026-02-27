#!/usr/bin/env bash
   # update_clocks.sh — Download latest clock files from IPTA repository
   # Usage: ./update_clocks.sh [target_dir]
   #   target_dir defaults to data/clock/ relative to this script's location

   set -euo pipefail

   REPO_URL="https://raw.githubusercontent.com/ipta/pulsar-clock-corrections/main/T2runtime/clock"
   SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
   CLOCK_DIR="${1:-$SCRIPT_DIR/../data/clock}"

   mkdir -p "$CLOCK_DIR"

   FILES=(
     # GPS/UTC
     gps2utc.clk
     gpst2utc_tempo2.clk
     # Effelsberg
     eff2gps.clk
     effix2gps.clk
     # LEAP (tied to Effelsberg)
     leap2effix.clk
     # Jodrell Bank
     jb2gps.clk
     jbroach2jb.clk
     jbdfb2jb.clk
     # Nançay
     ncyobs2obspm.clk
     obspm2gps.clk
     # Westerbork
     wsrt2gps.clk
     # GBT
     gbt2gps.clk
     # Arecibo
     ao2gps.clk
     # Parkes
     pks2gps.clk
     # MeerKAT
     mk2utc.clk
     mk2utc_observatory.clk
     # VLA
     vla2gps.clk
     # GMRT
     gmrt2gps.clk
     # SRT
     srt2gps.clk
     # BIPM TT(TAI)
     tai2tt_bipm2024.clk
   )

   echo "Updating clock files in $CLOCK_DIR"
   failed=0
   for f in "${FILES[@]}"; do
     printf "  %-30s " "$f"
     if curl -fsSL -o "$CLOCK_DIR/$f" "$REPO_URL/$f"; then
       echo "OK"
     else
       echo "FAILED"
       ((failed++)) || true
     fi
   done

   echo ""
   if [ "$failed" -eq 0 ]; then
     echo "All ${#FILES[@]} clock files updated successfully."
   else
     echo "Done with $failed failure(s) out of ${#FILES[@]} files."
     exit 1
   fi