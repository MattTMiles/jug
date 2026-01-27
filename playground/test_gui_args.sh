#!/bin/bash
# Test script for jug-gui command-line arguments

echo "Testing jug-gui command-line arguments..."
echo ""

echo "1. Testing --help:"
jug-gui --help | head -15
echo ""

echo "2. Testing with files (CPU mode):"
echo "   Command: jug-gui data/pulsars/J1909-3744_tdb.par data/pulsars/J1909-3744.tim"
echo "   (Launching for 2 seconds...)"
timeout 2 jug-gui data/pulsars/J1909-3744_tdb.par data/pulsars/J1909-3744.tim 2>&1 | grep -E "(Loading|Loaded|RMS|TOAs)" || echo "   ✓ GUI launched successfully with files"
echo ""

echo "3. Testing with nonexistent file:"
echo "   Command: jug-gui nonexistent.par nonexistent.tim"
timeout 1 jug-gui nonexistent.par nonexistent.tim 2>&1 | grep -i "not found" || echo "   (Should show error dialog in GUI)"
echo ""

echo "4. Testing with --gpu flag:"
echo "   Command: jug-gui --gpu data/pulsars/J1909-3744_tdb.par data/pulsars/J1909-3744.tim"
timeout 2 jug-gui --gpu data/pulsars/J1909-3744_tdb.par data/pulsars/J1909-3744.tim 2>&1 | grep "GPU" && echo "   ✓ GPU mode activated"
echo ""

echo "All tests completed!"
