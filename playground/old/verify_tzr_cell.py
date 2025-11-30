#!/usr/bin/env python3
"""Verify that the TZR calculation cell is in the debug notebook"""

import json
from pathlib import Path

notebook_path = Path("residual_maker_playground_claude_debug.ipynb")

with open(notebook_path) as f:
    nb = json.load(f)

print(f"Notebook: {notebook_path}")
print(f"Total cells: {len(nb['cells'])}")
print()

# Find TZR cell
for i, cell in enumerate(nb['cells']):
    source = ''.join(cell['source'])
    if 'TZR PHASE OFFSET CALCULATION' in source:
        print(f"✓ Found TZR calculation cell at index {i}")
        print(f"  Cell has {len(cell['source'])} lines")
        print(f"  First few lines:")
        for line in cell['source'][:10]:
            print(f"    {line.rstrip()}")
        print()

        # Check if it updates the model
        if 'model = SpinDMModel(' in source:
            print("✓ Cell creates new model with updated phase_offset_cycles")
        else:
            print("✗ WARNING: Cell doesn't update model!")

        # Check for binary delay calculation
        if 'delay_roemer' in source and 'delay_shapiro' in source:
            print("✓ Cell computes binary delays (Roemer, Einstein, Shapiro)")
        else:
            print("✗ WARNING: Binary delay calculation missing!")

        # Check for DM delay
        if 'dm_delay_sec_tzr' in source:
            print("✓ Cell computes DM delay at TZR")
        else:
            print("✗ WARNING: DM delay calculation missing!")

        break
else:
    print("✗ ERROR: TZR calculation cell NOT FOUND!")
    print("\nCell headers in notebook:")
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            first_line = cell['source'][0] if cell['source'] else '(empty)'
            print(f"  {i}: {first_line[:60]}")
