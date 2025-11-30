"""Check if clock files cover the full data range"""
import numpy as np

# Get data range
parfile = '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744_tdb.par'
timfile = '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744.tim'

# Parse tim file to get MJD range
mjds = []
with open(timfile) as f:
    for line in f:
        if line.strip() and not line.startswith(('C', '#', 'FORMAT')):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    mjd = float(parts[2])
                    mjds.append(mjd)
                except:
                    pass

mjds = np.array(mjds)
print(f"Data MJD range: {mjds.min():.2f} to {mjds.max():.2f}")
print(f"Data span: {mjds.max() - mjds.min():.1f} days ({(mjds.max()-mjds.min())/365.25:.1f} years)")

# Check clock file coverage
import glob
clock_files = glob.glob('data/clock/*.clk')
print(f"\nFound {len(clock_files)} clock files")

for cf in sorted(clock_files):
    if 'mk2utc' in cf or 'bipm' in cf:
        # Parse clock file
        with open(cf) as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
        
        if lines:
            # First and last MJD
            first_mjd = float(lines[0].split()[0])
            last_mjd = float(lines[-1].split()[0])
            
            coverage = "✓ FULL" if last_mjd >= mjds.max() else f"⚠ ENDS AT {last_mjd:.1f}"
            print(f"  {cf.split('/')[-1]:30s}: {first_mjd:.1f} to {last_mjd:.1f}  {coverage}")

