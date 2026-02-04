
import numpy as np

# PINT values from previous runs
# FD1: -3.6623e-05
# Freqs approx ~900-1000 MHz

freq_mhz = 907.899
fd1 = -3.6623e-05

# JUG uses ln(f/1GHz)
freq_ghz = freq_mhz / 1000.0
log_val_ln = np.log(freq_ghz)
delay_ln = fd1 * log_val_ln

# Hypothesis: PINT uses log10?
log_val_10 = np.log10(freq_ghz)
delay_10 = fd1 * log_val_10

print(f"Freq: {freq_mhz} MHz")
print(f"FD1: {fd1}")
print(f"ln(f) delay:    {delay_ln:.6e} s")
print(f"log10(f) delay: {delay_10:.6e} s")
print(f"Diff:           {abs(delay_ln - delay_10)*1e6:.6f} µs")

# For J0613, FD4 and FD5 are large.
fd4 = 0.000411768
fd5 = -0.000451307

delay_ln_high = fd4 * (log_val_ln**4) + fd5 * (log_val_ln**5)
delay_10_high = fd4 * (log_val_10**4) + fd5 * (log_val_10**5)
print(f"High Order Ln:    {delay_ln_high:.6e} s")
print(f"High Order Log10: {delay_10_high:.6e} s")
print(f"High Order Diff:  {abs(delay_ln_high - delay_10_high)*1e6:.6f} µs")
