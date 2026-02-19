# Ephemeris Handling in JUG

## Overview

JUG uses JPL planetary ephemerides (DE files) for computing barycentric corrections. JUG automatically downloads ephemerides from JPL servers on first use and caches them locally.

## Automatic Ephemeris Resolution

JUG intelligently handles ephemeris availability:

1. **Standard ephemerides** (DE440, DE430, etc.) → Downloaded from NASA NAIF server
2. **DE436 and DE441** → Downloaded from JPL SSD server (alternative source)
3. **Old ephemerides** (DE421, DE414, etc.) → Downloaded from NAIF archive
4. **Download failures** → Falls back to DE440 with a warning

### Available Ephemerides

- ✅ **DE440** (114 MB) - **Current JPL standard** (2020+)
- ✅ **DE436** (114 MB) - Common in existing datasets (from JPL SSD server)
- ✅ **DE430** (114 MB) - Previous standard (2013-2020)
- ✅ **builtin** - Astropy's built-in ephemeris (lower accuracy, always works offline)

### DE436 Special Handling

**DE436 is no longer on the standard NAIF server**, but JUG automatically downloads it from JPL's Solar System Dynamics (SSD) server:

```
Downloading DE436 from JPL SSD server...
✓ DE436 downloaded successfully.
```

This happens transparently - no user intervention needed!

## Pre-downloading Ephemerides

To ensure JUG can work offline or to pre-cache for faster first use:

```bash
python tools/download_ephemerides.py
```

This downloads and caches DE440, DE436, and DE430 to `~/.astropy/cache/`.

## Updating .par Files (Optional)

Your .par files can specify any ephemeris. JUG will handle it automatically:

```
EPHEM          DE436    # Works! Downloaded from JPL SSD
EPHEM          DE440    # Recommended current standard
EPHEM          DE430    # Previous standard
```

## Where Ephemerides Are Cached

- **Linux/Mac**: `~/.astropy/cache/download/url/`
- **Windows**: `%USERPROFILE%\.astropy\cache\download\url\`

Cached files are reused across all astropy-based applications.

## Technical Details

The automatic resolution is implemented in `jug/residuals/simple_calculator.py`:

```python
_SSD_EPHEMERIDES = {
    'de436': 'https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de436.bsp',
    'de441': 'https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de441.bsp',
}
```

The `_resolve_ephemeris()` function:
1. Checks if ephemeris is in `_SSD_EPHEMERIDES` → downloads from SSD
2. Checks if it's an old ephemeris → downloads from NAIF archive  
3. Otherwise → uses astropy's standard mechanism
4. On any failure → falls back to DE440 with console warning

## Tempo2 Ephemeris Files

Tempo2 uses a different format (JPL PLAN format, `.1950.2050` files) that is incompatible with astropy/JUG. However, **JUG now downloads the proper SPICE/BSP format directly from JPL**, so you don't need to convert Tempo2 files.

If your dataset was processed with Tempo2 using DE436, JUG will automatically use the same ephemeris (downloaded from JPL SSD) for consistency.

## Impact on Results

Differences between recent JPL ephemerides (DE430, DE436, DE440) are **negligible for pulsar timing**:
- Typically at the nanosecond level
- Only extremely high-precision analyses spanning decades might see measurable differences
- Using DE436 (from JPL SSD) gives exact matching with Tempo2 results

## Troubleshooting

### Download fails
If ephemeris download fails, JUG will print a warning and fall back to DE440:

```
======================================================================
WARNING: Could not download DE436 from JPL SSD server.
         Error: ...
         Falling back to DE440.
======================================================================
```

Check your internet connection or try pre-downloading with the script.

### Verify cached ephemerides

```bash
ls -lh ~/.astropy/cache/download/url/*/contents
```

### Clear cache and re-download

```bash
rm -rf ~/.astropy/cache/download
python tools/download_ephemerides.py
```
