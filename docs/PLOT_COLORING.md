# Plot Coloring in JUG

This document describes the backend-colored residual points system and noise realization coloring in the JUG GUI.

## Overview

As of the latest version, JUG defaults to coloring residual points by **backend/receiver** (identified from TOA flags). Users can switch between three coloring modes via **View → Color By**:
- **Backend/Receiver** (default): Colors points by backend, with a legend
- **Frequency (continuous)**: Colors points by observing frequency with a colorbar
- **None (single color)**: Uses the theme default color

Noise realization overlays also use theme-aware colors that can be customized.

## Backend Identity Rule

The backend for each TOA is resolved using the `flag_mapping.resolve_backends()` function, which tries flag keys in order:
1. `f` (most common: `-f` flag in `.tim` files)
2. `be` (alternate backend flag)
3. `backend` (explicit backend flag)

If none of these keys are present, the backend is labeled `__unknown__`.

This is consistent with how JUG's noise model matching (EFAC/EQUAD) identifies backends.

## Default Palettes

JUG provides curated categorical palettes for light and dark themes, each with 12 distinct colors:

### Light Theme Palette
Muted but distinct colors designed for warm paper background (`#FAF8F5`):

| Index | Color Name       | RGBA                 | Hex       |
|-------|------------------|----------------------|-----------|
| 0     | Navy             | (43, 65, 98, 220)    | #2b4162   |
| 1     | Burgundy         | (94, 24, 3, 220)     | #5e1803   |
| 2     | Teal             | (31, 77, 74, 220)    | #1f4d4a   |
| 3     | Warm Amber       | (181, 137, 10, 220)  | #b5890a   |
| 4     | Muted Green      | (45, 106, 79, 220)   | #2d6a4f   |
| 5     | Muted Red        | (158, 42, 43, 220)   | #9e2a2b   |
| 6     | Purple           | (88, 57, 114, 220)   | #583972   |
| 7     | Terracotta       | (204, 97, 62, 220)   | #cc613e   |
| 8     | Royal Blue       | (65, 105, 225, 200)  | #4169e1   |
| 9     | Saddle Brown     | (139, 69, 19, 220)   | #8b4513   |
| 10    | Steel Blue       | (70, 130, 180, 200)  | #4682b4   |
| 11    | Olive            | (128, 128, 0, 200)   | #808000   |

### Dark Theme Palette
Saturated neon-ish colors for dark background (`#2a2139`):

| Index | Color Name       | RGBA                  | Hex       |
|-------|------------------|-----------------------|-----------|
| 0     | Neon Cyan        | (54, 249, 246, 220)   | #36f9f6   |
| 1     | Neon Pink        | (255, 126, 219, 220)  | #ff7edb   |
| 2     | Neon Yellow      | (254, 222, 93, 220)   | #fede5d   |
| 3     | Neon Orange      | (249, 126, 114, 220)  | #f97e72   |
| 4     | Neon Green       | (114, 241, 184, 220)  | #72f1b8   |
| 5     | Neon Red         | (254, 68, 80, 220)    | #fe4450   |
| 6     | Electric Blue    | (3, 237, 249, 220)    | #03edf9   |
| 7     | Neon Purple      | (187, 134, 252, 220)  | #bb86fc   |
| 8     | Warm Orange      | (255, 184, 108, 220)  | #ffb86c   |
| 9     | Lime Green       | (108, 226, 108, 220)  | #6ce26c   |
| 10    | Hot Pink         | (255, 105, 180, 220)  | #ff69b4   |
| 11    | Light Sky Blue   | (135, 206, 250, 220)  | #87cefa   |

**Design Constraints:**
- Colorblind-safe (avoids red/green-only distinctions)
- High contrast against plot background and grid
- Stable assignment: backends are sorted alphabetically and assigned colors in order

If a dataset has more than 12 backends, colors wrap around (backend 13 gets color 0, etc.).

## Noise Realization Colors

Noise process overlays use theme-aware colors with transparency:

### Light Theme
- **RedNoise**: Muted red `(158, 42, 43, 180)`
- **DMNoise**: Muted green `(45, 106, 79, 180)`
- **ECORR**: Purple `(88, 57, 114, 180)`

### Dark Theme
- **RedNoise**: Neon red `(254, 68, 80, 180)`
- **DMNoise**: Neon green `(114, 241, 184, 180)`
- **ECORR**: Neon purple `(187, 134, 252, 180)`

## User Customization

Users can override default colors via **View → Color By → Customize Colors...**

### Backend Colors Tab
- Lists all backends detected in the current dataset
- Click the color swatch to open a color picker
- Changes apply immediately (live preview)
- "Reset to Defaults" restores theme palettes

### Noise Colors Tab
- Lists common noise processes (RedNoise, DMNoise, ECORR)
- Same color picker workflow
- Applies to noise realization overlays

### Persistence
Color overrides are stored in `QSettings` (cross-platform):
- **Backend colors**: `JUG/PlotColors/backends/<backend_name>` → hex color
- **Noise colors**: `JUG/PlotColors/noise/<process_name>` → hex color

Overrides persist across sessions and are theme-independent (user's red stays red in both light and dark mode, but defaults change with theme).

### Merge Policy
When loading a new dataset with different backends:
- Backends present in the dataset use their override color if one exists
- Backends not in the dataset retain their override in settings (for future datasets)
- New backends (no override) get default palette colors

## Where Overrides Are Stored

On Linux: `~/.config/JUG/JUG.conf` (INI format)  
On macOS: `~/Library/Preferences/com.JUG.JUG.plist`  
On Windows: `HKEY_CURRENT_USER\Software\JUG\JUG`

Example contents (Linux):
```ini
[PlotColors%2Fbackends]
GUPPI=#2b4162dc
ASP=#5e1803dc

[PlotColors%2Fnoise]
RedNoise=#9e2a2bdc
DMNoise=#2d6a4fdc
```

## Implementation Details

### Architecture
- **`jug/gui/theme.py`**: Defines `BACKEND_PALETTE_LIGHT`, `BACKEND_PALETTE_DARK`, `NOISE_COLORS_LIGHT`, `NOISE_COLORS_DARK`
- **`jug/gui/plot_colors.py`**: Vectorized helpers (`get_backend_labels`, `build_backend_brush_array`, `build_noise_colors`)
- **`jug/gui/widgets/backend_legend.py`**: Legend widget showing backend names + colors
- **`jug/gui/dialogs/plot_colors_dialog.py`**: Customization dialog
- **`jug/gui/main_window.py`**: Integrates everything, handles persistence

### Performance
- **O(N) complexity**: Uses `numpy.unique()` + boolean masking, no per-TOA Python loops
- **Caching**: Backend brush array is cached and invalidated on:
  - Data load
  - Theme change
  - Color mode switch
  - User override change

### Testing
Run tests with:
```bash
python -m pytest jug/tests/test_plot_colors.py -v
```

Tests cover:
- Backend label extraction (including fallback keys)
- Stable color assignment
- Theme-aware palette switching
- User override application
- QSettings serialization round-trip

## How to Extend

### Adding a New Noise Process
1. Add default colors to `theme.py`:
   ```python
   NOISE_COLORS_LIGHT = {
       ...
       "NewProcess": (r, g, b, a),
   }
   NOISE_COLORS_DARK = {
       ...
       "NewProcess": (r, g, b, a),
   }
   ```

2. Process will automatically appear in the customization dialog when detected in data.

### Increasing Palette Size
To add more default colors (e.g., for datasets with >12 backends):

1. Extend `BACKEND_PALETTE_LIGHT` and `BACKEND_PALETTE_DARK` in `theme.py`
2. Ensure new colors meet design constraints (colorblind-safe, high contrast)

### Changing Backend Resolution Logic
The backend key resolution order is defined in `jug/engine/flag_mapping.py`:
```python
DEFAULT_BACKEND_CONFIG = FlagMappingConfig(
    candidates=["f", "be", "backend"],
    ...
)
```

Modify `candidates` to change key priority or add new keys.

## Manual Verification Checklist

Before release, verify:

- [ ] Light theme: Backend colors are distinct and readable
- [ ] Dark theme: Backend colors are distinct and readable
- [ ] Backend legend appears when "Backend/Receiver" mode is active
- [ ] Backend legend hides when switching to "None" or "Frequency"
- [ ] Many backends (8+): Colors remain distinguishable
- [ ] Switching theme updates backend colors to new palette (preserves explicit overrides)
- [ ] Noise realization overlays use theme-aware colors
- [ ] Customization dialog opens and shows current dataset backends
- [ ] Changing a color in dialog updates plot immediately (live preview)
- [ ] "Reset to Defaults" restores theme palette
- [ ] "Apply" saves overrides to QSettings
- [ ] "Cancel" reverts to pre-dialog state
- [ ] Overrides persist across app restarts
- [ ] Loading a new dataset with different backends merges correctly (keeps old overrides)

## Known Limitations

1. **Fixed palette size**: If a dataset has >12 backends, colors wrap. This is acceptable since 12 distinct colors is approaching the human perception limit. Users can override individual backends if needed.

2. **No per-theme overrides**: User overrides are theme-independent. If a user sets Backend_A to bright red, it stays bright red in both light and dark mode. This is by design for simplicity.

3. **No colormap for backends**: Backends use discrete categorical colors, not a continuous colormap. This is appropriate since backends are categorical data.

## References

- Colorblind-safe palettes: [ColorBrewer](https://colorbrewer2.org/)
- Qt color management: [QColor docs](https://doc.qt.io/qt-6/qcolor.html)
- QSettings: [QSettings docs](https://doc.qt.io/qt-6/qsettings.html)
