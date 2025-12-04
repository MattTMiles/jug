# JUG Documentation

This directory contains detailed documentation, session notes, and technical reports.

---

## Organization

### Root Level (Main Docs)

These stay at the repository root for easy access:

- **`QUICK_REFERENCE.md`** - Quick start guide and command reference (read this first!)
- **`JUG_PROGRESS_TRACKER.md`** - Milestone tracking and project status
- **`JUG_implementation_guide.md`** - Implementation details and architecture

### This Directory (`docs/`)

Detailed documentation and reports:

- **`CLEANUP_MANIFEST.md`** - Code cleanup manifest (2025-12-04)
- **`CLEANUP_SUMMARY.md`** - Cleanup results and verification

### Examples

- **`examples/full_walkthrough.ipynb`** - Complete tutorial notebook

### Playground

- **`playground/`** - Session notes, benchmarks, and analysis (Markdown only - Python archived)
  - `SESSION_*.md` - Development session summaries
  - `BENCHMARK_*.md` - Performance analysis reports
  - Various technical notes and findings

### Archival Code

- **`archival_code/`** - Experimental code safely archived but not deleted
  - See `archival_code/README.md` for full manifest

---

## Quick Links

### Getting Started
- Start here: [QUICK_REFERENCE.md](../QUICK_REFERENCE.md)
- Tutorial: [examples/full_walkthrough.ipynb](../examples/full_walkthrough.ipynb)

### Development
- Project status: [JUG_PROGRESS_TRACKER.md](../JUG_PROGRESS_TRACKER.md)
- Architecture: [JUG_implementation_guide.md](../JUG_implementation_guide.md)

### Recent Changes
- Code cleanup: [CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md)

---

## Documentation Guidelines

When creating new documentation:

1. **User-facing guides** → Root level or `examples/`
2. **Technical reports** → `docs/`
3. **Session notes** → `playground/` (Markdown only)
4. **Archived code** → `archival_code/`

**Exception**: Progress tracker, implementation guide, and quick reference always stay at root.
