# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Static site generator that fits multiple weight-loss curve models to a CSV of measurements and emits `docs/index.html` for GitHub Pages. Every push that touches `weights.csv`, `src/**`, or `requirements.txt` triggers `.github/workflows/deploy.yml`, which runs `python -m src.generate` and deploys `docs/` via `actions/deploy-pages@v4`.

## Commands

```bash
# Activate venv first if present (venv/Scripts/Activate.ps1 or .venv/Scripts/Activate.ps1)
pip install -r requirements.txt

# Regenerate docs/index.html from weights.csv (run from repo root; paths are resolved relative to src/)
python -m src.generate

# Tests
pytest                              # full suite
pytest tests/test_models.py         # single file
pytest tests/test_models.py::test_linear_model_predicts_decrease  # single test
```

Always invoke the generator as a module (`python -m src.generate`), not `python src/generate.py` — the code uses absolute imports (`from src.data import ...`).

## Architecture

Three-stage pipeline composed in `src/generate.py`:

1. **`src/data.py::load_weights`** — parses `weights.csv` (columns: `date` in `DD/MM/YYYY`, `weight`, `notes`), sorts by date, and adds a `days` column (float days since first measurement). The `days` column is the x-axis used by every model; date parsing uses `format="%d/%m/%Y"` — don't change the CSV date format without updating this.

2. **`src/models.py::fit_all_models`** — fits six models against `(days, weights)`: linear, polynomial degree 2/3 (via `np.polyfit`), exponential decay `a·exp(-b·t)+c`, Hall linearized 2011 `w_final + (w0-w_final)·exp(-t/τ)`, and Thomas 2013 `w0 - Δw·(1-exp(-k·t))` (via `scipy.optimize.curve_fit`, `maxfev=10000`). Each model returns `{"predict": callable, "r_squared": float, "residual_std": float}`. Non-linear fits that raise `RuntimeError` are silently dropped from the result dict — callers must not assume all six keys exist.

3. **`src/chart.py::generate_html`** — projects every surviving model to `TARGET_DATE = 2026-12-31` and writes a single self-contained HTML file (dark Plotly theme, CDN-loaded `plotly-2.35.2`). Key behaviors:
   - **BMI filter**: models predicting `weight < MIN_BMI * HEIGHT_M² = 18.5 * 1.75²` at the target date are excluded from the chart/table and listed in a "filtered out" footer.
   - **Gaussian confidence bands**: 20 stacked `fill="toself"` scatter traces per model approximate a Gaussian envelope out to ±3σ; opacity per band is `0.15·(exp(-½·inner²) - exp(-½·outer²))`.
   - **Extrapolation scaling**: past the last data point, the band width is scaled by `1 + sqrt(extrap_days / last_data_day)` so uncertainty grows with projection distance.
   - The HTML embeds both a model-summary table (mean ± 1/2/3σ ranges at target date) and a full data table.

## Conventions

- Keep source code, comments, and HTML content in English. The CSV `notes` column is Italian — leave it untranslated.
- `weights.csv` is the single source of truth; there is no database. Adding a row and pushing is the normal update workflow (CI regenerates the page).
- Tests reference `weights.csv` at the repo root — run `pytest` from the repo root.
