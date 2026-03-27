# Ilaria Weight Tracking - Design Spec

Created: 2026-03-27 09:32 UTC

## Overview

Weight tracking application for Ilaria. Reads historical weight measurements, fits multiple mathematical models, projects trends to end of 2026, and displays results as an interactive chart on a GitHub Pages site.

## Data

- Source: `weights.csv` at repo root
- Format: `date` (DD/MM/YYYY), `weight` (kg), `notes` (free text)
- Update workflow: manual edit + push to main; GitHub Actions regenerates the page

## Fitting Models

Six models fitted via `scipy.optimize.curve_fit` (or `numpy.polyfit` for polynomials) on historical data, then projected to 2026-12-31:

1. **Linear** - `W(t) = a*t + b`
2. **Polynomial degree 2** - `W(t) = a*t^2 + b*t + c`
3. **Polynomial degree 3** - `W(t) = a*t^3 + b*t^2 + c*t + d`
4. **Exponential decay** - `W(t) = a * exp(-b*t) + c`
5. **Hall Linearized (Lancet 2011)** - `W(t) = W_final + (W0 - W_final) * exp(-t/tau)`. Biologically motivated: exponential approach to a steady-state weight.
6. **Thomas (Int J Obes 2013)** - `W(t) = W0 - delta_W * (1 - exp(-k*t))`. Clinically simplified version of Hall's model.

Each model reports: R-squared on historical data, projected weight at 2026-12-31.

Note: Models 5 and 6 are mathematically equivalent (reparameterizations of the same curve). Both are included because they appear in different papers and their parameter names carry different clinical interpretations.

## Frontend

Single-page static HTML (`docs/index.html`) with:

- Plotly.js loaded via CDN
- Interactive chart: scatter points for real measurements + one trace per model projection
- Toggleable traces (Plotly legend click)
- Summary table below the chart: model name, R-squared, projected weight at 2026-12-31
- Minimal design, light background, responsive layout

## Generation Pipeline

`src/generate.py`:

1. Read and parse `weights.csv`
2. Convert dates to numeric days-since-first-measurement
3. Fit each model to historical data
4. Generate projection points from first measurement to 2026-12-31
5. Compute R-squared for each model
6. Build Plotly figure data as JSON
7. Render `docs/index.html` from an HTML template with embedded Plotly JSON

## GitHub Actions

`.github/workflows/deploy.yml`:

- Trigger: push to `main` that modifies `weights.csv`, `src/**`, or `requirements.txt`
- Steps: checkout, setup Python 3.11, install requirements, run `src/generate.py`, deploy `docs/` to GitHub Pages

## Project Structure

```
weights.csv
requirements.txt
src/generate.py
docs/index.html              (generated output, committed by CI)
.github/workflows/deploy.yml
```

## Dependencies

`requirements.txt`:
- numpy
- scipy
- pandas
- plotly

## V2 (Future)

- AI-generated commentary (Sonnet) on overall progress and latest measurement, displayed on the page
