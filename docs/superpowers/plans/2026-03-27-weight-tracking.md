# Ilaria Weight Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python pipeline that fits 6 mathematical models to weight data and generates an interactive Plotly chart deployed via GitHub Pages.

**Architecture:** A single generation script reads `weights.csv`, fits models using scipy/numpy, builds a Plotly figure, and writes a self-contained `docs/index.html`. GitHub Actions runs this on push and deploys to Pages.

**Tech Stack:** Python 3.11+, numpy, scipy, pandas, plotly; GitHub Actions for CI/CD; GitHub Pages for hosting.

---

## File Structure

```
weights.csv                          # Already exists - weight measurements
requirements.txt                     # Python dependencies
src/__init__.py                      # Empty, makes src a package
src/data.py                          # CSV loading and date conversion
src/models.py                        # Model definitions, fitting, R-squared
src/chart.py                         # Plotly chart + HTML generation
src/generate.py                      # Main entry point: orchestrates pipeline
tests/__init__.py                    # Empty
tests/test_data.py                   # Tests for data loading
tests/test_models.py                 # Tests for model fitting
tests/test_chart.py                  # Tests for chart generation
.github/workflows/deploy.yml         # CI/CD pipeline
```

---

### Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```
numpy
scipy
pandas
plotly
pytest
```

- [ ] **Step 2: Create empty package init files**

Create empty `src/__init__.py` and `tests/__init__.py`.

- [ ] **Step 3: Create virtual environment and install dependencies**

Run:
```bash
python3 -m venv /home/alessandro/.virtualenvs/ilaria-weight
source /home/alessandro/.virtualenvs/ilaria-weight/bin/activate
pip install -r requirements.txt
```

- [ ] **Step 4: Commit**

```bash
git add requirements.txt src/__init__.py tests/__init__.py
git commit -m "feat: add project setup with dependencies"
```

---

### Task 2: Data Loading Module

**Files:**
- Create: `tests/test_data.py`
- Create: `src/data.py`

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd
from src.data import load_weights

def test_load_weights_returns_dataframe():
    df = load_weights("weights.csv")
    assert isinstance(df, pd.DataFrame)
    assert "date" in df.columns
    assert "weight" in df.columns
    assert "days" in df.columns
    assert df["days"].iloc[0] == 0
    assert df["date"].dtype == "datetime64[ns]"
    assert len(df) > 0

def test_load_weights_days_monotonically_increase():
    df = load_weights("weights.csv")
    assert (df["days"].diff().dropna() > 0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source /home/alessandro/.virtualenvs/ilaria-weight/bin/activate && python -m pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Write minimal implementation**

```python
import pandas as pd

def load_weights(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    df = df.sort_values("date").reset_index(drop=True)
    df["days"] = (df["date"] - df["date"].iloc[0]).dt.days.astype(float)
    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source /home/alessandro/.virtualenvs/ilaria-weight/bin/activate && python -m pytest tests/test_data.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data.py tests/test_data.py
git commit -m "feat: add data loading module with date-to-days conversion"
```

---

### Task 3: Model Fitting Module

**Files:**
- Create: `tests/test_models.py`
- Create: `src/models.py`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from src.models import fit_all_models, compute_r_squared

def test_fit_all_models_returns_dict():
    days = np.array([0, 10, 20, 30, 40, 50], dtype=float)
    weights = np.array([95, 94, 93, 92.5, 92, 91.5], dtype=float)
    results = fit_all_models(days, weights)
    assert isinstance(results, dict)
    expected_names = [
        "Linear",
        "Polynomial (degree 2)",
        "Polynomial (degree 3)",
        "Exponential Decay",
        "Hall Linearized (2011)",
        "Thomas (2013)",
    ]
    for name in expected_names:
        assert name in results, f"Missing model: {name}"
        assert "predict" in results[name]
        assert "r_squared" in results[name]
        assert callable(results[name]["predict"])
        assert 0 <= results[name]["r_squared"] <= 1.0

def test_linear_model_predicts_decrease():
    days = np.array([0, 10, 20, 30, 40, 50], dtype=float)
    weights = np.array([95, 94, 93, 92.5, 92, 91.5], dtype=float)
    results = fit_all_models(days, weights)
    predict = results["Linear"]["predict"]
    assert predict(100) < predict(0)

def test_compute_r_squared():
    actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    predicted = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
    r2 = compute_r_squared(actual, predicted)
    assert 0.95 < r2 <= 1.0

def test_compute_r_squared_perfect():
    actual = np.array([1.0, 2.0, 3.0])
    r2 = compute_r_squared(actual, actual)
    assert r2 == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source /home/alessandro/.virtualenvs/ilaria-weight/bin/activate && python -m pytest tests/test_models.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation**

```python
import numpy as np
from scipy.optimize import curve_fit

def compute_r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    if ss_tot == 0:
        return 1.0
    return 1.0 - ss_res / ss_tot

def _fit_polynomial(days, weights, degree):
    coeffs = np.polyfit(days, weights, degree)
    poly = np.poly1d(coeffs)
    predicted = poly(days)
    r2 = compute_r_squared(weights, predicted)
    return {"predict": lambda t, p=poly: p(t), "r_squared": r2}

def _fit_exponential_decay(days, weights):
    def model(t, a, b, c):
        return a * np.exp(-b * t) + c

    try:
        p0 = [weights[0] - weights[-1], 0.01, weights[-1]]
        popt, _ = curve_fit(model, days, weights, p0=p0, maxfev=10000)
        predicted = model(days, *popt)
        r2 = compute_r_squared(weights, predicted)
        return {"predict": lambda t, p=popt: model(t, *p), "r_squared": r2}
    except RuntimeError:
        return None

def _fit_hall_linearized(days, weights):
    def model(t, w_final, w0, tau):
        return w_final + (w0 - w_final) * np.exp(-t / tau)

    try:
        p0 = [weights[-1] - 5, weights[0], 200.0]
        popt, _ = curve_fit(model, days, weights, p0=p0, maxfev=10000)
        predicted = model(days, *popt)
        r2 = compute_r_squared(weights, predicted)
        return {"predict": lambda t, p=popt: model(t, *p), "r_squared": r2}
    except RuntimeError:
        return None

def _fit_thomas(days, weights):
    def model(t, w0, delta_w, k):
        return w0 - delta_w * (1 - np.exp(-k * t))

    try:
        p0 = [weights[0], weights[0] - weights[-1], 0.01]
        popt, _ = curve_fit(model, days, weights, p0=p0, maxfev=10000)
        predicted = model(days, *popt)
        r2 = compute_r_squared(weights, predicted)
        return {"predict": lambda t, p=popt: model(t, *p), "r_squared": r2}
    except RuntimeError:
        return None

def fit_all_models(days: np.ndarray, weights: np.ndarray) -> dict:
    results = {}
    results["Linear"] = _fit_polynomial(days, weights, 1)
    results["Polynomial (degree 2)"] = _fit_polynomial(days, weights, 2)
    results["Polynomial (degree 3)"] = _fit_polynomial(days, weights, 3)

    exp_result = _fit_exponential_decay(days, weights)
    if exp_result:
        results["Exponential Decay"] = exp_result

    hall_result = _fit_hall_linearized(days, weights)
    if hall_result:
        results["Hall Linearized (2011)"] = hall_result

    thomas_result = _fit_thomas(days, weights)
    if thomas_result:
        results["Thomas (2013)"] = thomas_result

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source /home/alessandro/.virtualenvs/ilaria-weight/bin/activate && python -m pytest tests/test_models.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/models.py tests/test_models.py
git commit -m "feat: add model fitting module with 6 weight loss models"
```

---

### Task 4: Chart and HTML Generation

**Files:**
- Create: `tests/test_chart.py`
- Create: `src/chart.py`

- [ ] **Step 1: Write the failing tests**

```python
import os
import numpy as np
import pandas as pd
from src.chart import generate_html

def test_generate_html_creates_file(tmp_path):
    df = pd.DataFrame({
        "date": pd.to_datetime(["2025-10-28", "2025-12-02", "2026-01-10"]),
        "weight": [95.35, 92.95, 91.6],
        "days": [0.0, 35.0, 74.0],
    })
    models = {
        "Linear": {
            "predict": lambda t: 95 - 0.05 * t,
            "r_squared": 0.95,
        },
    }
    output_path = str(tmp_path / "index.html")
    generate_html(df, models, output_path)
    assert os.path.exists(output_path)
    content = open(output_path).read()
    assert "plotly" in content.lower()
    assert "Linear" in content
    assert "95.35" in content

def test_generate_html_contains_summary_table(tmp_path):
    df = pd.DataFrame({
        "date": pd.to_datetime(["2025-10-28", "2025-12-02"]),
        "weight": [95.35, 92.95],
        "days": [0.0, 35.0],
    })
    models = {
        "Linear": {
            "predict": lambda t: 95 - 0.05 * t,
            "r_squared": 0.95,
        },
    }
    output_path = str(tmp_path / "index.html")
    generate_html(df, models, output_path)
    content = open(output_path).read()
    assert "R" in content
    assert "2026-12-31" in content or "31/12/2026" in content
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source /home/alessandro/.virtualenvs/ilaria-weight/bin/activate && python -m pytest tests/test_chart.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation**

```python
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go

TARGET_DATE = datetime(2026, 12, 31)

COLORS = [
    "#2196F3",  # Blue - Linear
    "#FF9800",  # Orange - Poly 2
    "#9C27B0",  # Purple - Poly 3
    "#4CAF50",  # Green - Exponential
    "#F44336",  # Red - Hall
    "#00BCD4",  # Cyan - Thomas
]

def generate_html(df: pd.DataFrame, models: dict, output_path: str) -> None:
    first_date = df["date"].iloc[0]
    last_day = (TARGET_DATE - first_date).days
    projection_days = np.linspace(0, last_day, 500)
    projection_dates = [first_date + timedelta(days=float(d)) for d in projection_days]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["weight"],
        mode="markers+lines",
        name="Actual Weight",
        marker=dict(size=8, color="#333333"),
        line=dict(color="#333333", width=1, dash="dot"),
    ))

    table_rows = []
    for i, (name, model) in enumerate(models.items()):
        color = COLORS[i % len(COLORS)]
        predicted = np.array([model["predict"](d) for d in projection_days])
        fig.add_trace(go.Scatter(
            x=projection_dates,
            y=predicted,
            mode="lines",
            name=name,
            line=dict(color=color, width=2),
        ))
        weight_at_target = model["predict"](last_day)
        table_rows.append({
            "name": name,
            "r_squared": model["r_squared"],
            "target_weight": weight_at_target,
        })

    fig.update_layout(
        title="Ilaria - Weight Tracking & Projections",
        xaxis_title="Date",
        yaxis_title="Weight (kg)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    plot_json = fig.to_json()

    table_html = "<table><thead><tr>"
    table_html += "<th>Model</th><th>R&sup2;</th><th>Projected Weight (2026-12-31)</th>"
    table_html += "</tr></thead><tbody>"
    for row in table_rows:
        table_html += f'<tr><td>{row["name"]}</td>'
        table_html += f'<td>{row["r_squared"]:.4f}</td>'
        table_html += f'<td>{row["target_weight"]:.1f} kg</td></tr>'
    table_html += "</tbody></table>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ilaria - Weight Tracking</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background: #fafafa;
            color: #333;
        }}
        h1 {{
            text-align: center;
            font-weight: 300;
            margin-bottom: 30px;
        }}
        #chart {{
            width: 100%;
            height: 500px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 30px;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #f5f5f5;
            font-weight: 600;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            color: #999;
            font-size: 0.85em;
        }}
    </style>
</head>
<body>
    <h1>Ilaria - Weight Tracking & Projections</h1>
    <div id="chart"></div>
    {table_html}
    <div class="footer">
        Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M")} |
        Data points: {len(df)} |
        Projection target: 2026-12-31
    </div>
    <script>
        var figure = {plot_json};
        Plotly.newPlot('chart', figure.data, figure.layout, {{responsive: true}});
    </script>
</body>
</html>"""

    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source /home/alessandro/.virtualenvs/ilaria-weight/bin/activate && python -m pytest tests/test_chart.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/chart.py tests/test_chart.py
git commit -m "feat: add chart and HTML generation with Plotly"
```

---

### Task 5: Main Entry Point

**Files:**
- Create: `src/generate.py`

- [ ] **Step 1: Write the entry point**

```python
import os

from src.data import load_weights
from src.models import fit_all_models
from src.chart import generate_html

def main():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "weights.csv")
    output_path = os.path.join(os.path.dirname(__file__), "..", "docs", "index.html")

    df = load_weights(csv_path)
    models = fit_all_models(df["days"].values, df["weight"].values)
    generate_html(df, models, output_path)

    print(f"Generated {output_path}")
    print(f"  Data points: {len(df)}")
    print(f"  Models fitted: {len(models)}")
    for name, model in models.items():
        print(f"    {name}: R^2 = {model['r_squared']:.4f}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full pipeline**

Run:
```bash
source /home/alessandro/.virtualenvs/ilaria-weight/bin/activate
python -m src.generate
```
Expected: prints summary, creates `docs/index.html`

- [ ] **Step 3: Verify the output file exists and is valid**

Run: `ls -la docs/index.html && head -5 docs/index.html`
Expected: file exists, starts with `<!DOCTYPE html>`

- [ ] **Step 4: Run all tests**

Run: `source /home/alessandro/.virtualenvs/ilaria-weight/bin/activate && python -m pytest tests/ -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/generate.py
git commit -m "feat: add main entry point for pipeline"
```

---

### Task 6: GitHub Actions Workflow

**Files:**
- Create: `.github/workflows/deploy.yml`

- [ ] **Step 1: Write the workflow**

```yaml
name: Generate and Deploy

on:
  push:
    branches: [main]
    paths:
      - 'weights.csv'
      - 'src/**'
      - 'requirements.txt'
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - run: pip install -r requirements.txt

      - run: python -m src.generate

      - uses: actions/upload-pages-artifact@v3
        with:
          path: docs

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 2: Commit**

```bash
mkdir -p .github/workflows
git add .github/workflows/deploy.yml
git commit -m "ci: add GitHub Actions workflow for Pages deployment"
```

---

### Task 7: Final Integration Test and Cleanup

- [ ] **Step 1: Run full test suite**

Run: `source /home/alessandro/.virtualenvs/ilaria-weight/bin/activate && python -m pytest tests/ -v`
Expected: all tests PASS

- [ ] **Step 2: Run full pipeline and verify output**

Run: `source /home/alessandro/.virtualenvs/ilaria-weight/bin/activate && python -m src.generate`
Expected: `docs/index.html` generated with all 6 models

- [ ] **Step 3: Open HTML locally to visually verify**

Run: `wslview docs/index.html` (opens in Windows browser)
Expected: interactive Plotly chart with data points and 6 model projections, summary table below

- [ ] **Step 4: Commit generated HTML and push**

```bash
git add docs/index.html
git commit -m "feat: add generated tracking page"
```
