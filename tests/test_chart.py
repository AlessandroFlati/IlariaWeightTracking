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
