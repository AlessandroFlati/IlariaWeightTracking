import os
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go

TARGET_DATE = datetime(2026, 12, 31)
HEIGHT_M = 1.75
MIN_BMI = 18.5
MIN_WEIGHT = MIN_BMI * HEIGHT_M ** 2

COLORS = [
    "#64B5F6",  # Blue - Linear
    "#FFB74D",  # Orange - Poly 2
    "#CE93D8",  # Purple - Poly 3
    "#81C784",  # Green - Exponential
    "#E57373",  # Red - Hall
    "#4DD0E1",  # Cyan - Thomas
]


def generate_html(df: pd.DataFrame, models: dict, output_path: str) -> None:
    first_date = df["date"].iloc[0].to_pydatetime()
    last_day = (TARGET_DATE - first_date).days
    projection_days = np.linspace(0, last_day, 500)
    projection_dates = [first_date + timedelta(days=float(d)) for d in projection_days]

    fig = go.Figure()

    table_rows = []
    filtered_models = []
    color_idx = 0
    for name, model in models.items():
        weight_at_target = model["predict"](last_day)
        if weight_at_target < MIN_WEIGHT:
            filtered_models.append(name)
            continue
        color = COLORS[color_idx % len(COLORS)]
        color_idx += 1
        predicted = np.array([model["predict"](d) for d in projection_days])
        std = model.get("residual_std", 0)
        last_data_day = df["days"].iloc[-1]
        extrapolation = np.maximum(projection_days - last_data_day, 0)
        scale = 1.0 + np.sqrt(extrapolation / max(last_data_day, 1.0))
        hex_color = color.lstrip("#")
        r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        n_bands = 20
        for band_i in range(n_bands, 0, -1):
            outer = band_i / n_bands * 3.0
            inner = (band_i - 1) / n_bands * 3.0
            opacity = 0.15 * (np.exp(-0.5 * inner**2) - np.exp(-0.5 * outer**2))
            upper = predicted + outer * std * scale
            lower = predicted - outer * std * scale
            fig.add_trace(go.Scatter(
                x=projection_dates + projection_dates[::-1],
                y=np.concatenate([upper, lower[::-1]]).tolist(),
                fill="toself",
                fillcolor=f"rgba({r},{g},{b},{opacity:.3f})",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            ))
        fig.add_trace(go.Scatter(
            x=projection_dates,
            y=predicted,
            mode="lines",
            name=name,
            line=dict(color=color, width=2),
        ))
        last_data_day_val = df["days"].iloc[-1]
        extrap = max(last_day - last_data_day_val, 0)
        target_scale = 1.0 + np.sqrt(extrap / max(last_data_day_val, 1.0))
        target_std = std * target_scale
        table_rows.append({
            "name": name,
            "r_squared": model["r_squared"],
            "target_weight": weight_at_target,
            "target_std": target_std,
        })

    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["weight"],
        mode="markers+lines",
        name="Actual Weight",
        marker=dict(size=8, color="#ffffff"),
        line=dict(color="#ffffff", width=1, dash="dot"),
    ))

    fig.update_layout(
        title="Ilaria - Weight Tracking & Projections",
        xaxis_title="Date",
        yaxis_title="Weight (kg)",
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    plot_json = fig.to_json()

    table_html = "<table><thead><tr>"
    table_html += "<th>Model</th><th>R&sup2;</th><th>Mean</th>"
    table_html += "<th>&plusmn;1&sigma; (68%)</th>"
    table_html += "<th>&plusmn;2&sigma; (95%)</th>"
    table_html += "<th>&plusmn;3&sigma; (99.7%)</th>"
    table_html += "</tr></thead><tbody>"
    for row in table_rows:
        w = row["target_weight"]
        s = row["target_std"]
        table_html += f'<tr><td>{row["name"]}</td>'
        table_html += f'<td>{row["r_squared"]:.4f}</td>'
        table_html += f'<td>{w:.1f} kg</td>'
        table_html += f'<td>{w - s:.1f} - {w + s:.1f} kg</td>'
        table_html += f'<td>{w - 2*s:.1f} - {w + 2*s:.1f} kg</td>'
        table_html += f'<td>{w - 3*s:.1f} - {w + 3*s:.1f} kg</td></tr>'
    table_html += "</tbody></table>"

    data_table_html = "<table><thead><tr>"
    data_table_html += "<th>Date</th><th>Weight (kg)</th>"
    data_table_html += "</tr></thead><tbody>"
    for _, row in df.iterrows():
        data_table_html += f'<tr><td>{row["date"].strftime("%Y-%m-%d")}</td>'
        data_table_html += f'<td>{row["weight"]}</td></tr>'
    data_table_html += "</tbody></table>"

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
            background: #0f0f23;
            color: #e0e0e0;
        }}
        h1 {{
            text-align: center;
            font-weight: 300;
            margin-bottom: 30px;
            color: #f0f0f0;
        }}
        #chart {{
            width: 100%;
            height: 500px;
            background: #1a1a2e;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.4);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 30px;
            background: #1a1a2e;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.4);
        }}
        th, td {{
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid #2a2a4a;
        }}
        th {{
            background: #16213e;
            font-weight: 600;
            color: #a0a0c0;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            color: #666;
            font-size: 0.85em;
        }}
        .filtered {{
            text-align: center;
            margin-top: 15px;
            color: #555;
            font-size: 0.8em;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <h1>Ilaria - Weight Tracking & Projections</h1>
    <div id="chart"></div>
    {table_html}
    {data_table_html}
    {"<div class='filtered'>Filtered out (unrealistic, BMI &lt; 18.5 at year end): " + ", ".join(filtered_models) + "</div>" if filtered_models else ""}
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

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
