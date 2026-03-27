import os
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
    first_date = df["date"].iloc[0].to_pydatetime()
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
    {data_table_html}
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
