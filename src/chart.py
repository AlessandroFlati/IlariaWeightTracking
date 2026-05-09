import os
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.models import fit_all_models

TARGET_DATE = datetime(2026, 12, 31)
HEIGHT_M = 1.75
MIN_BMI = 18.5
MIN_WEIGHT = MIN_BMI * HEIGHT_M ** 2

COLORS = [
    "#64B5F6",  # Blue - Linear
    "#4DD0E1",  # Cyan - Thomas
]

ANIM_N_MIN = 5
ANIM_N_BANDS = 10
ANIM_PROJ_POINTS = 150
MODEL_ORDER = ["Linear", "Thomas (2013)"]


def _hex_to_rgb(color):
    h = color.lstrip("#")
    return int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _build_animation_figure(df: pd.DataFrame) -> go.Figure:
    first_date = df["date"].iloc[0].to_pydatetime()
    last_day = (TARGET_DATE - first_date).days
    projection_days = np.linspace(0, last_day, ANIM_PROJ_POINTS)
    projection_dates = [first_date + timedelta(days=float(d)) for d in projection_days]

    n_max = len(df)
    n_values = list(range(ANIM_N_MIN, n_max + 1))
    color_for = dict(zip(MODEL_ORDER, COLORS))

    def traces_for_n(n):
        sub = df.iloc[:n]
        models = fit_all_models(sub["days"].values, sub["weight"].values)
        last_data_day = sub["days"].iloc[-1]
        traces = []
        for m_name in MODEL_ORDER:
            color = color_for[m_name]
            r, g, bcol = _hex_to_rgb(color)
            model = models.get(m_name)
            if model is not None and (not np.isnan(model["predict"](last_day))):
                predicted = np.array([model["predict"](d) for d in projection_days])
                std = model.get("residual_std", 0)
                extrapolation = np.maximum(projection_days - last_data_day, 0)
                scale = 1.0 + np.sqrt(extrapolation / max(last_data_day, 1.0))
                for band_i in range(ANIM_N_BANDS, 0, -1):
                    outer = band_i / ANIM_N_BANDS * 3.0
                    inner = (band_i - 1) / ANIM_N_BANDS * 3.0
                    opacity = 0.15 * (np.exp(-0.5 * inner ** 2) - np.exp(-0.5 * outer ** 2))
                    upper = predicted + outer * std * scale
                    lower = predicted - outer * std * scale
                    traces.append(go.Scatter(
                        x=projection_dates + projection_dates[::-1],
                        y=np.concatenate([upper, lower[::-1]]).tolist(),
                        fill="toself",
                        fillcolor=f"rgba({r},{g},{bcol},{opacity:.3f})",
                        line=dict(color="rgba(0,0,0,0)"),
                        showlegend=False,
                        hoverinfo="skip",
                    ))
                traces.append(go.Scatter(
                    x=projection_dates, y=predicted, mode="lines",
                    name=m_name, line=dict(color=color, width=2),
                ))
            else:
                for _ in range(ANIM_N_BANDS):
                    traces.append(go.Scatter(x=[], y=[], showlegend=False, hoverinfo="skip"))
                traces.append(go.Scatter(
                    x=[], y=[], mode="lines", name=m_name, line=dict(color=color, width=2),
                ))
        traces.append(go.Scatter(
            x=sub["date"], y=sub["weight"], mode="markers+lines",
            name="Actual Weight",
            marker=dict(size=8, color="#ffffff"),
            line=dict(color="#ffffff", width=1, dash="dot"),
        ))
        return traces

    frames = [go.Frame(data=traces_for_n(n), name=str(n)) for n in n_values]
    initial_traces = list(frames[-1].data)

    y_min = float(df["weight"].min()) - 5
    y_max = float(df["weight"].max()) + 5

    fig = go.Figure(data=initial_traces, frames=frames)
    fig.update_layout(
        title=f"Evolution of fits as data accumulates (N = {ANIM_N_MIN} -> {n_max})",
        xaxis_title="Date",
        yaxis_title="Weight (kg)",
        yaxis=dict(range=[y_min, y_max]),
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        updatemenus=[{
            "type": "buttons",
            "direction": "left",
            "x": 0.0, "y": -0.15,
            "xanchor": "left", "yanchor": "top",
            "pad": {"r": 10, "t": 0},
            "buttons": [
                {
                    "label": label,
                    "method": "animate",
                    "args": [None, {
                        "frame": {"duration": int(700 / speed), "redraw": True},
                        "fromcurrent": False,
                        "transition": {"duration": int(500 / speed), "easing": "cubic-in-out"},
                    }],
                }
                for label, speed in [("0.5x", 0.5), ("1x", 1.0), ("2x", 2.0), ("4x", 4.0)]
            ] + [
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    }],
                },
            ],
        }],
        sliders=[{
            "active": len(frames) - 1,
            "x": 0.30, "y": -0.15, "len": 0.68,
            "xanchor": "left", "yanchor": "top",
            "currentvalue": {"prefix": "N = ", "visible": True, "xanchor": "right"},
            "transition": {"duration": 400, "easing": "cubic-in-out"},
            "steps": [
                {
                    "method": "animate",
                    "args": [[f.name], {
                        "mode": "immediate",
                        "frame": {"duration": 400, "redraw": True},
                        "transition": {"duration": 400, "easing": "cubic-in-out"},
                    }],
                    "label": f.name,
                }
                for f in frames
            ],
        }],
    )
    return fig


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
    anim_json = _build_animation_figure(df).to_json() if len(df) >= ANIM_N_MIN else None

    if anim_json:
        anim_html_block = (
            "<div class='section-caption'>Curve evolution as data accumulates "
            "&mdash; press Play or drag the slider</div>"
            "<div id='animation'></div>"
        )
        anim_script_block = (
            "var animFig = " + anim_json + ";\n"
            "        Plotly.newPlot('animation', animFig.data, animFig.layout, "
            "{responsive: true}).then(function() { "
            "if (animFig.frames && animFig.frames.length) { "
            "Plotly.addFrames('animation', animFig.frames); } });"
        )
    else:
        anim_html_block = ""
        anim_script_block = ""

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
        #chart, #animation {{
            width: 100%;
            height: 500px;
            background: #1a1a2e;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.4);
        }}
        #animation {{
            margin-top: 30px;
            height: 560px;
        }}
        .section-caption {{
            text-align: center;
            margin-top: 35px;
            margin-bottom: 8px;
            color: #a0a0c0;
            font-size: 0.95em;
            font-weight: 300;
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
    {anim_html_block}
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
        {anim_script_block}
    </script>
</body>
</html>"""

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
