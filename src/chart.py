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
    "#4DD0E1",  # Cyan - Thomas
]

ANIM_N_MIN = 6
ANIM_PROJ_POINTS = 80
ANIM_N_BANDS = 5
ANIM_INTERMEDIATE = 8
ANIM_FRAME_MS = 110
MODEL_ORDER = ["Thomas (2013)"]


def _hex_to_rgb(color):
    h = color.lstrip("#")
    return int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)


_BMI_KNOTS = [16.0, 18.5, 21.75, 25.0, 30.0, 35.0, 40.0]
_BMI_R = [180, 220, 60, 220, 220, 200, 160]
_BMI_G = [0, 200, 180, 200, 130, 60, 0]
_BMI_B = [0, 80, 60, 80, 50, 30, 0]


def _bmi_rgb(bmi: float) -> tuple:
    bmi = max(min(bmi, _BMI_KNOTS[-1]), _BMI_KNOTS[0])
    return (
        int(np.interp(bmi, _BMI_KNOTS, _BMI_R)),
        int(np.interp(bmi, _BMI_KNOTS, _BMI_G)),
        int(np.interp(bmi, _BMI_KNOTS, _BMI_B)),
    )


def _compute_y_range(df: pd.DataFrame, models: dict, last_day: float) -> list:
    """Y range covering data + full-fit +/-3 sigma bands (BMI-filtered models excluded)."""
    lows = [float(df["weight"].min())]
    highs = [float(df["weight"].max())]
    if models:
        projection_days = np.linspace(0, last_day, 200)
        last_data_day = df["days"].iloc[-1]
        extrapolation = np.maximum(projection_days - last_data_day, 0)
        scale = 1.0 + np.sqrt(extrapolation / max(last_data_day, 1.0))
        for model in models.values():
            if model is None:
                continue
            if model["predict"](last_day) < MIN_WEIGHT:
                continue
            predicted = np.array([model["predict"](d) for d in projection_days])
            std = model.get("residual_std", 0)
            lows.append(float(np.min(predicted - 3 * std * scale)))
            highs.append(float(np.max(predicted + 3 * std * scale)))
    span = max(highs) - min(lows)
    margin = max(span * 0.05, 0.5)
    return [min(lows) - margin, max(highs) + margin]


def _build_bmi_figure(df: pd.DataFrame, models: dict, x_range: list, weight_y_range: list) -> go.Figure:
    height_sq = HEIGHT_M ** 2
    bmi_y_range = [
        min(17.0, weight_y_range[0] / height_sq),
        max(33.0, weight_y_range[1] / height_sq),
    ]

    first_date = df["date"].iloc[0].to_pydatetime()
    last_day = (TARGET_DATE - first_date).days
    projection_days = np.linspace(0, last_day, 500)
    projection_dates = [first_date + timedelta(days=float(d)) for d in projection_days]

    n_stripes = 80
    stripe_edges = np.linspace(bmi_y_range[0], bmi_y_range[1], n_stripes + 1)
    shapes = []
    for i in range(n_stripes):
        bmi_mid = 0.5 * (stripe_edges[i] + stripe_edges[i + 1])
        rr, gg, bb = _bmi_rgb(bmi_mid)
        shapes.append(dict(
            type="rect",
            xref="paper", x0=0, x1=1,
            yref="y", y0=stripe_edges[i], y1=stripe_edges[i + 1],
            fillcolor=f"rgba({rr},{gg},{bb},0.22)",
            line=dict(width=0),
            layer="below",
        ))

    fig = go.Figure()
    color_idx = 0
    for name, model in models.items():
        if model["predict"](last_day) < MIN_WEIGHT:
            continue
        color = COLORS[color_idx % len(COLORS)]
        color_idx += 1
        predicted_bmi = np.array([model["predict"](d) for d in projection_days]) / height_sq
        std_bmi = model.get("residual_std", 0) / height_sq
        last_data_day = df["days"].iloc[-1]
        extrapolation = np.maximum(projection_days - last_data_day, 0)
        scale = 1.0 + np.sqrt(extrapolation / max(last_data_day, 1.0))
        rr, gg, bb = _hex_to_rgb(color)
        n_bands = 20
        for band_i in range(n_bands, 0, -1):
            outer = band_i / n_bands * 3.0
            inner = (band_i - 1) / n_bands * 3.0
            opacity = 0.15 * (np.exp(-0.5 * inner ** 2) - np.exp(-0.5 * outer ** 2))
            upper = predicted_bmi + outer * std_bmi * scale
            lower = predicted_bmi - outer * std_bmi * scale
            fig.add_trace(go.Scatter(
                x=projection_dates + projection_dates[::-1],
                y=np.concatenate([upper, lower[::-1]]).tolist(),
                fill="toself",
                fillcolor=f"rgba({rr},{gg},{bb},{opacity:.3f})",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            ))
        fig.add_trace(go.Scatter(
            x=projection_dates, y=predicted_bmi, mode="lines",
            name=name, line=dict(color=color, width=2),
        ))

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["weight"] / height_sq, mode="markers+lines",
        name="Actual BMI",
        marker=dict(size=8, color="#ffffff"),
        line=dict(color="#ffffff", width=1, dash="dot"),
    ))

    fig.update_layout(
        title=f"Ilaria - BMI Tracking (height {HEIGHT_M:.2f} m)",
        xaxis=dict(title="Date", range=x_range),
        yaxis=dict(title="BMI", range=bmi_y_range),
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        shapes=shapes,
    )
    return fig


def _build_animation_figure(df: pd.DataFrame, x_range: list, y_range: list) -> go.Figure:
    first_date = df["date"].iloc[0].to_pydatetime()
    last_day = (TARGET_DATE - first_date).days
    projection_days = np.linspace(0, last_day, ANIM_PROJ_POINTS)
    projection_dates = [first_date + timedelta(days=float(d)) for d in projection_days]
    band_x = projection_dates + projection_dates[::-1]

    n_max = len(df)
    color = COLORS[0]
    r, g, b = _hex_to_rgb(color)
    model_name = MODEL_ORDER[0]

    band_outer = [(i / ANIM_N_BANDS) * 3.0 for i in range(ANIM_N_BANDS, 0, -1)]
    band_inner = [((i - 1) / ANIM_N_BANDS) * 3.0 for i in range(ANIM_N_BANDS, 0, -1)]
    band_opacities = [
        0.15 * (np.exp(-0.5 * inner ** 2) - np.exp(-0.5 * outer ** 2))
        for inner, outer in zip(band_inner, band_outer)
    ]

    def fit_data_for_n(n):
        sub = df.iloc[:n]
        models = fit_all_models(sub["days"].values, sub["weight"].values)
        model = models.get(model_name)
        if model is None or np.isnan(model["predict"](last_day)):
            return None
        predicted = np.array([model["predict"](d) for d in projection_days])
        std = model.get("residual_std", 0)
        last_data_day_n = sub["days"].iloc[-1]
        extrapolation = np.maximum(projection_days - last_data_day_n, 0)
        scale = 1.0 + np.sqrt(extrapolation / max(last_data_day_n, 1.0))
        bands = []
        for outer in band_outer:
            bands.append({
                "upper": predicted + outer * std * scale,
                "lower": predicted - outer * std * scale,
            })
        return {"predicted": predicted, "bands": bands}

    cached = {n: fit_data_for_n(n) for n in range(ANIM_N_MIN, n_max + 1)}

    def blend(data_n, data_n1, t):
        if data_n is None or data_n1 is None:
            return None
        blended_bands = [
            {
                "upper": (1 - t) * b_n["upper"] + t * b_n1["upper"],
                "lower": (1 - t) * b_n["lower"] + t * b_n1["lower"],
            }
            for b_n, b_n1 in zip(data_n["bands"], data_n1["bands"])
        ]
        return {
            "predicted": (1 - t) * data_n["predicted"] + t * data_n1["predicted"],
            "bands": blended_bands,
        }

    def make_traces(data, marker_count):
        traces = []
        if data is not None:
            for band, opacity in zip(data["bands"], band_opacities):
                traces.append(go.Scatter(
                    x=band_x,
                    y=np.concatenate([band["upper"], band["lower"][::-1]]).tolist(),
                    fill="toself",
                    fillcolor=f"rgba({r},{g},{b},{opacity:.3f})",
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False,
                    hoverinfo="skip",
                ))
            traces.append(go.Scatter(
                x=projection_dates, y=data["predicted"], mode="lines",
                name=model_name, line=dict(color=color, width=2.5),
            ))
        else:
            for _ in range(ANIM_N_BANDS):
                traces.append(go.Scatter(x=[], y=[], showlegend=False, hoverinfo="skip"))
            traces.append(go.Scatter(
                x=[], y=[], mode="lines", name=model_name,
                line=dict(color=color, width=2.5),
            ))
        sub = df.iloc[:marker_count]
        traces.append(go.Scatter(
            x=sub["date"], y=sub["weight"], mode="markers+lines",
            name="Actual Weight",
            marker=dict(size=8, color="#ffffff"),
            line=dict(color="#ffffff", width=1, dash="dot"),
        ))
        return traces

    n_total_traces = ANIM_N_BANDS + 2
    animated_indices = list(range(n_total_traces))

    frames = []
    frames.append(go.Frame(
        data=make_traces(cached[ANIM_N_MIN], ANIM_N_MIN),
        traces=animated_indices,
        name=str(ANIM_N_MIN),
    ))
    for n in range(ANIM_N_MIN, n_max):
        data_n = cached[n]
        data_n1 = cached[n + 1]
        if data_n is None or data_n1 is None:
            continue
        for k in range(1, ANIM_INTERMEDIATE + 2):
            t = k / (ANIM_INTERMEDIATE + 1)
            blended = blend(data_n, data_n1, t)
            if k == ANIM_INTERMEDIATE + 1:
                marker_count = n + 1
                name = str(n + 1)
            else:
                marker_count = n
                name = f"{n}.{k}"
            frames.append(go.Frame(
                data=make_traces(blended, marker_count),
                traces=animated_indices,
                name=name,
            ))

    initial_data = list(frames[-1].data)
    fig = go.Figure(data=initial_data, frames=frames)
    fig.update_layout(
        title=f"Evolution of fits as data accumulates (N = {ANIM_N_MIN} -> {n_max})",
        xaxis=dict(title="Date", range=x_range),
        yaxis=dict(title="Weight (kg)", range=y_range),
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "direction": "left",
            "x": 0.0, "y": -0.1,
            "xanchor": "left", "yanchor": "top",
            "pad": {"r": 10, "t": 0},
            "bgcolor": "#16213e",
            "bordercolor": "#4DD0E1",
            "borderwidth": 1,
            "font": {"color": "#4DD0E1"},
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {
                        "frame": {"duration": ANIM_FRAME_MS, "redraw": False},
                        "fromcurrent": False,
                        "transition": {"duration": 0},
                    }],
                },
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

    x_range = [df["date"].iloc[0].to_pydatetime(), TARGET_DATE]
    y_range = _compute_y_range(df, models, last_day)

    fig.update_layout(
        title="Ilaria - Weight Tracking & Projections",
        xaxis=dict(title="Date", range=x_range),
        yaxis=dict(title="Weight (kg)", range=y_range),
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    plot_json = fig.to_json()
    anim_json = _build_animation_figure(df, x_range, y_range).to_json() if len(df) >= ANIM_N_MIN else None
    bmi_json = _build_bmi_figure(df, models, x_range, y_range).to_json()

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
        #chart, #animation, #bmi-chart {{
            width: 100%;
            height: 500px;
            background: #1a1a2e;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.4);
        }}
        #bmi-chart {{
            margin-top: 30px;
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
    <div class="section-caption">BMI tracking &mdash; green is the optimal range, red the extremes</div>
    <div id="bmi-chart"></div>
    <div class="footer">
        Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M")} |
        Data points: {len(df)} |
        Projection target: 2026-12-31
    </div>
    <script>
        var figure = {plot_json};
        Plotly.newPlot('chart', figure.data, figure.layout, {{responsive: true}});
        {anim_script_block}
        var bmiFig = {bmi_json};
        Plotly.newPlot('bmi-chart', bmiFig.data, bmiFig.layout, {{responsive: true}});
    </script>
</body>
</html>"""

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
