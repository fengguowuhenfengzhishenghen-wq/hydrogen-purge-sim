"""Streamlit pipe animation helpers.

This module is display-only. It maps the verified 1D mole-fraction profiles to
a stable representative-particle animation and must not be treated as CFD.
"""

from __future__ import annotations

import json

import numpy as np
import streamlit.components.v1 as components

SPECIES_COLORS = {"H2": "#2563eb", "N2": "#1f9d55", "Air": "#e5533d"}


def _mix_color(h2: float, n2: float, air: float) -> str:
    r = int(37 * h2 + 31 * n2 + 229 * air)
    g = int(99 * h2 + 157 * n2 + 83 * air)
    b = int(235 * h2 + 85 * n2 + 61 * air)
    return f"rgb({r},{g},{b})"


def pipe_frame_payload(result, particles: int) -> dict:
    """Build deterministic animation frames from 1D mole-fraction profiles."""

    cell_ids = np.linspace(0, len(result.x_grid) - 1, 160, dtype=int)
    n_particles = min(int(particles), 520)
    rng = np.random.default_rng(2026)
    particle_x = rng.uniform(0.05, 0.95, n_particles)
    particle_y = np.clip(0.50 + rng.normal(0.0, 0.16, n_particles), 0.20, 0.80)
    particle_q = rng.uniform(0.0, 1.0, n_particles)
    particle_r = rng.uniform(1.7, 3.1, n_particles)
    particle_a = rng.uniform(0.24, 0.44, n_particles)
    particle_phase = rng.uniform(0.0, 2.0 * np.pi, n_particles)
    frames = []
    for idx, prof in enumerate(result.profiles):
        metric = result.metrics[idx]
        cell_colors = [_mix_color(*prof[k]) for k in cell_ids]
        fr = float(metric.get("Fr", 3.0))
        strat = max(0.0, min(1.0, (3.0 - fr) / 3.0))
        dots = []
        drift = 0.010 * np.sin(0.16 * idx + particle_phase)
        x_now = np.clip(particle_x + drift, 0.05, 0.95)
        y_now = np.clip(particle_y + 0.025 * np.sin(0.21 * idx + particle_phase), 0.18, 0.82)
        picks = np.clip(((x_now - 0.05) / 0.90 * (len(prof) - 1)).astype(int), 0, len(prof) - 1)
        for j, k in enumerate(picks):
            h2, n2, air = prof[k]
            q = particle_q[j]
            if q <= h2:
                comp = "H2"
            elif q <= h2 + n2:
                comp = "N2"
            else:
                comp = "Air"
            y = y_now[j] + (comp == "Air") * 0.070 * strat - (comp == "H2") * 0.105 * strat
            dots.append(
                {
                    "x": float(x_now[j]),
                    "y": float(np.clip(y, 0.18, 0.82)),
                    "c": SPECIES_COLORS[comp],
                    "r": float(particle_r[j]),
                    "a": float(particle_a[j]),
                }
            )
        frames.append(
            {
                "time": float(result.times[idx] / 60.0),
                "cells": cell_colors,
                "dots": dots,
                "mixed": float(metric["mixed_length_m"]),
                "flammable": float(metric["flammable_length_m"]),
                "n2": float(metric["effective_n2_length_m"]),
                "fr": float(metric["Fr"]),
            }
        )
    return {"initial": int(len(result.times) // 2), "frames": frames}


def render_pipe(result, idx: int, particles: int) -> int:
    payload = pipe_frame_payload(result, particles)
    payload["initial"] = int(idx)
    data = json.dumps(payload, ensure_ascii=False)
    html = f"""
    <div id="app" style="font-family:Arial,'Microsoft YaHei',sans-serif;border:1px solid #d4deea;border-radius:12px;background:#eef5fc;padding:16px;color:#344054;">
      <div style="display:flex;justify-content:space-between;font-weight:700;font-size:18px;">
        <div>\u5165\u53e3<br><span style="font-weight:400;font-size:15px;">0 km</span></div>
        <div style="text-align:right;">\u51fa\u53e3<br><span style="font-weight:400;font-size:15px;">12 km</span></div>
      </div>
      <canvas id="pipeCanvas" width="1200" height="330" style="width:100%;height:330px;display:block;"></canvas>
      <div style="display:flex;gap:12px;align-items:center;margin-top:8px;flex-wrap:wrap;">
        <button id="playBtn" style="border:1px solid #b8c4d4;border-radius:7px;background:white;padding:7px 18px;cursor:pointer;">\u64ad\u653e</button>
        <input id="frameSlider" type="range" min="0" max="0" value="0" step="1" style="flex:1;min-width:240px;">
        <span id="timeLabel" style="font-weight:700;min-width:92px;"></span>
      </div>
      <div id="legend" style="display:flex;gap:16px;flex-wrap:wrap;margin-top:10px;font-size:15px;">
        <span><b style="color:{SPECIES_COLORS['H2']}">\u25a0</b> H2</span>
        <span><b style="color:{SPECIES_COLORS['N2']}">\u25a0</b> N2</span>
        <span><b style="color:{SPECIES_COLORS['Air']}">\u25a0</b> Air</span>
        <span id="metrics"></span>
      </div>
      <div style="font-size:13px;margin-top:8px;color:#667085;">3D \u7ba1\u9053\u4e3a\u4e00\u7ef4\u6469\u5c14\u5206\u6570\u573a\u7684\u53ef\u89c6\u5316\u6620\u5c04\uff0c\u4e0d\u662f CFD \u6c42\u89e3\u7ed3\u679c\u3002</div>
    </div>
    <script>
    const payload = {data};
    const frames = payload.frames;
    const canvas = document.getElementById('pipeCanvas');
    const ctx = canvas.getContext('2d');
    const slider = document.getElementById('frameSlider');
    const playBtn = document.getElementById('playBtn');
    const timeLabel = document.getElementById('timeLabel');
    const metrics = document.getElementById('metrics');
    let frame = Math.max(0, Math.min(payload.initial, frames.length - 1));
    let timer = null;
    slider.max = frames.length - 1;
    slider.value = frame;

    function roundedRect(x, y, w, h, r) {{
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.lineTo(x + w - r, y);
      ctx.quadraticCurveTo(x + w, y, x + w, y + r);
      ctx.lineTo(x + w, y + h - r);
      ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
      ctx.lineTo(x + r, y + h);
      ctx.quadraticCurveTo(x, y + h, x, y + h - r);
      ctx.lineTo(x, y + r);
      ctx.quadraticCurveTo(x, y, x + r, y);
      ctx.closePath();
    }}

    function draw(i) {{
      const f = frames[i];
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#eef5fc';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      const x0 = 70, y0 = 122, w = 1060, h = 74, r = 37;

      ctx.save();
      ctx.shadowColor = 'rgba(51, 65, 85, 0.18)';
      ctx.shadowBlur = 10;
      ctx.shadowOffsetY = 4;
      roundedRect(x0, y0, w, h, r);
      ctx.fillStyle = 'rgba(255,255,255,0.28)';
      ctx.fill();
      ctx.restore();

      ctx.save();
      roundedRect(x0, y0, w, h, r);
      ctx.clip();
      const gasGrad = ctx.createLinearGradient(x0, y0, x0 + w, y0);
      for (let j = 0; j < f.cells.length; j++) {{
        gasGrad.addColorStop(j / Math.max(1, f.cells.length - 1), f.cells[j]);
      }}
      const fog = ctx.createLinearGradient(x0, y0, x0, y0 + h);
      fog.addColorStop(0.00, 'rgba(255,255,255,0.50)');
      fog.addColorStop(0.18, 'rgba(255,255,255,0.08)');
      fog.addColorStop(0.55, 'rgba(255,255,255,0.00)');
      fog.addColorStop(1.00, 'rgba(15,23,42,0.10)');
      ctx.globalAlpha = 0.30;
      ctx.fillStyle = gasGrad;
      ctx.fillRect(x0, y0 + 4, w, h - 8);
      ctx.globalAlpha = 0.34;
      ctx.fillStyle = fog;
      ctx.fillRect(x0, y0, w, h);
      ctx.globalAlpha = 0.06;
      ctx.strokeStyle = '#e2e8f0';
      ctx.lineWidth = 1;
      for (let s = 0; s < 10; s++) {{
        const yy = y0 + 8 + s * (h - 16) / 9;
        ctx.beginPath();
        ctx.moveTo(x0 + 18, yy);
        ctx.lineTo(x0 + w - 18, yy);
        ctx.stroke();
      }}
      ctx.globalAlpha = 1.0;
      ctx.restore();
      ctx.strokeStyle = '#6b7d92';
      ctx.lineWidth = 3;
      roundedRect(x0, y0, w, h, r);
      ctx.stroke();
      ctx.strokeStyle = 'rgba(255,255,255,0.70)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x0 + 42, y0 + 9);
      ctx.lineTo(x0 + w - 42, y0 + 9);
      ctx.stroke();
      ctx.strokeStyle = 'rgba(107,125,146,0.45)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x0 + 20, y0 + h + 22);
      ctx.lineTo(x0 + w - 20, y0 + h + 22);
      ctx.stroke();
      for (const p of f.dots) {{
        ctx.beginPath();
        ctx.fillStyle = p.c;
        ctx.globalAlpha = p.a;
        ctx.arc(70 + p.x * 1060, 122 + p.y * 74, p.r, 0, Math.PI * 2);
        ctx.fill();
      }}
      ctx.globalAlpha = 1.0;
      timeLabel.textContent = f.time.toFixed(1) + ' min';
      metrics.textContent = `\u6df7\u6c14\u6bb5 ${{f.mixed.toFixed(0)}} m | \u53ef\u71c3\u98ce\u9669\u6bb5 ${{f.flammable.toFixed(0)}} m | \u6709\u6548 N2 ${{f.n2.toFixed(0)}} m | Fr ${{f.fr.toFixed(2)}}`;
      slider.value = i;
    }}

    function stop() {{
      if (timer) clearInterval(timer);
      timer = null;
      playBtn.textContent = '\u64ad\u653e';
    }}

    function play() {{
      stop();
      playBtn.textContent = '\u6682\u505c';
      timer = setInterval(() => {{
        frame += 1;
        if (frame >= frames.length) frame = 0;
        draw(frame);
      }}, 180);
    }}

    playBtn.onclick = () => timer ? stop() : play();
    slider.oninput = (e) => {{
      stop();
      frame = parseInt(e.target.value, 10);
      draw(frame);
    }};
    draw(frame);
    </script>
    """
    components.html(html, height=475)
    return idx
