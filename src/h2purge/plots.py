"""Matplotlib plotting utilities for validation and task outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io_utils import ensure_dir


def configure_matplotlib_cjk() -> bool:
    """Configure Matplotlib to render Chinese labels when Windows fonts exist."""

    preferred = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for font_path in preferred:
        path = Path(font_path)
        if path.exists():
            fm.fontManager.addfont(str(path))
            font_name = fm.FontProperties(fname=str(path)).get_name()
            mpl.rcParams["font.family"] = "sans-serif"
            mpl.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            mpl.rcParams["axes.unicode_minus"] = False
            return True
    mpl.rcParams["axes.unicode_minus"] = False
    return False


configure_matplotlib_cjk()


def _save(fig, path):
    p = Path(path)
    ensure_dir(p.parent)
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    return p


def plot_concentration_profiles(result, path, time_indices: Optional[Iterable[int]] = None, title: str = ""):
    if time_indices is None:
        time_indices = np.linspace(0, len(result.times) - 1, min(5, len(result.times)), dtype=int)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"H2": "#2563eb", "N2": "#1f9d55", "Air": "#e5533d"}
    for idx in time_indices:
        alpha = 0.35 + 0.65 * (idx / max(len(result.times) - 1, 1))
        for j, name in enumerate(result.species):
            ax.plot(
                result.x_grid / 1000.0,
                result.profiles[idx, :, j],
                color=colors[name],
                alpha=alpha,
                lw=1.2,
                label=f"{name} t={result.times[idx]:.0f}s" if idx in (0, len(result.times) - 1) else None,
            )
    ax.set_xlabel("\u7ba1\u9053\u4f4d\u7f6e x (km)")
    ax.set_ylabel("\u6469\u5c14\u5206\u6570 / \u4f53\u79ef\u5206\u6570")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title or "\u7ec4\u5206\u6cbf\u7a0b\u6d53\u5ea6\u5206\u5e03")
    ax.grid(True, alpha=0.25)
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(ncol=3, fontsize=8)
    return _save(fig, path)


def plot_metric_time(result, metric_key: str, path, ylabel: str, title: str = ""):
    df = pd.DataFrame(result.metrics)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df["time_s"] / 60.0, df[metric_key], lw=2.0, color="#2563eb")
    ax.set_xlabel("\u65f6\u95f4 (min)")
    ax.set_ylabel(ylabel)
    ax.set_title(title or ylabel)
    ax.grid(True, alpha=0.25)
    return _save(fig, path)


def plot_bar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    path,
    hue_col: str | None = None,
    title: str = "",
    xlabel: str | None = None,
    ylabel: str | None = None,
    hue_label: str | None = None,
):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    if hue_col is None:
        ax.bar(df[x_col].astype(str), df[y_col], color="#2f6db3", width=0.52)
    else:
        labels = sorted(df[x_col].unique())
        hues = sorted(df[hue_col].unique())
        width = 0.8 / max(len(hues), 1)
        xpos = np.arange(len(labels))
        for k, hue in enumerate(hues):
            sub = df[df[hue_col] == hue].set_index(x_col)
            vals = [sub.loc[label, y_col] if label in sub.index else np.nan for label in labels]
            hue_text = f"{float(hue):g}" if isinstance(hue, (int, float, np.integer, np.floating)) else str(hue)
            legend_label = hue_label.format(value=hue_text) if hue_label and "{value}" in hue_label else f"{hue_label or hue_col}={hue_text}"
            ax.bar(xpos + (k - (len(hues) - 1) / 2) * width, vals, width=width, label=legend_label)
        ax.set_xticks(xpos)
        ax.set_xticklabels([str(v) for v in labels])
        ax.legend(fontsize=8)
    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel or y_col)
    ax.set_title(title or y_col)
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _save(fig, path)


def plot_heatmap_table(
    df: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
    path,
    title: str = "",
    xlabel: str | None = None,
    ylabel: str | None = None,
    colorbar_label: str | None = None,
):
    pivot = df.pivot_table(index=index, columns=columns, values=values, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticklabels([str(i) for i in pivot.index])
    ax.set_xlabel(xlabel or columns)
    ax.set_ylabel(ylabel or index)
    ax.set_title(title or values)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.0f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label=colorbar_label or values)
    return _save(fig, path)
