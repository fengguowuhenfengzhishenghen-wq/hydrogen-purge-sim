"""Small filesystem helpers for experiment outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def output_dir(*parts: str) -> Path:
    return ensure_dir(project_root() / "outputs" / Path(*parts))


def save_rows_csv(rows: list[dict], path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    pd.DataFrame(rows).to_csv(p, index=False)
    return p
