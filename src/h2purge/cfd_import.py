"""Import local CFD/review results for Streamlit display.

This module only reads result files generated outside the Streamlit page. It
does not synthesize, estimate, or solve fields while the UI is rendering.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


IMAGE_FILENAMES = {
    "xz_slice_h2": "xz_slice_h2.png",
    "cross_section_h2": "cross_section_h2.png",
    "flammable_region": "flammable_region.png",
    "velocity_magnitude": "velocity_magnitude.png",
}


def list_cfd_cases(base_dir: str | Path = "outputs/cfd3d") -> list[Path]:
    """Return local CFD/review case directories below *base_dir*."""

    base = Path(base_dir)
    if not base.exists() or not base.is_dir():
        return []
    return sorted(path for path in base.iterdir() if path.is_dir())


def load_cfd_metrics(case_dir: str | Path) -> dict[str, Any]:
    """Load metrics.json from a local CFD/review case directory."""

    path = Path(case_dir) / "metrics.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_cfd_images(case_dir: str | Path) -> dict[str, Path]:
    """Return available image paths for a local CFD/review case directory."""

    case = Path(case_dir)
    images: dict[str, Path] = {}
    for key, filename in IMAGE_FILENAMES.items():
        path = case / filename
        if path.exists():
            images[key] = path
    return images


def find_cfd_volume_file(case_dir: str | Path) -> Path | None:
    """Return optional VTK/VTU result file if present."""

    case = Path(case_dir)
    for filename in (
        "openfoam_scalarTransport_300.vtu",
        "cfd_result.vtu",
        "cfd_result.vtk",
    ):
        path = case / filename
        if path.exists():
            return path
    return None
