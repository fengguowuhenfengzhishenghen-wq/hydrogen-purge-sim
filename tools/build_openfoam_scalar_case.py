"""Build a minimal OpenFOAM scalarTransportFoam local case.

The case transports a scalar T initialized from the 1D H2 mole-fraction profile
in the local full-isolation window. It is an OpenFOAM engineering scaffold for
the local shutdown window, not a full H2/N2/Air buoyant multi-species CFD model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


HEADER = r"""/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v1912                                 |
|   \\  /    A nd           |                                                 |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
"""


def field_header(cls: str, obj: str) -> str:
    return (
        HEADER
        + f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {obj};
}}
// ************************************************************************* //
"""
    )


def dict_header(obj: str) -> str:
    return (
        HEADER
        + f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      {obj};
}}
// ************************************************************************* //
"""
    )


def build(case_id: str, nx: int = 240, nz: int = 48) -> Path:
    source = ROOT / "cfd_cases" / case_id / "windows" / "full_isolation_zone_1200m" / "initial_1d_profile.csv"
    if not source.exists():
        raise FileNotFoundError(source)
    df = pd.read_csv(source)
    L = float(df["x_local_m"].max() - df["x_local_m"].min())
    D = 1.2
    thickness = 0.02
    case = ROOT / "openfoam_cases" / f"{case_id}_scalarTransport"
    for sub in ["0", "constant", "system"]:
        (case / sub).mkdir(parents=True, exist_ok=True)

    x_centres = (np.arange(nx) + 0.5) * L / nx
    h2 = np.interp(x_centres, df["x_local_m"].to_numpy(), df["x_H2_molfrac"].to_numpy())
    values = []
    # OpenFOAM blockMesh cell order for a single hex block is x fastest, then y, then z.
    for _iz in range(nz):
        for _iy in range(1):
            for ix in range(nx):
                values.append(float(h2[ix]))

    write_block_mesh(case / "system" / "blockMeshDict", L, D, thickness, nx, nz)
    write_control(case / "system" / "controlDict")
    write_fv_schemes(case / "system" / "fvSchemes")
    write_fv_solution(case / "system" / "fvSolution")
    write_transport(case / "constant" / "transportProperties")
    write_U(case / "0" / "U")
    write_T(case / "0" / "T", values)
    write_readme(case / "README.md", case_id, nx, nz)
    write_run(case / "runAll.sh")
    write_run(case / "cleanAll.sh", clean=True)
    return case


def write_block_mesh(path: Path, L: float, D: float, thickness: float, nx: int, nz: int) -> None:
    y0, y1 = -thickness / 2.0, thickness / 2.0
    z0, z1 = -D / 2.0, D / 2.0
    text = f"""{dict_header("blockMeshDict")}
convertToMeters 1;

vertices
(
    (0 {y0} {z0})
    ({L} {y0} {z0})
    ({L} {y1} {z0})
    (0 {y1} {z0})
    (0 {y0} {z1})
    ({L} {y0} {z1})
    ({L} {y1} {z1})
    (0 {y1} {z1})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} 1 {nz}) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    inlet
    {{
        type patch;
        faces ((0 4 7 3));
    }}
    outlet
    {{
        type patch;
        faces ((1 2 6 5));
    }}
    lowerWall
    {{
        type wall;
        faces ((0 3 2 1));
    }}
    upperWall
    {{
        type wall;
        faces ((4 5 6 7));
    }}
    frontAndBack
    {{
        type empty;
        faces ((0 1 5 4) (3 7 6 2));
    }}
);

mergePatchPairs
(
);
"""
    path.write_text(text, encoding="utf-8")


def write_control(path: Path) -> None:
    text = f"""{dict_header("controlDict")}
application     scalarTransportFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         300;
deltaT          1;
writeControl    timeStep;
writeInterval   60;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""
    path.write_text(text, encoding="utf-8")


def write_fv_schemes(path: Path) -> None:
    text = f"""{dict_header("fvSchemes")}
ddtSchemes
{{
    default         Euler;
}}
gradSchemes
{{
    default         Gauss linear;
}}
divSchemes
{{
    default         none;
    div(phi,T)      Gauss linearUpwind grad(T);
}}
laplacianSchemes
{{
    default         none;
    laplacian(DT,T) Gauss linear corrected;
}}
interpolationSchemes
{{
    default         linear;
}}
snGradSchemes
{{
    default         corrected;
}}
"""
    path.write_text(text, encoding="utf-8")


def write_fv_solution(path: Path) -> None:
    text = f"""{dict_header("fvSolution")}
solvers
{{
    T
    {{
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-10;
        relTol          0;
    }}
}}
SIMPLE
{{
}}
PISO
{{
}}
"""
    path.write_text(text, encoding="utf-8")


def write_transport(path: Path) -> None:
    text = (
        field_header("dictionary", "transportProperties")
        + """
DT              [0 2 -1 0 0 0 0] 7.8e-05;
"""
    )
    path.write_text(text, encoding="utf-8")


def write_U(path: Path) -> None:
    text = f"""{field_header("volVectorField", "U")}
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);
boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform (0 0 0);
    }}
    outlet
    {{
        type            fixedValue;
        value           uniform (0 0 0);
    }}
    lowerWall
    {{
        type            fixedValue;
        value           uniform (0 0 0);
    }}
    upperWall
    {{
        type            fixedValue;
        value           uniform (0 0 0);
    }}
    frontAndBack
    {{
        type            empty;
    }}
}}
"""
    path.write_text(text, encoding="utf-8")


def write_T(path: Path, values: list[float]) -> None:
    vals = "\n".join(f"{v:.8g}" for v in values)
    text = f"""{field_header("volScalarField", "T")}
dimensions      [0 0 0 0 0 0 0];
internalField   nonuniform List<scalar>
{len(values)}
(
{vals}
)
;
boundaryField
{{
    inlet
    {{
        type            zeroGradient;
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    lowerWall
    {{
        type            zeroGradient;
    }}
    upperWall
    {{
        type            zeroGradient;
    }}
    frontAndBack
    {{
        type            empty;
    }}
}}
"""
    path.write_text(text, encoding="utf-8")


def write_readme(path: Path, case_id: str, nx: int, nz: int) -> None:
    text = f"""# OpenFOAM scalarTransport local case

Source 1D stop case: `{case_id}`

This is a minimal OpenFOAM `scalarTransportFoam` scaffold:

- Geometry: local 2D x-z slab, 1200 m by 1.2 m.
- Field `T`: initialized from the 1D H2 mole fraction in the full isolation window.
- Velocity `U`: zero shutdown field.
- Diffusivity `DT`: 7.8e-5 m2/s.
- Mesh: {nx} x 1 x {nz} cells.

This case validates OpenFOAM project structure, mesh generation, passive-scalar
shutdown diffusion, and VTK export. It is not yet a full buoyant multi-species
H2/N2/Air Navier-Stokes CFD model.
"""
    path.write_text(text, encoding="utf-8")


def write_run(path: Path, clean: bool = False) -> None:
    if clean:
        text = """#!/usr/bin/env bash
set -e
rm -rf constant/polyMesh [1-9]* 0.* postProcessing VTK log.*
"""
    else:
        text = """#!/usr/bin/env bash
set -e
export WM_PROJECT_DIR=${WM_PROJECT_DIR:-/usr/share/openfoam}
export FOAM_ETC=${FOAM_ETC:-/usr/share/openfoam/etc}
blockMesh | tee log.blockMesh
checkMesh | tee log.checkMesh
scalarTransportFoam | tee log.scalarTransportFoam
foamToVTK | tee log.foamToVTK
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", default="DN1200_u7_p010_stop60")
    parser.add_argument("--nx", type=int, default=240)
    parser.add_argument("--nz", type=int, default=48)
    args = parser.parse_args()
    print(build(args.case_id, nx=args.nx, nz=args.nz))


if __name__ == "__main__":
    main()
