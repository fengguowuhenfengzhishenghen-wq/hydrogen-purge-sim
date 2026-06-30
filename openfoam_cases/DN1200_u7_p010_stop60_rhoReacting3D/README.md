# Offline 3D multi-species CFD package

Source 1D case: `DN1200_u7_p010_stop60`
Source window: `full_isolation_zone_1200m`

This directory is a prepared offline 3D CFD package, not solved output.

Prepared files:

- `initial_3d_samples.csv`: cylindrical 3D sample points with mole fractions `x_*` and mass fractions `Y_*`.
- `initial_3d_points.vtk`: ParaView-readable point cloud for checking the mapped 3D initial field.
- `initial_3d_preview.png`: 3D preview of the H2 mole fraction field.
- `metrics.json`: field consistency and package metadata.
- `openfoam_solver_notes.md`: recommended OpenFOAM setup route.

Species convention:

The 1D model uses H2/N2/Air mole fractions.  For external CFD, Air is split into
O2 and N2:

```text
x_O2 = 0.21 x_Air
x_N2,total = x_N2 + 0.79 x_Air
Y_i = x_i M_i / sum_j(x_j M_j)
```

Current status: `prepared_input_not_solved`.

The Streamlit app should show this package as an offline 3D CFD input package
until a real solver writes `outputs/cfd3d/<case>/metrics.json` and result
images/VTK files.
