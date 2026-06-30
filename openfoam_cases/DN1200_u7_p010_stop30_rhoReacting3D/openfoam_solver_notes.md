# Recommended offline OpenFOAM route

Use this package as an initialization source for a real 3D circular-pipe CFD
case.  The intended solver family is a compressible transient flow solver with
multi-species transport, for example a `rhoReactingFoam`/`reactingFoam`-style
case with reactions disabled, or an equivalent custom solver.

Minimum physics to include:

- transient compressible or low-Mach variable-density momentum equation;
- gravity `g = (0 0 -9.81)`;
- H2/N2/O2 species transport;
- ideal-gas mixture density;
- no-slip pipe wall;
- closed inlet/outlet for shutdown stratification review;
- initial mass fractions from `initial_3d_samples.csv`.

Recommended workflow:

```bash
# 1. Build/import a circular local pipe mesh in OpenFOAM/Fluent.
# 2. Map initial_3d_samples.csv onto mesh cell centres.
# 3. Run 300 s shutdown stratification simulation.
# 4. Export:
#    outputs/cfd3d/<case>/
#      metrics.json
#      xz_slice_h2.png
#      cross_section_h2.png
#      velocity_magnitude.png
#      flammable_region.png
#      cfd_result.vtu
```

Do not rename this prepared package as solved CFD output until the external
solver has actually run.
