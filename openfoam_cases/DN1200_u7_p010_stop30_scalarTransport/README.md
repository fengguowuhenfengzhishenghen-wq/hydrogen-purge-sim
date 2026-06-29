# OpenFOAM scalarTransport local case

Source 1D stop case: `DN1200_u7_p010_stop30`

This is a minimal OpenFOAM `scalarTransportFoam` scaffold:

- Geometry: local 2D x-z slab, 1200 m by 1.2 m.
- Field `T`: initialized from the 1D H2 mole fraction in the full isolation window.
- Velocity `U`: zero shutdown field.
- Diffusivity `DT`: 7.8e-5 m2/s.
- Mesh: 240 x 1 x 48 cells.

This case validates OpenFOAM project structure, mesh generation, passive-scalar
shutdown diffusion, and VTK export. It is not yet a full buoyant multi-species
H2/N2/Air Navier-Stokes CFD model.
