# OpenFOAM 3D multi-species pipe shutdown case

Source 1D case: `DN1200_u7_p010_stop60`
Source window: `full_isolation_zone_1200m`

This is a runnable local 3D circular-pipe OpenFOAM case for shutdown
stratification review.

Physics included:

- solver: `rhoReactingFoam`;
- compressible perfect-gas multi-species mixture;
- species: H2/O2/N2, with Air split to O2+N2 before writing mass fractions;
- gravity: `(0 0 -9.81)`;
- chemistry/combustion disabled;
- closed inlet/outlet and no-slip pipe wall;
- initial velocity zero.

Mesh:

- local length: 48 m;
- diameter: 1.2 m;
- cells: 32 x 10 x 10 = 3200.

Run in WSL/OpenFOAM:

```bash
cd 'C:/Users/陈雨婷/Desktop/ppt/hydrogen_purge_sim/openfoam_cases/DN1200_u7_p010_stop60_rhoReactingPipe3D'
bash runAll.sh
```
