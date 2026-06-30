#!/usr/bin/env bash
set -e
src=/usr/share/doc/openfoam-examples/examples/combustion/reactingFoam/RAS/DLR_A_LTS/constant
cp "$src/reactionsGRI" constant/reactionsGRI
cp "$src/thermo.compressibleGasGRI" constant/thermo.compressibleGasGRI
