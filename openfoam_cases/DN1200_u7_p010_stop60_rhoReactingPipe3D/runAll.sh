#!/usr/bin/env bash
set -eo pipefail
export WM_PROJECT_DIR=${WM_PROJECT_DIR:-/usr/share/openfoam}
export FOAM_ETC=${FOAM_ETC:-/usr/share/openfoam/etc}
bash constant/COPY_FROM_OPENFOAM_TEMPLATE.sh
blockMesh > log.blockMesh 2>&1
checkMesh > log.checkMesh 2>&1
rhoReactingFoam > log.rhoReactingFoam 2>&1
foamToVTK > log.foamToVTK 2>&1
