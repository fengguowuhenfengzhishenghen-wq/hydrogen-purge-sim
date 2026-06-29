#!/usr/bin/env bash
set -e
export WM_PROJECT_DIR=${WM_PROJECT_DIR:-/usr/share/openfoam}
export FOAM_ETC=${FOAM_ETC:-/usr/share/openfoam/etc}
blockMesh | tee log.blockMesh
checkMesh | tee log.checkMesh
scalarTransportFoam | tee log.scalarTransportFoam
foamToVTK | tee log.foamToVTK
