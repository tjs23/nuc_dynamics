#!/bin/sh

export NUC_DYNAMICS_DIR="$(cd $(dirname "$0"); pwd)"
export PYTHONPATH="${PYTHONPATH}:.:${NUC_DYNAMICS_DIR}"

python -m nuc_dynamics.main $*

