#!/usr/bin/env bash

VAWS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo $VAWS_DIR

export PYTHONPATH=${VAWS_DIR}/src:${PYTHONPATH}
echo ${PYTHONPATH}

echo "Starting vaws gui"
python ${VAWS_DIR}/src/vaws/simulation.py "$@"

