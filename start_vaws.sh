#!/usr/bin/bash

VAWS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Trim the last dir from the path
export VAWS_DIR=${VAWS_DIR%/*}

echo $VAWS_DIR

export PYTHONPATH=${MP_DIR}/src:${PYTHONPATH}

echo ${PYTHONPATH}

echo "Starting vaws gui"
python gui/main.py

