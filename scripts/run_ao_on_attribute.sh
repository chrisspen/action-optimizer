#!/bin/bash
# Runs optimization analysis on a single attribute.
set -e

if [[ -z "$ACTION_OPTIMIZER_DATAFILE" ]]; then
    echo "Environment variable ACTION_OPTIMIZER_DATAFILE not specified." 1>&2
    exit 1
fi

SCRIPT_DIR="$(dirname "$(readlink -f ${BASH_SOURCE[0]})")"
cd $SCRIPT_DIR/..

[ ! -d .env ] && ./init_virtualenv.sh

echo "Analyzing attribute $1..."
.env/bin/python -m action_optimizer.optimizer $ACTION_OPTIMIZER_DATAFILE --only-attribute=$1
