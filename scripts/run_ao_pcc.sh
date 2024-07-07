#!/bin/bash
# Run Pearson correlation coefficient analysis on all attributes.
set -e

if [[ -z "$ACTION_OPTIMIZER_DATAFILE" ]]; then
    echo "Environment variable ACTION_OPTIMIZER_DATAFILE not specified." 1>&2
    exit 1
fi

SCRIPT_DIR="$(dirname "$(readlink -f ${BASH_SOURCE[0]})")"
cd $SCRIPT_DIR/..

[ ! -d .env ] && ./init_virtualenv.sh

echo "Calculating Pearson correlation coefficients..."
exit_code=0
.env/bin/python -m action_optimizer.optimizer analyze $ACTION_OPTIMIZER_DATAFILE --calculate-pcc || exit_code=$?
exit $exit_code
