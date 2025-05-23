#!/bin/bash
# Run optimization analysis on spreadsheet.
set -e

if [[ -z "$ACTION_OPTIMIZER_DATAFILE" ]]; then
    echo "Environment variable ACTION_OPTIMIZER_DATAFILE not specified." 1>&2
    exit 1
fi

SCRIPT_DIR="$(dirname "$(readlink -f ${BASH_SOURCE[0]})")"
cd $SCRIPT_DIR/..

[ ! -d .env ] && ./init_virtualenv.sh

exit_code=0
.env/bin/python -m action_optimizer.optimizer analyze $ACTION_OPTIMIZER_DATAFILE --stop-on-error || exit_code=$?
exit $exit_code
# Outputs ensemble.ods.
