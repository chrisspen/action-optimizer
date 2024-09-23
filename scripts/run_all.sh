#!/bin/bash
# Runs entire pipeline.
set -e

if [[ -z "$ACTION_OPTIMIZER_DATAFILE" ]]; then
    echo "Environment variable ACTION_OPTIMIZER_DATAFILE not specified." 1>&2
    exit 1
fi

SCRIPT_DIR="$(dirname "$(readlink -f ${BASH_SOURCE[0]})")"
cd $SCRIPT_DIR/..

[ ! -d .env ] && ./init_virtualenv.sh

. .env/bin/activate

cd $SCRIPT_DIR

./run_ao.sh &
./run_ao_pcc.sh &
wait

cd $SCRIPT_DIR
./combine_reports.py
