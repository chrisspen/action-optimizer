#!/bin/bash
set -e
. .env/bin/activate
./pep8.sh
export TESTNAME=; tox
