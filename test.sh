#!/bin/bash
set -e
. .env/bin/activate
export TESTNAME=; tox -v
