#!/bin/bash
set -e
./pep8.sh
export TESTNAME=; tox
