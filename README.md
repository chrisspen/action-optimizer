[![CI](https://github.com/chrisspen/action-optimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/chrisspen/action-optimizer/actions/)

Action Optimizer
================

Installation
------------

    sudo apt-get install weka libsvm-java libsvm3-java
    ./init_virtualenv.sh

Usage
-----

    set -e
    [ ! -d .env ] && ./init_virtualenv.sh
    .env/bin/python -m action_optimizer.optimizer path/to/spreadsheet.ods

Development
-----------

Run tests with:

    tox

Run a specific test with:

    export TESTNAME=.test_causal_trend; tox
