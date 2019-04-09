Action Optimizer
================

Installation
------------

    sudo apt-get install weka libsvm-java libsvm3-java
    ./init_virtualenv.sh

Development
-----------

Run tests with:

    tox

Run a specific test with:

    export TESTNAME=.test_causal_trend; tox
