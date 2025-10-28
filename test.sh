#!/bin/bash
set -e
. .env/bin/activate
export TESTNAME=; tox -v
#python -m unittest action_optimizer.tests.test_autofill.Tests.test_autofill
