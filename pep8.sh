#!/bin/bash
. .env/bin/activate
pylint --rcfile=pylint.rc action_optimizer setup.py
