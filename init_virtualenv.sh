#!/bin/bash
set -e
[ -d .env] && rm -Rf .env
virtualenv -p python3.7 .env
. .env/bin/activate
pip install -r requirements.txt
