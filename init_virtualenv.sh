#!/bin/bash
virtualenv -p python3 .env
. .env/bin/activate
pip install -r requirements.txt
