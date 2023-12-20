#!/usr/bin/env bash

set -e
set -x

mypy "text_machina" 
flake8 "text_machina" --ignore=E501,W503,E203,E402
black "text_machina" --check -l 80
