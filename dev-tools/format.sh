#!/bin/sh -e
set -x

autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place "text_machina" --exclude=__init__.py
isort "text_machina"
black "text_machina" -l 80
