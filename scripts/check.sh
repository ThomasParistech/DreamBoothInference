#!/bin/bash

ROOT_DIR=$(dirname ${0})
cd $ROOT_DIR/..

unimport $(find . -name '*.py') --remove
absolufy-imports $(find . -name '*.py')
isort --py 310 --force-single-line-imports --line-length 120 --dont-order-by-type .
