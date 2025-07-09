#!/bin/bash
pip install -q pandas numpy matplotlib
mkdir -p tests/test_results

pytest -vv --tb=long tests > tests/test_results/test_results.txt
