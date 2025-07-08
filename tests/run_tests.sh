#!/bin/bash
mkdir -p tests/test_results

pytest -vv --tb=long tests/test_synthesizer.py                     > tests/test_results/test_synthesizer.txt