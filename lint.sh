#!/bin/bash

flake8 . --count --statistics --max-complexity=14 --max-line-length=120 \
	--exclude='pubchem.py, Pubchem.py,venv' \
	--per-file-ignores="__init__.py:F401, missing_handler.py:F401"