#!/bin/bash

flake8 . --count --statistics --max-complexity=14 --max-line-length=120 \
	--exclude='Pubchem.py, PubChem.py, pubchem.py' \
	--per-file-ignores='
		__init__.py:F401, 
		missing_handler.py:F401, 
		optimizer_utils.py:C901, 
		statistical_analysis.py:C901, 
		feature_selector_utils.py:C901
		model_validation.py:C901
		'