.. _pipeline_module:

Full pipeline (ProQSAR)
=======================

The ``proqsar.qsar.ProQSAR`` class provides a single-call, opinionated end-to-end QSAR workflow that
chains the core modules (standardization, featurization, preprocessing, splitting, feature selection,
model development, hyperparameter optimisation and evaluation) into a reproducible experiment.

The class is intended for quick experiments and reproducible benchmarking. Internally it uses the
modular building blocks available in :mod:`proqsar.Data`, :mod:`proqsar.Preprocessor` and
:mod:`proqsar.Model`.

Overview
--------
Typical steps performed by ``ProQSAR.run_all()``:

- Standardize SMILES and build RDKit molecules.
- Generate features (fingerprints / descriptors) via the Featurizer.
- Run the Preprocessor pipeline (duplicate / missing / low-variance / outlier handling, rescaling).
- Split the dataset (scaffold / stratified / random / time-based).
- Perform feature selection (optional; cross-validated).
- Benchmark candidate models and pick the best model family.
- Run hyperparameter optimisation for the selected model (Optuna by default).
- Evaluate final model on hold-out test set and save artifacts (model, CV results, metrics, plots).

API — usage
-----------

.. code-block:: python

   from proqsar.qsar import ProQSAR
   from proqsar.Config.debug import force_quiet
   import pandas as pd

   force_quiet()

   data = [
       {'Smiles': 'O=C(N[C@H]1CCc2ccccc21)c1nc(-c2cccs2)nc(O)c1O', 'pChEMBL': 7.69897, 'id': 0},
       {'Smiles': 'CN1Cc2c(c(O)c3ncc(Cc4ccc(F)c(Cl)c4)cc3c2N(C)S(C)(=O)=O)C1=O', 'pChEMBL': 6.57675, 'id': 1},
       ...
   ]

   pipeline = ProQSAR(
       activity_col='pChEMBL',
       id_col='id',
       smiles_col='Smiles',
       n_jobs=4,
       project_name='Project',
       scoring_target='r2',
       n_splits=3,
       n_repeats=1,
       random_state=42
   )

   # run the full pipeline (alpha controls statistical significance threshold; see description)
   result = pipeline.run_all(pd.DataFrame(data), alpha=0.05)

Parameters (key)
----------------
:activity_col:    Column name containing the target (continuous regression value e.g. pChEMBL).  
:id_col:          Column name used to track samples (kept for provenance).  
:smiles_col:      Column with raw SMILES strings to standardize.  
:n_jobs:          Number of parallel workers to use where supported.  
:project_name:    Folder/name used to save artifacts for this run.  
:scoring_target:  Metric used to compare models (e.g. ``r2``, ``rmse``, ``mae``).  
:n_splits:        Number of CV folds for benchmarking/selection.  
:n_repeats:       Number of repeats for repeated CV.  
:random_state:    Seed for deterministic behaviour across splits, selection and optimization.  
:alpha:           Significance threshold (p-value) used where the pipeline performs statistical
                  hypothesis testing (for example, paired comparisons during model selection or
                  feature-selection tests). Default is 0.05. If you do not need statistical tests,
                  set ``alpha=None`` or disable the relevant comparisons in a config.

Return value / Saved artifacts
------------------------------
The behaviour below describes the default and typical outputs. Exact keys may vary by release.

- **Return:** a dictionary-like `result` describing the experiment, commonly containing:
  - ``result['model']`` — the final fitted model object (sklearn-compatible or wrapped estimator).
  - ``result['best_model_name']`` — selected model family (e.g. ``'Ridge'``).
  - ``result['best_params']`` — best hyperparameters found by the optimizer.
  - ``result['metrics']`` — evaluation metrics on train/val/test (dict).
  - ``result['cv_results']`` — cross-validation results and comparison statistics.
  - ``result['feature_selector']`` — information about selected features (if selection used).
  - ``result['artifacts_path']`` — path to the saved run folder containing model, plots and logs.

- **Saved to disk** (``project_name`` folder):
  - fitted model (pickle/serialized form),
  - hyperparameter study (Optuna study file or CSV),
  - CV/validation and test metrics tables,
  - diagnostic plots (predicted vs observed, residuals, feature importances),
  - featurizer / preprocessor metadata so runs are reproducible.

Notes & recommendations
-----------------------
- **Reproducibility:** set ``random_state`` and keep the saved artifacts. The pipeline uses the same seed for splits,
  feature selection and optimisation to make runs repeatable.
- **Performance:** for larger datasets set ``n_jobs`` greater than 1. Be mindful of memory when `n_jobs` is high.
- **Small-sample caution:** with very small datasets (like the toy example above) CV statistics and optimisation
  are unreliable — use larger datasets for production modelling.
- **Alpha parameter:** used for statistical tests only. It does not change model hyperparameters by itself — instead,
  it controls how conservatively the pipeline accepts differences between competing models / features. Lower
  ``alpha`` → stricter evidence needed to prefer one model over another.

Troubleshooting
---------------
- If the run fails early, first check the **standardization** step: malformed SMILES will typically raise RDKit parse errors.
- If no features are returned, confirm RDKit (if using chemistry featurizers) and that the `smiles_col` is present.
- For long-running Optuna studies, use a smaller `n_splits`/`n_repeats` during debugging.

See Also
--------
- :mod:`proqsar.Data.Standardizer`  
- :mod:`proqsar.Data.Featurizer`  
- :mod:`proqsar.Preprocessor`  
- :mod:`proqsar.Model.FeatureSelector`  
- :mod:`proqsar.Model.ModelDeveloper`  
- :mod:`proqsar.Model.Optimizer`

Example — expected quick output
-------------------------------
A short example of what you might inspect after `run_all` returns:

.. code-block:: python

   print(result['best_model_name'])
   # >> "Ridge"

   print(result['best_params'])
   # >> {'alpha': 0.21685361128059533}

   print(result['metrics']['test'])
   # >> {'r2': 0.62, 'rmse': 0.45, 'mae': 0.33}

(Exact keys and values depend on the pipeline version and configuration.)

