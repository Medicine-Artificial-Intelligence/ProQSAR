.. _pipeline_module:

Full pipeline (ProQSAR)
=======================

The ``proqsar.qsar.ProQSAR`` class provides a single-call, opinionated end-to-end QSAR workflow that
chains the core modules (standardization, featurization, preprocessing, splitting, feature selection,
model development, hyperparameter optimisation and evaluation) into a reproducible experiment.

This example demonstrates a reproducible run: training + optimisation followed by inference with
prediction intervals and applicability-domain flags.

Training & optimisation (reproducible)
--------------------------------------

.. code-block:: python

   import pandas as pd
   from proqsar.qsar import ProQSAR
   from proqsar.Config.config import Config

   # small reproducible demo dataset (public)
   url = "https://raw.githubusercontent.com/Medicine-Artificial-Intelligence/ProQSAR/main/Data/testcase.csv"
   data = pd.read_csv(url).iloc[:50, :]   # use small sample for quick demo
   data['id'] = data.index

   # centralised config: change splitter / optimizer settings here
   cfgs = Config(
       splitter={'option': 'scaffold'},
       optimizer={'n_trials': 50}
   )

   # create pipeline — set random_state for reproducibility
   pipeline = ProQSAR(
       activity_col='pChEMBL',
       id_col='id',
       smiles_col='Smiles',
       n_jobs=4,
       project_name='Demo',
       scoring_target='r2',
       n_splits=5,
       n_repeats=5,
       config=cfgs,
       random_state=42
   )

   # run the full training + optimisation pipeline (alpha used for internal statistical tests)
   result = pipeline.run_all(pd.DataFrame(data), alpha=0.05)

Notes:
- For full reproducibility ensure: fixed ``random_state``; deterministic platform (same Python/RDKit/NumPy versions); and a fixed Optuna seed when running long studies (set in Config/optimizer if supported).
- Use a small ``n_trials`` and CV folds for quick debugging; increase for production.

Pipeline summary (object repr)
------------------------------

After successful training the pipeline prints a concise summary. Example:

.. code-block:: text

   ┌────────────────────────────────────────────────────────────────────┐
   │ ProQSAR Pipeline                                                   │
   ├────────────────────────────────────────────────────────────────────┤
   │ Project: Demo                                                      │
   │ Save Dir: Project/Demo                                             │
   │ Selected feature: 'RDK5'                                           │
   │ Fitted: True                                                       │
   │ Models registered: 1                                               │
   │ Selected model: 'XGBRegressor'                                     │
   │ CV (XGBRegressor): 0.683 ± 0.298                                   │
   │ n_jobs: 4                                                          │
   │ scoring_target: 'r2'                                               │
   │ Optimizer: enabled    ConfPred: enabled    AD: enabled             │
   └────────────────────────────────────────────────────────────────────┘

Key fields:
- ``Selected feature`` — name of the selected feature set (e.g. RDK5).  
- ``Selected model`` — model family chosen after benchmarking.  
- ``CV (...)`` — cross-validated score ± std (scoring_target).  
- ``Optimizer``, ``ConfPred``, ``AD`` — flags for hyperparameter search, conformal prediction, and applicability-domain checks.

Inference (predict using :class:`Inference`)
-------------------------------------------

Use the :class:`proqsar.infer.Inference` wrapper to run predictions with the same
output format returned by :meth:`pipeline.predict` (point predictions, conformal
prediction intervals and applicability-domain flags). The wrapper additionally
stores metadata about the last run and prints a compact summary when the
:class:`Inference` object is printed.

.. code-block:: python

   import pandas as pd
   from proqsar.infer import Inference

   # load test / new set
   url = "https://raw.githubusercontent.com/Medicine-Artificial-Intelligence/ProQSAR/main/Data/testcase.csv"
   test = pd.read_csv(url)
   # optional: create id column if you prefer an explicit id in output
   test["id"] = test.index

   # wrap the trained pipeline (inplace controls whether input DF may be modified)
   infer = Inference(pipeline, inplace=True)

   # run inference (id_key=None => index will be used; ground_truth optional)
   preds_df = infer.run(
       test,
       smiles_key="Smiles",
       id_key=None,             # None means infer will pass-through the index as `id`
       ground_truth="pChEMBL",  # provide if you want observed activity in output
       alpha=0.05               # 95% prediction intervals
   )

   # preds_df is the same-format table produced by pipeline.predict(...)
   print(preds_df.head())

Example output (first rows)
---------------------------

.. code-block:: text

   id    pChEMBL    Predicted value   Prediction Interval (alpha=0.05)    Applicability domain
   0     7.698970   6.720965          [4.429, 8.584]                          in
   1     6.576754   7.760520          [5.270, 9.386]                          in
   2     5.970000   6.018114          [4.426, 8.542]                          in
   3     5.602060   5.681627          [3.624, 7.740]                          in
   4     5.397940   5.718508          [3.785, 7.902]                          in
   ...
   49    5.761954   5.681627          [3.624, 7.740]                          in

Column explanations
-------------------
- ``id`` — input sample identifier (passed-through). When ``id_key=None``, the
  DataFrame index is used and passed through as ``id``.  
- ``pChEMBL`` — observed activity if present (passed-through / used for evaluation).  
- ``Predicted value`` — model point prediction (mean/median depending on estimator/wrapping).  
- ``Prediction Interval (alpha=0.05)`` — conformal prediction interval for the chosen ``alpha``.  
- ``Applicability domain`` — in/out flag indicating whether the sample lies within the model's AD.

Printing the Inference object (compact summary)
-----------------------------------------------
After running :meth:`Inference.run`, printing the :class:`Inference` object
displays a compact summary for the last run (row count, AD split, prediction
statistics and quantiles). Example:

.. code-block:: text

   ┌────────────────────────────────────────────────────────────────────────┐
   │ Inference (ProQSAR)                                                    │
   ├────────────────────────────────────────────────────────────────────────┤
   │ Project: Demo                                                          │
   │ Save Dir: Project/Demo                                                 │
   │ Selected feature: 'RDK5'                                               │
   │ Last run (rows): 50                                                    │
   │ Applicability domain column: Applicability domain                      │
   │ AD: in=50 (100.00%)  out=0                                             │
   │ Predictions — mean: 5.978  std: 1.208  nan%: 0.00%                     │
   │ Quantiles (10/50/90): 4.619 / 5.730 / 7.778                            │
   └────────────────────────────────────────────────────────────────────────┘

Notes
-----
- ``inplace``: when ``True`` the input DataFrame may be modified in-place; set
  ``inplace=False`` to preserve the original.  
- ``alpha``: conformal prediction level (e.g. ``0.05`` → 95% PI).  
- If you encounter a ``KeyError`` for ``smiles_key`` or ``id_key``, verify the
  input DataFrame column names and pass the correct keys to :meth:`Inference.run`.

Reproducibility checklist
-------------------------
- Fix ``random_state`` in the pipeline and any downstream components that accept a seed.  
- Pin environment versions (Python, RDKit, scikit-learn, xgboost/optuna)
- Save artifacts from runs (``pipeline.save_dir``) — the folder includes model, CV results, Optuna study and plots.  
- When comparing runs, keep ``alpha`` / CV settings / optimizer budget identical.

Troubleshooting
---------------
- If predictions are unexpectedly constant or many samples are marked ``out`` in AD, inspect:
  - preprocessing logs (duplicates / missing / low-variance steps),
  - feature generation (are features identical?),
  - applicability-domain thresholds / distance metric settings.
- If Optuna optimisation produces noisy outcomes, increase ``n_trials`` and/or use a deterministic sampler/seed.

