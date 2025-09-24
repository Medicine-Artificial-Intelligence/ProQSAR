.. _model_module:

Model Module
============

The ``proqsar.Model`` module provides components for feature selection, model development, and hyperparameter optimization.  
Each class is scikit-learn compatible and follows the ``fit`` / ``transform`` (or ``fit`` / ``predict``) API.

FeatureSelector
---------------

The ``FeatureSelector`` automates feature selection across different strategies.

.. code-block:: python

   from proqsar.Model.FeatureSelector.feature_selector import FeatureSelector

   feat_sel = FeatureSelector(
       activity_col='pChEMBL',
       id_col='id',
       cross_validate=True,
       n_jobs=2,
       n_splits=2,
       n_repeats=3,
       random_state=42,
       select_method=[
           "NoFS",
           "Anova",
           "RandomForestRegressor",
           "ExtraTreesRegressor",
       ]
   )

   feat_sel.fit(train_clean)
   print(feat_sel.select_method)
   # >> "Anova"

   train_feat = feat_sel.transform(train_clean)
   test_feat = feat_sel.transform(test_clean)

ModelDeveloper
--------------

The ``ModelDeveloper`` evaluates multiple algorithms and selects the best-performing model.

.. code-block:: python

   from proqsar.Model.ModelDeveloper.model_developer import ModelDeveloper

   model = ModelDeveloper(
       activity_col='pChEMBL',
       id_col='id',
       cross_validate=True,
       n_jobs=2,
       n_splits=2,
       n_repeats=3,
       random_state=42,
       select_model=[
           "SVR",
           "Ridge",
           "RandomForestRegressor",
           "ExtraTreesRegressor",
       ]
   )

   model.fit(train_feat)
   print(model.select_model)
   # >> "Ridge"

Optimizer
---------

The ``Optimizer`` performs hyperparameter optimization (via Optuna) for the selected model.

.. code-block:: python

   from proqsar.Model.Optimizer.optimizer import Optimizer

   optimizer = Optimizer(
       activity_col="pChEMBL",
       id_col="id",
       scoring="r2",
       study_name="study_regression",
       select_model="Ridge",
       n_splits=2,  # small config
       n_repeats=3
   )

   best_params, best_score = optimizer.optimize(train_feat)
   print(best_params)
   # >> {'alpha': 0.21685361128059533}

Summary
-------

- **FeatureSelector** → tests multiple feature-selection methods.  
- **ModelDeveloper** → benchmarks candidate models and picks the best.  
- **Optimizer** → tunes hyperparameters for the chosen model.  

Together, these components allow you to build, evaluate, and refine predictive QSAR models in a reproducible pipeline.

See Also
--------

- :mod:`proqsar.Model.FeatureSelector.feature_selector`  
- :mod:`proqsar.Model.ModelDeveloper.model_developer`  
- :mod:`proqsar.Model.Optimizer.optimizer`  