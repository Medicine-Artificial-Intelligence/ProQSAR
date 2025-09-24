.. _preprocessor_module:

Preprocessor Module
===================

The ``proqsar.Preprocessor`` module provides a collection of scikit-learn–compatible transformers for cleaning and normalizing QSAR datasets.  
All classes implement the ``fit`` / ``transform`` API, so they can be used independently or chained inside a ``Pipeline``.

Individual Handlers
-------------------

DuplicateHandler
~~~~~~~~~~~~~~~~
Removes duplicate **feature rows**.

- Compares the feature matrix for identical entries.
- ``id_col`` and ``activity_col`` are tracked but not used to define duplicates.
- If duplicate rows have conflicting activity values, they are flagged or dropped.
- Ensures only unique feature–activity pairs are retained.

.. code-block:: python

   from proqsar.Preprocessor.Clean import DuplicateHandler

   dup = DuplicateHandler(activity_col='pChEMBL', id_col='id')
   train_no_dup = dup.fit_transform(train)

MissingHandler
~~~~~~~~~~~~~~
Handles missing values in the dataset.

- Inspects feature columns for NaNs or null values.
- Removes or imputes rows depending on configuration.
- ``id_col`` and ``activity_col`` are preserved for traceability.

.. code-block:: python

   from proqsar.Preprocessor.Clean import MissingHandler

   miss = MissingHandler(activity_col='pChEMBL', id_col='id')
   train_no_missing = miss.fit_transform(train_no_dup)

LowVarianceHandler
~~~~~~~~~~~~~~~~~~
Drops low-information **features**.

- Eliminates feature columns with zero or near-zero variance.
- Helps reduce dimensionality and noise before modeling.
- Activity and ID columns are not altere

.. code-block:: python

   from proqsar.Preprocessor.Clean import LowVarianceHandler

   lowvar = LowVarianceHandler(activity_col='pChEMBL', id_col='id')
   train_var = lowvar.fit_transform(train_no_missing)

UnivariateOutliersHandler
~~~~~~~~~~~~~~~~~~~~~~~~~
Removes outliers based on univariate statistics.

- Applies z-score, IQR, or other cutoffs to individual features.
- Flags or removes samples with extreme values.
- ``id_col`` and ``activity_col`` are retained.

.. code-block:: python

   from proqsar.Preprocessor.Outlier.univariate_outliers import UnivariateOutliersHandler

   univ = UnivariateOutliersHandler(activity_col='pChEMBL', id_col='id')
   train_univ = univ.fit_transform(train_var)

KBinHandler
~~~~~~~~~~~
Applies binning to the **feature matrix** as an additional safeguard against outliers.

- Operates on feature columns (not the activity column).
- Groups continuous feature values into discrete bins.
- Especially useful for samples still marked as outliers after univariate filtering.
- ``id_col`` and ``activity_col`` are carried along unchanged.

.. code-block:: python

   from proqsar.Preprocessor.Outlier.kbin_handler import KBinHandler

   kbin = KBinHandler(activity_col='pChEMBL', id_col='id')
   train_binned = kbin.fit_transform(train_univ)

MultivariateOutliersHandler
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Detects outliers across multiple features jointly.

- Uses multivariate statistics (e.g., Mahalanobis distance, PCA).
- Removes samples that deviate strongly from the population.
- ``id_col`` and ``activity_col`` are carried through.

.. code-block:: python

   from proqsar.Preprocessor.Outlier.multivariate_outliers import MultivariateOutliersHandler

   multi = MultivariateOutliersHandler(activity_col='pChEMBL', id_col='id')
   train_multi = multi.fit_transform(train_binned)

Rescaler
~~~~~~~~
Rescales features values (e.g., normalization or standard scaling).

.. code-block:: python

   from proqsar.Preprocessor.Clean import Rescaler

   rescale = Rescaler(activity_col='pChEMBL', id_col='id')
   train_rescaled = rescale.fit_transform(train_multi)


Full Pipeline
-------------

You can chain all preprocessing steps into a single scikit-learn ``Pipeline``:

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from proqsar.Preprocessor.Clean import DuplicateHandler, MissingHandler, LowVarianceHandler, Rescaler
   from proqsar.Preprocessor.Outlier.kbin_handler import KBinHandler
   from proqsar.Preprocessor.Outlier.univariate_outliers import UnivariateOutliersHandler
   from proqsar.Preprocessor.Outlier.multivariate_outliers import MultivariateOutliersHandler

   pipeline = Pipeline([
       ("duplicate", DuplicateHandler(activity_col='pChEMBL', id_col='id')),
       ("missing", MissingHandler(activity_col='pChEMBL', id_col='id')),
       ("lowvar", LowVarianceHandler(activity_col='pChEMBL', id_col='id')),
       ("univ_outlier", UnivariateOutliersHandler(activity_col='pChEMBL', id_col='id')),
       ("kbin", KBinHandler(activity_col='pChEMBL', id_col='id')),
       ("multiv_outlier", MultivariateOutliersHandler(activity_col='pChEMBL', id_col='id')),
       ("rescaler", Rescaler(activity_col='pChEMBL', id_col='id')),
   ])

   pipeline.fit(train)
   train_clean = pipeline.transform(train)
   test_clean = pipeline.transform(test)

Summary
-------

- Each handler can be used **individually** for fine-grained control.  
- Combining them in a ``Pipeline`` ensures **reproducibility** and **consistent preprocessing** across train/test splits.  
- The pipeline is scikit-learn compatible, so you can append featurizers or models after the preprocessing steps.

See Also
--------

- :mod:`proqsar.Preprocessor.Clean` - duplicate/missing handling, low variance filtering, rescaling  
- :mod:`proqsar.Preprocessor.Outlier` - feature binning for residual outliers  
- :mod:`proqsar.Preprocessor.Outlier.univariate_outliers` - univariate statistical outlier detection  
- :mod:`proqsar.Preprocessor.Outlier.multivariate_outliers` - multivariate outlier detection  

