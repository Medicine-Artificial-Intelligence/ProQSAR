.. _data_module:

Data Module
===========

The ``proqsar.Data`` module provides utilities for **standardization**, **featurization**, and **splitting strategies** that form the foundation of QSAR workflows.

This section demonstrates the typical usage with a small dictionary-based dataset.

Example dataset
---------------

We start with a toy dataset containing SMILES strings and pChEMBL values:

.. code-block:: python

   from proqsar.Config.debug import force_quiet

   # silence logging
   force_quiet()

   data = [
       {'Smiles': 'O=C(N[C@H]1CCc2ccccc21)c1nc(-c2cccs2)nc(O)c1O', 'pChEMBL': 7.69897, 'id': 0},
       {'Smiles': 'CN1Cc2c(c(O)c3ncc(Cc4ccc(F)c(Cl)c4)cc3c2N(C)S(C)(=O)=O)C1=O', 'pChEMBL': 6.57675, 'id': 1},
       {'Smiles': 'Cc1ccccc1Cc1ccc2[nH]cc(C(=O)O)c(=O)c2c1', 'pChEMBL': 5.97, 'id': 2},
       {'Smiles': 'O=C(c1ccc(-c2ccccc2)cc1)c1cc(=O)n(O)c(=O)[nH]1', 'pChEMBL': 5.60206, 'id': 3},
       {'Smiles': 'O=P(O)(O)C(O)(Cc1ccc(Cl)c(Cl)c1)P(=O)(O)O', 'pChEMBL': 5.39794, 'id': 4},
       {'Smiles': 'O=C(O)c1cc(Br)cc(C(=O)/C=C/c2cc(Cl)cc(Cl)c2Cl)c1O', 'pChEMBL': 4.88606, 'id': 5},
       {'Smiles': 'N/C(=C\\C(=O)c1cn(Cc2ccc(F)cc2)cc1-c1ccccc1)C(=O)O', 'pChEMBL': 7.36653, 'id': 6},
       {'Smiles': 'CCO[C@@H]1C[C@@H](c2nc(C(=O)NCc3ccc(F)cc3)c(O)c(=O)n2C)N(C)C1', 'pChEMBL': 6.69897, 'id': 7},
       {'Smiles': 'Oc1ccc(C(c2c[nH]c3ccccc23)c2c[nH]c3ccccc23)cc1O', 'pChEMBL': 5.0, 'id': 8},
       {'Smiles': 'CCCCCC12CCC(C(=O)Nc3cccc(C(=O)CC(=O)C(=O)O)c3)(CC1)CC2', 'pChEMBL': 4.85387, 'id': 9}
   ]


Standardization
---------------

The ``SMILESStandardizer`` prepares molecules for consistent downstream processing (tautomer handling, aromaticity normalization, charge correction).

.. code-block:: python

   from proqsar.Data.Standardizer.smiles_standardizer import SMILESStandardizer

   std = SMILESStandardizer(smiles_col='Smiles')
   std_df = std.standardize_dict_smiles(data)
   print(std_df[0])

Output:

.. code-block:: text

   {
     'Smiles': 'O=C(N[C@H]1CCc2ccccc21)c1nc(-c2cccs2)nc(O)c1O',
     'pChEMBL': 7.69897,
     'id': 0,
     'standardized_Smiles': 'O=C(N[C@H]1CCc2ccccc21)c1[nH]c(-c2cccs2)nc(=O)c1O',
     'standardized_mol': <rdkit.Chem.rdchem.Mol at 0x...>
   }

Featurization
-------------

The ``FeatureGenerator`` computes molecular representations such as fingerprints and descriptors.

.. code-block:: python

   from proqsar.Data.Featurizer.feature_generator import FeatureGenerator

   feat = FeatureGenerator(
       mol_col='standardized_mol',
       activity_col='pChEMBL',
       id_col='id',
       smiles_col='standardized_Smiles',
       feature_types=['ECFP2', 'RDK5'],
       n_jobs=2,
       verbose=2
   )

   features = feat.generate_features(std_df)
   print(features.keys())

Output:

.. code-block:: text

   dict_keys(['ECFP2', 'RDK5'])

Data Splitting
--------------

The ``Splitter`` supports multiple strategies: random, stratified, scaffold, and time-based splits.

.. code-block:: python

   from proqsar.Data.Splitter.data_splitter import Splitter

   split = Splitter(
       activity_col='pChEMBL',
       smiles_col='standardized_Smiles',
       mol_col='standardized_mol',
       option='scaffold',
       test_size=0.2,
       random_state=42
   )

   train, test = split.fit(features['ECFP2']) # use ecfp2 as feature
   print(test)

Example test output:

.. code-block:: text

   Smiles                                            pChEMBL   id
   0  CN1Cc2c(c(O)c3ncc(Cc4ccc(F)c(Cl)c4)cc3c2N(C).. 6.57675    1
   1  O=C(N[C@H]1CCc2ccccc21)c1nc(-c2cccs2)nc(O)c1O  7.69897    0


Summary
-------
- **Standardizer** ensures consistent molecular input.
- **Featurizer** generates multiple types of fingerprints.
- **Splitter** enables flexible dataset partitioning.


See Also
--------

- :mod:`proqsar.Data.Standardizer`  
- :mod:`proqsar.Data.Featurizer`  
- :mod:`proqsar.Data.Splitter`  


