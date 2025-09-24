.. _getting-started-proqsar:

.. |pypi-badge| image:: https://img.shields.io/pypi/v/proqsar.svg
   :alt: PyPI version
   :target: https://pypi.org/project/proqsar/

.. |conda-badge| image:: https://img.shields.io/conda/vn/tieulongphan/proqsar.svg?label=conda
   :alt: conda
   :target: https://anaconda.org/tieulongphan/proqsar

.. |docker-badge| image:: https://img.shields.io/docker/pulls/tieulongphan/proqsar.svg
   :alt: Docker pulls
   :target: https://hub.docker.com/r/tieulongphan/proqsar

|pypi-badge| |conda-badge| |docker-badge|


Getting Started
===============

Welcome to **ProQSAR** — an opinionated, reproducible pipeline for QSAR modelling and small-molecule featurisation.

This quickstart shows how to install ProQSAR, verify the installation, and run a minimal pipeline. For full reference documentation, see the project docs and the API reference in the repository.

Introduction
------------
**ProQSAR** is a lightweight toolkit for end-to-end QSAR workflows: data standardization, featurization, dataset splitting, model training, uncertainty estimation, and evaluation. It provides a simple CLI and a modular Python API so you can run quick experiments or embed ProQSAR components into larger CADD pipelines.

Requirements
------------
- **Python** >= 3.11
- Recommended: an isolated virtual environment (venv/virtualenv or Conda)

Virtual environment (recommended)
---------------------------------
Using a virtual environment prevents dependency conflicts.

1. Using ``venv`` (cross-platform)

.. code-block:: bash

   python3 -m venv proqsar-env
   source proqsar-env/bin/activate    # Linux / macOS
   proqsar-env\Scripts\activate     # Windows (PowerShell)

2. Using Conda

.. code-block:: bash

   conda create -n proqsar-env python=3.11 -y
   conda activate proqsar-env


Installation
------------
Install the package from PyPI, Conda (channel: ``tieulongphan``), or using the official Docker image.

**From PyPI**::

   pip install proqsar

**From conda**::

   conda install -c tieulongphan proqsar

**Docker**::

   docker pull tieulongphan/proqsar:latest
   docker run --rm tieulongphan/proqsar:latest proqsar --help

Quick verification
------------------
Verify the installed package and check the version:

.. code-block:: bash

   python -c "import importlib.metadata as m; print(m.version('proqsar'))"

Development & contributing
--------------------------
1. Fork the repository and work on a feature branch.
2. Add unit tests and run the test-suite locally.
3. Follow code style (PEP8, type hints) and run pre-commit hooks.
4. Open a PR with a clear description and tests.

Support
-------
Report bugs or request features on GitHub:

`ProQSAR Issues <https://github.com/Medicine-Artificial-Intelligence/proqsar/issues>`_

Further reading
---------------
- Project repository: `ProQSAR on GitHub <https://github.com/Medicine-Artificial-Intelligence/proqsar>`_
- Full documentation: `ProQSAR Docs <http://proqsar.readthedocs.io/en/latest/>`_

Enjoy using **ProQSAR**! 

