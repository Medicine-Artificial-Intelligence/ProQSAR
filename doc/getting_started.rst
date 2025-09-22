.. _getting-started-template:

.. image:: https://img.shields.io/pypi/v/proqsar.svg
   :alt: PyPI version
   :align: right

===============================
Getting Started with Template
===============================

Welcome to the **template** documentation!

This repository provides a standardized template for building, testing, documenting, and releasing Python-based methods. It includes recommended layouts, tooling integrations, and CI/CD workflows to streamline development.

🛠 Development Workflow
-----------------------

1. **Source Layout**  
   Store all library code under ``src/`` (rename folder as desired).

2. **Unit Tests**  
   Keep tests in ``tests/`` and run via pytest::

      pytest --maxfail=1 --disable-warnings -q

3. **Linting & Formatting**  
   - Follow PEP 8 style and PEP 257 docstring conventions.  
   - Use ``black`` for automatic formatting and ``flake8`` to enforce quality.  
   - Run both via::

        ./lint.sh

   - Edit ``lint.sh`` to adjust rules or exclude files.

4. **Documentation**  
   - Write docs in ``docs/`` using Sphinx.  
   - Build locally with::

        ./build_doc.sh

   - Automate publishing via ReadTheDocs with ``.readthedocs.yml``.

5. **Dependency Management**  
   - Use ``env.yml`` for Conda (pins ``rdkit>=2025.3.1``, ``pytest``, ``black``, ``flake8``).  
   - Alternatively, manage with ``requirements.txt`` for pip users.

6. **Release & Packaging**  
   - Define package metadata in ``pyproject.toml``.  
   - Local install::

        pip install .

   - CI/CD workflows:  
     - **PyPI**: ``.github/workflows/publish-package.yml``  
     - **Docker**: ``.github/workflows/docker-publish.yml``

7. **Automated Dependency Updates**  
   - ``.github/dependabot.yml`` configured for weekly checks and PRs.

*Feel free to adapt any naming conventions or tooling versions to fit your project needs.*  
