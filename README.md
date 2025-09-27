# ProQSAR

[![PyPI version](https://img.shields.io/pypi/v/proqsar.svg)](https://pypi.org/project/proqsar/)
[![conda](https://img.shields.io/conda/vn/tieulongphan/proqsar.svg?label=conda)](https://anaconda.org/tieulongphan/proqsar)
[![Docker Pulls](https://img.shields.io/docker/pulls/tieulongphan/proqsar.svg)](https://hub.docker.com/r/tieulongphan/proqsar)
[![Docker Image Version](https://img.shields.io/docker/v/tieulongphan/proqsar/latest?label=container)](https://hub.docker.com/r/tieulongphan/proqsar)
[![License](https://img.shields.io/github/license/Medicine-Artificial-Intelligence/proqsar.svg)](https://github.com/Medicine-Artificial-Intelligence/proqsar/blob/main/LICENSE)
[![Release](https://img.shields.io/github/v/release/Medicine-Artificial-Intelligence/proqsar.svg)](https://github.com/Medicine-Artificial-Intelligence/proqsar/releases)
[![Last Commit](https://img.shields.io/github/last-commit/Medicine-Artificial-Intelligence/proqsar.svg)](https://github.com/Medicine-Artificial-Intelligence/proqsar/commits)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.17209616.svg)](https://doi.org/10.5281/zenodo.17209616)
[![CI](https://github.com/Medicine-Artificial-Intelligence/proqsar/actions/workflows/test-and-lint.yml/badge.svg?branch=main)](https://github.com/Medicine-Artificial-Intelligence/proqsar/actions/workflows/test-and-lint.yml)
[![Stars](https://img.shields.io/github/stars/Medicine-Artificial-Intelligence/proqsar.svg?style=social&label=Star)](https://github.com/Medicine-Artificial-Intelligence/proqsar/stargazers)
---

**ProQSAR — automatic pipeline for quantitative structure–activity relationship (QSAR) modeling**

A reproducible toolkit for end-to-end QSAR: data standardization, featurization, splitting, model training, uncertainty estimation, and evaluation. Designed for reproducible experiments, continuous integration, and easy integration into ML/CADD pipelines. Full documentation for ProQSAR is available at [ReadTheDocs](https://proqsar.readthedocs.io/en/latest/).

![ProQSAR](https://raw.githubusercontent.com/Medicine-Artificial-Intelligence/ProQSAR/main/doc/fig/proqsar.png)



## Key features

- Data standardization and sanitization (SMILES normalization, valence checks, tautomer/charge handling).
- Modular featurizers: fingerprints, descriptors, learned featurizers (pluggable API).
- Flexible dataset splitting: random, scaffold, stratified.
- Built-in pipelines for training and evaluation with uncertainty estimation.
- Simple CLI and Python API for reproducible experiments and batch processing.
- CI-tested with unit tests and example notebooks.


## Installation

Choose the preferred installation method.

**From PyPI**

```bash
pip install proqsar
```

**From conda (anaconda.org/tieulongphan)**

```bash
conda install -c tieulongphan proqsar
```

**Docker (containerized)**

```bash
docker pull tieulongphan/proqsar:latest
# run an example container (bind-mount your project directory)
docker run --rm -v $(pwd):/workspace -w /workspace tieulongphan/proqsar:latest proqsar --help
```

**From source (developer)**

```bash
git clone https://github.com/Medicine-Artificial-Intelligence/proqsar.git
cd proqsar
pip install -e .[dev]
```

## Development & contributing

Thanks for your interest in contributing! A quick checklist:

1. Fork the repository and create a feature branch.
2. Implement your changes and include unit tests.
3. Run linting and tests locally (`pre-commit`, `flake8`, `pytest`).
4. Open a Pull Request describing the change and add tests/examples.

## Citation / Publication

If you use ProQSAR in research, please cite the project. Example BibTeX placeholder:

```bibtex
@misc{proqsar2025,
  title = {ProQSAR: Automatic pipeline for QSAR modeling},
  author = {Tuyet-Minh Phan and Tieu-Long Phan and Phuoc-Chung Nguyen Van and contributors},
  year = {2025},
  howpublished = {\url{https://github.com/Medicine-Artificial-Intelligence/proqsar}}
}
```

## Authors & Contributors

- [Tuyet-Minh Phan](https://github.com/tuyetminhphan)
- [Tieu-Long Phan](https://tieulongphan.github.io/)
- [Phuoc-Chung Nguyen Van](https://github.com/phuocchung123)
- [Thanh-An Pham](https://github.com/Thanh-An-Pham)


## License

This project is licensed under MIT License - see the [License](LICENSE) file for details.

## Acknowledgments

This work has received support from the Korea International Cooperation Agency (KOICA) under the project entitled “Education and Research Capacity Building Project at University of Medicine and Pharmacy at Ho Chi Minh City”, conducted from 2024 to 2025 (Project No. 2021-00020-3).