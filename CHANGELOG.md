# ProQSAR — Release v0.0.6
**Release date:** 2025-09-24

**Short:** This release improves the `Inference` wrapper (parity with `pipeline.predict`), tightens documentation, and updates packaging for PyPI / Conda / Docker.

**Full documentation:** http://proqsar.readthedocs.io/en/latest/

---

## Highlights — v0.0.6
- `Inference` now returns the same DataFrame format as `pipeline.predict(...)` (point predictions, conformal prediction intervals, and AD flags) and stores last-run metadata with a compact printed summary.
- Improved/reformatted documentation for the ProQSAR pipeline and inference workflow.
- Packaging & distribution updates: PyPI, Conda channel and Docker images published for v0.0.6.
- Various minor bugfixes and unit-test improvements.

---

## Install
Binary/distribution artifacts are available on the usual channels (PyPI, Conda, DockerHub). See the project release page or documentation for exact install commands and tags.

---

## ProQSAR pipeline & Inference pipeline
The release includes documentation and runnable artifacts for:
- ProQSAR pipeline (training + optimisation)
- Inference wrapper (prediction, conformal PIs, applicability-domain flags)

See the documentation for usage examples and API details.

---

## Reproducibility checklist
- Fix `random_state` in `ProQSAR` and any downstream components that accept seeds.
- Pin environment versions (Python, RDKit, scikit-learn, xgboost, optuna).
- Save artifacts from runs (`pipeline.save_dir`) — includes model, CV results, Optuna study & plots.
- When comparing runs, keep `alpha`, CV settings and optimizer budget identical.

---

## Troubleshooting pointers
- If predictions are constant or many samples are marked `out` in AD: inspect preprocessing (duplicates, missing, low-variance), feature generation, and AD thresholds.
- If conformal PIs look wrong: verify calibration split/residuals and that there is no data leakage.
- If you encounter a `KeyError` for `smiles_key` or `id_key`: confirm input DataFrame column names and pass the correct keys to `Inference.run`.

---

## Changelog (summary)
**v0.0.6**
- Feature: `Inference.run(...)` mirrors `pipeline.predict(...)` output exactly and stores last-run metadata.
- Docs: improved examples and RST/Markdown snippets for pipeline and inference usage.
- Packaging: released PyPI, Conda and Docker artifacts for v0.0.6.
- Tests/bugfixes: assorted stability fixes and test improvements.

For a granular commit-level changelog, consult the repository's CHANGELOG or release notes.

---

## How to get help / report issues
- Documentation: http://proqsar.readthedocs.io/en/latest/
- Issues & PRs: GitHub repository (open an issue for bugs / feature requests).
- When reporting issues, include a minimal reproducible example, environment `requirements.txt`, and any relevant error traces.

---

Thank you for using ProQSAR — if you'd like, I can also produce:
- a GitHub release body ready to paste, or
- a short announcement post for social platforms, or
- a conventional `CHANGELOG.md` entry in Conventional Commits format.
