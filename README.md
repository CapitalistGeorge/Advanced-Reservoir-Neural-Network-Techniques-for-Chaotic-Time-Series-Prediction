# Reservoir Neural Networks @ ACS Lab ITMO

> Experiments and code around **reservoir computing (ESN/RC)** for forecasting **chaotic** and non-stationary time series. The repo contains reusable library code, research notebooks, and utilities.

![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Data](#data)
- [Tests & Quality](#tests--quality)
- [Reproducibility](#reproducibility)
- [Contributing](#contributing)
- [License](#license)

## Overview
This project explores practical improvements to reservoir neural networks for time-series prediction (feature expansions, spectral tricks, predictability measures like Hurst and Lyapunov, etc.). The code is organized so you can reuse library pieces from notebooks and write tests against them.

## Project Structure

```
.
├── src/asc_itmo_lab/        # library code (reservoirs, features, utils)
├── notebooks/               # research notebooks and examples
├── tests/                   # pytest unit tests
├── data/                    # datasets and artifacts (see "Data")
├── .github/workflows/       # CI pipelines (optional)
├── .pre-commit-config.yaml  # quality hooks (ruff, etc.)
├── .ruff.toml               # linter/formatter config
├── requirements.txt         # project dependencies
└── LICENSE                  # MIT
```


## Installation
```bash
# 1) Create an environment
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

python -m pip install -U pip

# 2) Install dependencies
pip install -r requirements.txt

# 3) (optional) Quality hooks
pip install pre-commit
pre-commit install
```
## Notebooks
Exploratory and reproduction notebooks live under ```notebooks/```. Open them from the project root so imports and data paths resolve properly.

## Tests & Quality
```
pytest -q            # unit tests (./tests)
ruff check .         # lint
ruff format .        # format
pre-commit run -a    # run all hooks locally
```
## Citing
If this repository helps your research, please cite the related work:

Anton Kovantsev, Roman Vysotskiy. Advanced Reservoir Neural Network Techniques for Chaotic Time Series Prediction, SSRN, 2025. DOI: 10.2139/ssrn.5481760.

```
@article{KovantsevVysotskiy2025Reservoir,
  title   = {Advanced Reservoir Neural Network Techniques for Chaotic Time Series Prediction},
  author  = {Kovantsev, Anton and Vysotskiy, Roman},
  year    = {2025},
  journal = {SSRN Electronic Journal},
  doi     = {10.2139/ssrn.5481760},
  url     = {https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5481760}
}
```

## License
Released under the MIT License. See ```./LICENSE``` for details.
