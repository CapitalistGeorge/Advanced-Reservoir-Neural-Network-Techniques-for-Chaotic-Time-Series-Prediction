# 🧠 Reservoir Neural Networks with Fourier Layer
### Advanced Reservoir Computing Techniques for Chaotic Time Series Prediction
**By [ACS Lab, ITMO University](https://iai.itmo.ru/)** · 2025

Official repository for the paper:
> **A. Kovantsev, R. Vysotskiy (2025). _Advanced Reservoir Neural Network Techniques for Chaotic Time Series Prediction._ SSRN 5481760.**

---

## 📘 Overview

We present **ESN‑F** — an **Echo State Network (ESN)** enhanced with **Fourier features** and **polynomial expansion** for forecasting **chaotic / nonlinear** time series. The approach keeps the **reservoir untrained** and learns only a **ridge readout**, while enriching inputs with **periodic (sin/cos)** and **nonlinear** bases that improve long‑horizon stability.

Use cases include **finance/economics**, **risk modeling**, and other **non‑stationary** domains.

---

## 🗂 Project Layout

```
.
├── src/asc_itmo_lab/   # library code (reservoirs, features, utils)
├── notebooks/          # research notebooks & reproductions
├── tests/              # unit tests
├── data/               # datasets / artifacts (see notebooks)
├── requirements.txt    # dependencies (pinned)
└── LICENSE             # MIT
```

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/CapitalistGeorge/Advanced-Reservoir-Neural-Network-Techniques-for-Chaotic-Time-Series-Prediction.git
cd Advanced-Reservoir-Neural-Network-Techniques-for-Chaotic-Time-Series-Prediction

# (optional) create a clean environment
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

python -m pip install -U pip
pip install -r requirements.txt
```

### Run examples (notebooks-first)
```bash
jupyter lab  # or: jupyter notebook
# open notebooks/ and run the demos / reproductions
```

### Code quality (optional)
```bash
pytest -q        # unit tests (./tests)
ruff check .     # lint
ruff format .    # format
```

---

## 🧩 Motivation

Chaotic time series are inherently unpredictable due to their **sensitivity to initial conditions**, yet they exhibit **hidden regularities** that can be uncovered in the **frequency domain**.

Traditional neural networks tend to overfit such data or lose stability. To address this, we combine the **fast, low-training-cost dynamics of reservoir computing** with **Fourier-based feature expansion**.

> “The key idea is to move part of the temporal representation from the time domain to the frequency domain — allowing the model to selectively amplify meaningful periodicities while suppressing chaotic noise.”

---

## 🔬 Methodology (ESN‑F)

**Reservoir (leaky ESN)**
$ \mathbf{s}_t \;=\; (1-\alpha)\,\mathbf{s}_{t-1} \;+\; \alpha\,\tanh\!\big( W_{\text{in}}\,[1;\,\mathbf{x}_t] \;+\; W\,\mathbf{s}_{t-1} \big) \tag{1} $

- $\alpha \in (0,1]$ — leaking rate;
- $W_{\text{in}}\!\in\!\mathbb{R}^{N\times(d+1)}$, $W\!\in\!\mathbb{R}^{N\times N}$ — sparse reservoir weights (scaled by spectral radius).

**Polynomial expansion** (degree $p$)
\[
\boldsymbol{\phi}(\mathbf{x}_t) \;=\; \mathrm{Poly}_p(\mathbf{x}_t) \tag{2}
\[

**Fourier (harmonic) features** for $k=1,\dots,K$
$$
\boldsymbol{\psi}(\mathbf{x}_t) \;=\; \big[\,A_k,\ \sin(2\pi k\,\mathbf{x}_t),\ \cos(2\pi k\,\mathbf{x}_t)\,\big]_{k=1}^{K} \tag{3}
$$

**Feature concatenation**
$$
\mathbf{H}_t \;=\; \big[\,\mathbf{s}_t;\ \boldsymbol{\phi}(\mathbf{x}_t);\ \boldsymbol{\psi}(\mathbf{x}_t)\,\big] \tag{4}
$$

**Standardization & ridge readout**
$$
\hat{\mathbf{y}}_t \;=\; W_{\text{out}}\,\tilde{\mathbf{H}}_t,\qquad
W_{\text{out}}=\arg\min_W \|Y-WH\|_2^2+\lambda\|W\|_2^2 \tag{5}
$$

All equations (1)–(5) are exactly as in the paper’s **Design** section. Fourier & polynomial terms are **feature engineering** feeding the readout; the reservoir core remains **untrained**.

---

## 🧪 Datasets & Protocol (as in the paper)

- **M4 subset**: 1,500 real‑life series (length ≈1000), forecast horizon **15**, **15** rolling runs, metric **MAPE**.
- **Clustering by predictability** (features: **Hurst**, **Noise** factor, **Correlation dimension**, **max Lyapunov**, **KSE**, number of prevailing Fourier harmonics $N_{Fh}$) → **Good** (1176) vs **Bad** (324).
- **Real estate (Moscow)**: weekly series **2016‑12‑28 → 2024‑12‑25** (418 pts), smoothed by 13‑point moving average; expanding‑window CV (initial train **104**, horizon **52**).

---

## 📊 Experimental Results (from the paper)

### M4 (clustered by predictability, MAPE % ↓)
| Cluster | ESN‑F | ESN | LGBM | Prophet | SSA |
|---|---:|---:|---:|---:|---:|
| Good | **3.44** | 3.56 | 3.72 | 6.86 | 18.03 |
| Bad  | 5.26 | **5.19** | 5.39 | 8.57 | 20.05 |

In the **Bad** cluster, ESN‑F beats LGBM by ≥1 pp in **27%** of series (LGBM better in **15%**; remainder negligible).

### Moscow Real Estate (weekly, MAPE % ↓)
| Model | MAPE |
|---|---:|
| **ESN‑F** | **2.56** |
| ESN | 3.19 |
| LGBM | 7.18 |

Chaotic traits of the real‑estate series (for interpretation): Hurst **0.65**, Noise **0.99**, Corr. dimension **1.33**, max Lyapunov **0.01**, **KSE** **1.84**, $N_{Fh}=30$.

---

## 🔧 Reproducing the paper

- Run notebooks in `notebooks/` to reproduce **M4** experiments (15×15 protocol) and the **real‑estate** case study.
- See comments inside notebooks for dataset preparation, feature extraction, and CV splits.

---

## 📐 Predictability features (formulas)

**Hurst exponent**
$$
H \;=\; \frac{\ln\!\big(R(\tau)/S(\tau)\big)}{\ln(\alpha \tau)} \tag{6}
$$
with
$$
R(\tau) = \max_{1\le t\le \tau} \sum_{i=1}^{t} (x_i - \bar{x}_\tau) \;-\; \min_{1\le t\le \tau} \sum_{i=1}^{t} (x_i - \bar{x}_\tau), \quad
S(\tau)=\sqrt{\frac{1}{\tau}\sum_{t=1}^{\tau}(x_t-\bar{x}_\tau)^2}. \tag{7,8}
$$


**KSE (Kolmogorov–Sinai entropy)** — definition via entropy rate upper bound:
$$
h_\mu(T,\xi) = - \lim_{n\to\infty}\frac{1}{n} \sum_{i_1,\dots,i_n} \mu(T^{-1}C_{i_1}\cap\cdots\cap T^{-n}C_{i_n}) \ln \mu(\cdots), \quad
h^{KS}_\mu(T)=\sup_{\xi} h_\mu(T,\xi). \tag{9,10}
$$


**Correlation dimension**
$$
d_k = \lim_{r\to 0}\lim_{m\to\infty}\frac{\ln C(r)}{\ln r},\quad
C(r)=\frac{1}{m(m-1)}\sum_{i=1}^{m}\sum_{j=i+1}^{m}\theta(r-\rho(i,j)). \tag{11,12,13}
$$


**Prevailing Fourier harmonics**
$$
N_{Fh}=\sum_{i=1}^{k}\theta(A_i-\bar{A}). \tag{14}
$$


**Noise factor**
$$
F_N = 1 - \sqrt{ \frac{N}{N-1} \cdot \frac{\sum_{i=1}^{N-1}(x'_{i}-\bar{x}')^2}{\sum_{i=1}^{N}(x_{i}-\bar{x})^2} } \tag{15}
$$


---

## 🧰 Dependencies

Install from `requirements.txt`:
```bash
pip install -r requirements.txt
```
Typical stack: Python, NumPy/SciPy, scikit‑learn, Jupyter, plotting libs, and QA tooling (ruff, pytest).

---

## 🧑‍🔬 Authors and Credits

Developed by **[ACS Lab, ITMO University](https://iai.itmo.ru/)**
Maintainer: [@CapitalistGeorge](https://github.com/CapitalistGeorge)
Contributors: A. Kovantsev, R. Vysotskiy, and ACS Lab research team.

---

## 📚 Citation

```bibtex
@article{KovantsevVysotskiy2025Reservoir,
  title   = {Advanced Reservoir Neural Network Techniques for Chaotic Time Series Prediction},
  author  = {Kovantsev, Anton and Vysotskiy, Roman},
  year    = {2025},
  journal = {SSRN Electronic Journal},
  doi     = {10.2139/ssrn.5481760}
}
```

---

## 🪪 License

Released under the **MIT License** — free to use, modify, and distribute with attribution.

---

> _“Spectral insight meets chaotic dynamics — bridging periodicity and unpredictability.”_
