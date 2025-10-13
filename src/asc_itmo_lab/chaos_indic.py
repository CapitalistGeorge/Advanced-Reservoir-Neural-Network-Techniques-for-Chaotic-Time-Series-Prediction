import numpy as np
from itertools import combinations
from scipy.linalg import hankel
import pandas as pd
from scipy.spatial import cKDTree
from numpy.fft import fft

"""Наименьшие квадраты для одной переменной"""


def MLS(x, y):
    x = np.array(x).astype(float)
    y = np.array(y).astype(float)
    n = len(x)
    sumx, sumy = sum(x), sum(y)
    sumx2 = sum([t * t for t in x])
    sumxy = sum([t * u for t, u in zip(x, y)])
    a = (n * sumxy - (sumx * sumy)) / (n * sumx2 - sumx * sumx)
    b = (sumy - a * sumx) / n
    return a, b


"""Показатель Хёрста (R/S и H траектории)"""


def HurstTraj(ser):  # RS-trajectory of Hurst
    h = []
    z2, _, _ = Norm01(ser)
    tau = np.arange(3, len(z2))
    for t in tau:
        m, s = np.mean(z2[:t]), np.std(z2[:t])
        x = (z2[:t] - m).cumsum()
        r = max(x) - min(x)
        h.append(np.log(r / s) if r * s > 0.0 else 0.0)
    h = np.array(h)
    t = np.array([0.0])
    t = np.concatenate([t, np.log(tau[1:] / 2)])
    l = int(len(t) / 50)
    he, b = MLS(t[:l], h[:l])
    mem = np.where([(h[i + 1] - h[i]) < 0.0 for i in range(len(h) - 1)])[0]
    mem = mem[0] if len(mem) else 0
    return (
        t,
        h,
        he,
        mem,
    )  # t-ln(tau); h - R/S trajectory (Hurst's tr=h/t); he - Hurst's exponent; mem - series' memory


"""Линейное укладывание в диапазон [0,1], возвращает коэффициенты для восстановления (max(X))!=0"""


def Norm01(x):
    mi = np.nanmin(x)
    ma = np.nanmax(np.array(x) - mi)
    if ma > 0.0:
        x_n = (np.array(x) - mi) / ma
        return x_n, mi, ma
    else:
        return np.zeros(len(x)), mi, ma


"""мера шумности по ско разностей к ско ряда. (больше мера - меньше шума)"""


def NoiseFactor(data, axis=0, ddof=1):
    a = Norm01(data)[0]
    m = np.std(pd.Series(a).diff().dropna().abs())
    sd = a.std(axis=axis, ddof=ddof)
    return 1 - float(np.where(sd == 0, 0, m / sd))


"""Размерность вложения по корреляционному интегралу"""


def DimEmb(tser, eps=0.1):
    ser, _, _ = Norm01(tser)
    n = len(ser)
    cn = [1]
    d0 = 0
    h = hankel(ser)
    for k in range(2, n // 2):  #
        ent = sum(cn)
        w = h[: n - k, :k]
        ro = np.zeros([n - k, n - k])
        for i, j in combinations(np.arange(n - k), 2):
            norm = np.linalg.norm(w[i] - w[j])
            ro[i, j] = norm
            ro[j, i] = norm
        cl = []
        cn = []
        ls = np.linspace(ro[ro != 0].min(), ro.max(), num=20)
        for l in ls:
            c = np.heaviside(l - ro - np.diag(np.ones(n - k)), 1).sum() // 2
            cn.append(c / (n - k) ** 2)
            cl.append(np.log(c / (n - k) ** 2))
        dc = (cl[1] - cl[0]) / (np.log(ls[1]) - np.log(ls[0]))
        if abs(dc - d0) > eps:  # (ro.max() - ro.min())/50.:
            d0 = dc
        else:
            break
    k -= 1
    dc = d0
    ent = sum(cn) / ent
    return (
        k,
        dc,
        ent,
    )  # k - размерность вложения, dc - корреляционная размерность, ent - оценка энтропии.


def max_lyapunov(x, emb_dim=10, lag=1, fit_len=20):
    """
    Estimate the maximal Lyapunov exponent from a single time series using
    Rosenstein’s algorithm.

    Parameters
    ----------
    x : array-like
        1-dimensional time series.
    emb_dim : int
        Embedding dimension for phase-space reconstruction.
    lag : int
        Time delay used for embedding.
    fit_len : int
        Number of steps to track divergence and fit the slope.

    Returns
    -------
    float
        Maximal Lyapunov exponent estimate.
    """
    x = np.asarray(x, dtype=float)

    # Step 1: Phase-space reconstruction
    N = len(x) - (emb_dim - 1) * lag
    if N <= fit_len:
        raise ValueError("Time series too short for given parameters.")

    Y = np.column_stack([x[i * lag : i * lag + N] for i in range(emb_dim)])

    # Step 2: Find nearest neighbors (excluding trivial self-match)
    tree = cKDTree(Y)
    _, idxs = tree.query(Y, k=2)
    neighbors = idxs[:, 1]

    # Step 3: Measure divergence between neighbors over time
    divergences = np.zeros(fit_len)
    counts = np.zeros(fit_len)

    for i in range(N):
        j = neighbors[i]
        # Ensure we stay within bounds for both trajectories
        for k in range(fit_len):
            if i + k < N and j + k < N:
                dist = np.abs(x[i + k] - x[j + k])
                # Avoid log(0) by adding a small epsilon
                divergences[k] += np.log(dist + 1e-12)
                counts[k] += 1

    # Step 4: Average divergence at each step
    valid = counts > 0
    avg_divergence = divergences[valid] / counts[valid]

    # Fit a linear slope to the average divergence curve
    times = np.arange(fit_len)[valid]
    slope, _ = np.polyfit(times, avg_divergence, 1)

    # The slope is the maximal Lyapunov exponent
    return slope


def ks_entropy_partition(x, n_bins=10):
    """
    Partition‐based KS entropy estimator.

    1) Bin the 1D series x into n_bins equiprobable bins (quantiles).
    2) Form the symbol sequence s_t = bin_index(x_t).
    3) Estimate transition matrix P_{ij} from s_t -> s_{t+1}.
    4) Compute entropy rate h = - sum_i pi_i sum_j P_{ij} log P_{ij}.

    Parameters
    ----------
    x : array-like, shape (N,)
        The (finite, numerical) time series.
    n_bins : int
        Number of quantile‐based bins.

    Returns
    -------
    float
        Estimated KS entropy (in nats per sample).
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    # 1) Assign quantile bins
    # Use pandas qcut for equal‐size bins
    cats, bins = pd.qcut(x, q=n_bins, labels=False, retbins=True, duplicates="drop")
    s = np.array(cats, dtype=int)
    B = len(bins) - 1

    # 2) Build transition counts
    counts = np.zeros((B, B), dtype=int)
    for t in range(N - 1):
        i, j = s[t], s[t + 1]
        counts[i, j] += 1

    # 3) Row‐normalize to get P_{ij}
    row_sums = counts.sum(axis=1, keepdims=True)
    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        P = counts / row_sums
    P[np.isnan(P)] = 0.0

    # 4) Stationary frequencies pi_i
    pi = row_sums.flatten() / row_sums.sum()

    # 5) Compute entropy rate
    # Only sum where P>0
    Pnz = P > 0
    h = -np.sum(pi[:, None] * P[Pnz] * np.log(P[Pnz]))
    return float(h)


def fourier_harmonic_count(x):
    """
    Count of Fourier harmonics with amplitude above the mean.
    """
    # Xf = fft(x)
    # A = 2 * np.abs(Xf) / len(x)

    freq = np.fft.fftfreq(len(x), d=0.1)
    mag = 2 * np.abs(np.fft.fft(x)) / len(x)
    idx = np.argmax(freq < 0)

    mag = mag[1:idx]
    meanMag = mag.mean()
    return np.sum(mag > meanMag)
