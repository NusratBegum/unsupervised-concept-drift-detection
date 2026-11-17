# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# See LICENSE-EADD for full terms. Noncommercial use only.
"""
Synthetic feature-drift streams with known drift points.

Same kinds of drift as in the proposal and the JCSSE paper (abrupt, gradual,
incremental, recurring, plus a stream without drift), but harder: only 3 of
10 Gaussian features drift, by one standard deviation, so detectors do not
all find every drift at once. Each seed draws a different stream, so seeds
are independent samples (unlike the fixed real-world streams).

20,000 samples, drift points at 5,000, 10,000 and 15,000. For gradual and
incremental drift the drift point is where the change starts; the change
lasts 1,000 samples. The recurring stream returns to the old concept at
each drift point.
"""
import functools

import numpy as np

KINDS = ["Stable", "Abrupt", "Gradual", "Incremental", "Recurring"]
N, D, K, SHIFT, WIDTH = 20_000, 10, 3, 1.0, 1_000
POINTS = [5_000, 10_000, 15_000]


class SyntheticStream:
    def __init__(self, kind, seed):
        assert kind in KINDS, kind
        # seed and kind together, so different kinds do not share their noise
        rng = np.random.default_rng([seed, KINDS.index(kind)])
        mean = np.zeros(N)  # mean of the drifting features, in units of SHIFT
        if kind != "Stable":
            for k, p in enumerate(POINTS, 1):
                level = (k % 2) if kind == "Recurring" else k
                prev = mean[p - 1]
                if kind in ("Abrupt", "Recurring"):
                    mean[p:] = level
                elif kind == "Incremental":
                    mean[p:p + WIDTH] = prev + (level - prev) * np.arange(1, WIDTH + 1) / WIDTH
                    mean[p + WIDTH:] = level
                else:  # gradual: samples come from the new concept with rising probability
                    new = rng.random(WIDTH) < np.arange(1, WIDTH + 1) / WIDTH
                    mean[p:p + WIDTH] = np.where(new, level, prev)
                    mean[p + WIDTH:] = level
        self.X = rng.normal(size=(N, D))
        self.X[:, :K] += SHIFT * mean[:, None]
        self.y = (mean > 0).astype(int)
        self.drifts = [] if kind == "Stable" else list(POINTS)
        self.shifted = list(range(K))

    def __iter__(self):
        names = [f"x{j}" for j in range(D)]
        for x, y in zip(self.X, self.y):
            yield dict(zip(names, x)), int(y)


class ArrayStream:
    def __init__(self, X, drifts=()):
        self.X, self.drifts = X, list(drifts)

    def __iter__(self):
        names = [f"x{j}" for j in range(self.X.shape[1])]
        for x in self.X:
            yield dict(zip(names, x)), 0


def ar_stream(rho, seed, n=N, d=D):
    """Stationary AR(1) per feature, unit variance: no drift, but autocorrelated."""
    rng = np.random.default_rng([seed, 100 + int(round(rho * 1000))])
    e = rng.normal(size=(n, d)) * np.sqrt(1 - rho ** 2)
    X = np.empty((n, d))
    X[0] = rng.normal(size=d)
    for t in range(1, n):
        X[t] = rho * X[t - 1] + e[t]
    return ArrayStream(X)


@functools.lru_cache(maxsize=4)
def _real_rows(dataset):
    # parsing a CSV row by row takes seconds; each worker process does it once per dataset
    import datasets as ds
    return np.array([list(x.values()) for i, (x, _) in zip(range(100_000), getattr(ds, dataset)())], dtype=float)


def shuffled_stream(dataset, seed, n=N):
    """Rows of a real stream in random order: real marginals, no drift, no autocorrelation."""
    X = _real_rows(dataset)
    rng = np.random.default_rng([seed, 7])
    return ArrayStream(X[rng.permutation(len(X))[:n]])


INJECT_BASES = ["Electricity", "NOAAWeather", "PokerHand", "Powersupply", "RialtoBridgeTimelapse",
                "InsectsIncrementalBalanced"]


def injected_stream(kind, dataset, seed):
    """A shuffled real stream (no drift of its own) with known drifts injected at
    5,000/10,000/15,000: 30% of the features (at least one) move by one standard
    deviation, at once (Abrupt) or linearly over 1,000 samples (Incremental)."""
    assert kind in ("Abrupt", "Incremental"), kind
    X = shuffled_stream(dataset, seed).X.copy()
    rng = np.random.default_rng([seed, 13])
    k = max(1, round(0.3 * X.shape[1]))
    feats = rng.choice(X.shape[1], k, replace=False)
    level = np.zeros(len(X))
    for j, p in enumerate(POINTS, 1):
        if kind == "Abrupt":
            level[p:] = j
        else:
            level[p:p + WIDTH] = (j - 1) + np.arange(1, WIDTH + 1) / WIDTH
            level[p + WIDTH:] = j
    sd = X[:POINTS[0], feats].std(axis=0)
    X[:, feats] += level[:, None] * np.where(sd > 0, sd, 1.0)
    out = ArrayStream(X, POINTS)
    out.shifted = sorted(int(f) for f in feats)  # indices of the features that drift
    return out


def synthetic_stream(name, seed):
    """'Synth' + kind (e.g. 'SynthAbrupt'), 'AR0.9', 'Shuffled' + dataset class name,
    or 'Inject' + Abrupt/Incremental + '_' + dataset class name."""
    if name.startswith("Inject"):
        kind, dataset = name[len("Inject"):].split("_")
        return injected_stream(kind, dataset, seed)
    if name.startswith("AR"):
        return ar_stream(float(name[2:]), seed)
    if name.startswith("Shuffled"):
        return shuffled_stream(name[len("Shuffled"):], seed)
    return SyntheticStream(name[len("Synth"):], seed)


if __name__ == "__main__":
    for kind in KINDS:
        s = SyntheticStream(kind, 0)
        m = s.X[:, :K].mean(axis=1)
        print(f"{kind:<12} drifts={s.drifts} mean of drifting features per 5k block:",
              [round(float(m[i:i + 5000].mean()), 2) for i in range(0, N, 5000)])
    a, b = SyntheticStream("Abrupt", 1), SyntheticStream("Abrupt", 2)
    assert not np.allclose(a.X, b.X) and np.allclose(SyntheticStream("Abrupt", 1).X, a.X)
    assert not np.allclose(SyntheticStream("Stable", 1).X[:, K:], a.X[:, K:])  # kinds do not share noise
    assert SyntheticStream("Stable", 0).drifts == [] and len(list(SyntheticStream("Gradual", 0))) == N
    x = ar_stream(0.9, 0).X
    assert abs(np.corrcoef(x[1:, 0], x[:-1, 0])[0, 1] - 0.9) < 0.02 and abs(x.std() - 1) < 0.05
    print("ok")
