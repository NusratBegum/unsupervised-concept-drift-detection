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

    def __iter__(self):
        names = [f"x{j}" for j in range(D)]
        for x, y in zip(self.X, self.y):
            yield dict(zip(names, x)), int(y)


def synthetic_stream(name, seed):
    """name is 'Synth' + kind, e.g. 'SynthAbrupt'."""
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
    print("ok")
