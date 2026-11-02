# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# See LICENSE-EADD for full terms. Noncommercial use only.
"""
Type I Error of the ASPT Stopping Rules
=======================================
Simulates the two stopping rules of EADD._aspt_test under the null
hypothesis. Under H0 the observed AUC and the permuted AUCs are
exchangeable, so they are drawn as i.i.d. uniforms (the first one is the
observed AUC). The rejection rate of ASPT is compared with that of a fixed
test with the same maximum number of permutations.

Outputs:
  experiments/results_aspt_type1/summary.csv
"""
import os

import numpy as np
import pandas as pd

B_MAX = 199
B_MIN = 10
ALPHA = 0.05
REPS = 400_000
CHUNK = 50_000
SEED = 0


def simulate_chunk(rng, n):
    u = rng.random((n, B_MAX + 1))
    exceed = np.cumsum(u[:, 1:] >= u[:, :1], axis=1)  # c_i for i = 1..B_MAX
    i = np.arange(1, B_MAX + 1)
    early_reject = (exceed == 0) & (i >= B_MIN) & (1.0 / (i + 1) < ALPHA)
    early_accept = (exceed / i > 2 * ALPHA) & (i >= B_MIN)
    stop = early_reject | early_accept
    stopped = stop.any(axis=1)
    first = np.where(stopped, stop.argmax(axis=1), B_MAX - 1)
    rows = np.arange(n)
    full_reject = (exceed[:, -1] + 1) / (B_MAX + 1) < ALPHA
    aspt_reject = np.where(stopped, early_reject[rows, first], full_reject)
    return aspt_reject.sum(), full_reject.sum(), (first + 1).sum()


def main():
    rng = np.random.default_rng(SEED)
    aspt, fixed, perms = 0, 0, 0
    for start in range(0, REPS, CHUNK):
        a, f, p = simulate_chunk(rng, min(CHUNK, REPS - start))
        aspt, fixed, perms = aspt + a, fixed + f, perms + p
    summary = pd.DataFrame([{
        "alpha": ALPHA, "b_min": B_MIN, "b_max": B_MAX, "reps": REPS,
        "aspt_type1": aspt / REPS, "fixed_type1": fixed / REPS,
        "aspt_mean_permutations": perms / REPS,
    }])
    out_dir = os.path.join("experiments", "results_aspt_type1")
    os.makedirs(out_dir, exist_ok=True)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
