# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# See LICENSE-EADD for full terms. Noncommercial use only.
"""Development check (18 Nov 2026, not a registered test): false-rejection rate at the
first test after an alarm when EADD waits for 31, 59 or 99 earlier windows.
Reps 0-499 (registered tests used 1000-2999). Output experiments/results_confirm/dev_warm.jsonl"""
import os
from experiment_confirm import OUT_DIR
from experiment_validity import p_job, run

if __name__ == "__main__":
    run([dict(source=s, config=c, rep=r) for s in ["real_poker", "real_elec"] for c in ["warm31", "warm59", "warm99"]
         for r in range(500)], p_job, os.path.join(OUT_DIR, "dev_warm.jsonl"))
