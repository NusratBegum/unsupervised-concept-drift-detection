# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# See LICENSE-EADD for full terms. Noncommercial use only.
"""Development check (18 Nov 2026, not a registered test): false-rejection rate with
observed and null AUCs on references of the same size (null_match_size), steady state
and first test after an alarm. Reps 0-499. Output experiments/results_confirm/dev_match.jsonl"""
import os
import sys
from experiment_confirm import OUT_DIR
from experiment_validity import p_job, run

if __name__ == "__main__":
    sources = sys.argv[1:] or ["real_poker", "real_elec", "iid", "ar0.9"]
    run([dict(source=s, config=c, rep=r) for s in sources for c in ["warm31m", "history10m"] for r in range(500)],
        p_job, os.path.join(OUT_DIR, "dev_match.jsonl"))
