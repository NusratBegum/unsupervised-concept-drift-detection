"""Rerun thesis EADD runs with the current code and compare with the stored detections."""
import json
from concurrent.futures import ProcessPoolExecutor
from experiment_hypotheses import ANNOTATED, OUT_FILE, job_key, run_job

stored = {}
for line in open(OUT_FILE):
    r = json.loads(line)
    stored[r["key"]] = r["detections"]
jobs = [dict(part="annotated", dataset=d, detector="EADD", seed=s, tau=t, test="aspt")
        for d in ANNOTATED for s in (42, 242) for t in (0.5, 0.7)]
jobs = [j for j in jobs if job_key(j) in stored]
with ProcessPoolExecutor(9) as pool:
    res = list(pool.map(run_job, jobs))
same = sum(r["detections"] == stored[r["key"]] for r in res)
print(f"{same} of {len(res)} thesis EADD runs reproduced exactly with the new code")
