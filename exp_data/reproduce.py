"""Reproduce the offline results table (1M steps) from trqam-exp-data.pkl alone.

The offline endpoint is 1M steps, row 19 of each (30, 8) array. Cells are the seed mean and the
sample standard deviation (ddof = 1) in per cent, rounded half up.

    python exp_data/reproduce.py            every domain, and the all row
    python exp_data/reproduce.py --tasks    every task as well
"""
import os, pickle, argparse
import numpy as np

METHODS = ["FQL", "CGQL-L", "DSRL", "IFQL", "QAM", "QAM-E", "TRQAM"]
DOMAINS = ["antmaze-large-navigate", "antmaze-giant-navigate", "humanoidmaze-medium-navigate",
           "humanoidmaze-large-navigate", "scene-play", "puzzle-3x3-play", "puzzle-4x4-play",
           "cube-double-play", "cube-triple-play", "cube-quadruple-play"]
ROW_1M = 19                                      # 50K * (19 + 1)
R = lambda x: int(np.floor(x + 0.5))

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", action="store_true")
a = ap.parse_args()
D = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "trqam-exp-data.pkl"), "rb"))


def cell(name, m):
    v = 100 * D[(name, m)][ROW_1M]
    return f"{R(v.mean())}±{R(v.std(ddof=1))}"


print(f"{'':<46}" + "".join(f"{m:>10}" for m in METHODS))
for dom in DOMAINS:
    if a.tasks:
        for t in range(1, 6):
            n = f"{dom}-singletask-task{t}-v0"
            print(f"{n:<46}" + "".join(f"{cell(n, m):>10}" for m in METHODS))
    print(f"{dom:<46}" + "".join(f"{cell(dom, m):>10}" for m in METHODS))
# the all row is the mean of the ten printed domain means
allv = [R(np.mean([R(100 * D[(d, m)][ROW_1M].mean()) for d in DOMAINS])) for m in METHODS]
print(f"{'all':<46}" + "".join(f"{v:>10}" for v in allv))
