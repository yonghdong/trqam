# Experiment Data

We release the experiment data to facilitate future research. The layout follows the
[QAM release](https://github.com/ColinQiyangLi/qam/tree/main/exp_data).

`trqam-exp-data.pkl` is a dictionary keyed by `(name, method)` tuples. `name` is an OGBench task
(e.g. `cube-triple-play-singletask-task2-v0`), a domain (e.g. `cube-triple-play`), or `all`. Each
entry is a numpy array of shape `(31, 8)`. The array stores the success rate at a regular interval of
50K training steps, from 0 to 1.5M, for 8 seeds. Row `i` is step `50K * i`, so row 0 is the start of
training and row 20 is the offline endpoint. Steps up to 1M are offline and the rest are online.

- A task entry is that run's success rate.
- A domain entry is, per seed, the mean over the domain's five tasks.
- The `all` entry is, per seed, the mean over the ten domains.

Methods: `FQL`, `CGQL-L`, `DSRL`, `IFQL`, `QAM`, `QAM-E`, `TRQAM`.

Datasets: `antmaze-giant-navigate`, `puzzle-4x4-play` and `cube-triple-play` use the 10M datasets and
`cube-quadruple-play` the 100M dataset. The other six domains use the standard 1M datasets.

## Reproducing the offline results

    python exp_data/reproduce.py            every domain, and the all row
    python exp_data/reproduce.py --tasks    every task as well

The offline endpoint (1M steps) is row 20. Each cell is the seed mean and the sample standard
deviation (ddof = 1) in per cent, rounded half up. The `all` row is the mean of the ten domain means
as printed.
