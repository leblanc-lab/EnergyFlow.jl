# Single-pair EMD scaling benchmark — Python
# Usage: cd benchmark && python single_emd_benchmark_python.py
#
# Python counterpart to single_emd_benchmark.jl: times one EMD between two
# events of equal multiplicity n over the same data/event{0,1}_n*.csv samples
# and writes result/single_emd_python.md. scaling_plot.py merges the two into
# the scaling figure.
#
# Three Python paths are timed where available, because they answer different
# questions:
#
#   POT   — ot.lp.emd2 called directly on a cost matrix we build ourselves.
#           The like-for-like comparison against the Julia solvers: same
#           problem, same exact algorithm family, no wrapper in between.
#   EF    — energyflow.emd.emd_pot, the user-facing Python EnergyFlow call.
#           Shows what an analysis actually pays per pair, wrapper included.
#   Wass  — the wasserstein C++ library, the other compiled backend behind
#           Python EnergyFlow.
#
# Only numpy + POT are required. energyflow and wasserstein are optional: if
# they are not installed the script skips those curves and says so, so this
# runs on a cluster where `wasserstein` can be awkward to build.

import os
import sys
import time
from statistics import median

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envinfo import print_env, write_env_block
# Reuse the exact cost-matrix and unbalanced-EMD semantics already validated
# against EnergyFlow.jl in the pairwise benchmark, rather than restating them.
from emds_benchmark_python import euclidean_cost, ef_emd

SIZES = [2, 10, 50, 100, 200, 500, 1000, 2000, 3000]
SETUPS = [('Euclidean_norm', True), ('Euclidean_unnorm', False)]

TARGET_SECONDS = 2.0
MIN_REPS = 5
MAX_REPS = 50_000


def timed(fn):
    """Time fn() repeatedly; return (median_s, min_s, reps).

    Matches the adaptive scheme in single_emd_benchmark.jl so both sides of the
    figure are measured the same way: one untimed warmup call, a pilot
    measurement to size the repetition count against TARGET_SECONDS, then a
    median over those repetitions.
    """
    fn()
    start = time.perf_counter()
    fn()
    est = time.perf_counter() - start
    reps = int(min(max(TARGET_SECONDS / max(est, 1e-9), MIN_REPS), MAX_REPS))

    times = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return median(times), min(times), reps


def build_runners():
    """Return [(label, factory)] for each available Python implementation.

    Each factory takes (ev0, ev1, norm) and returns a zero-argument callable
    that performs one complete EMD, including cost-matrix construction — the
    Julia `emd` builds its cost matrix internally, so excluding it here would
    flatter the Python side.
    """
    runners = [('POT', lambda ev0, ev1, norm:
                (lambda: ef_emd(ev0, ev1, 1.0, 1.0, norm, euclidean_cost)))]

    try:
        from energyflow.emd import emd_pot as _ef_emd_pot
    except Exception as exc:
        print(f'energyflow not available, skipping EF curve ({exc.__class__.__name__})')
    else:
        def ef_factory(ev0, ev1, norm):
            # Python EnergyFlow takes (pT, y, phi) rows directly and applies
            # its own normalization, so `norm` selects the call rather than
            # changing the inputs.
            return lambda: _ef_emd_pot(ev0, ev1, R=1.0, beta=1.0, norm=norm)
        runners.append(('EF', ef_factory))

    try:
        import wasserstein
    except Exception as exc:
        print(f'wasserstein not available, skipping Wass curve ({exc.__class__.__name__})')
    else:
        def wass_factory(ev0, ev1, norm):
            emd_obj = wasserstein.EMD(R=1.0, beta=1.0, norm=norm)
            w0, c0 = np.ascontiguousarray(ev0[:, 0]), np.ascontiguousarray(ev0[:, 1:])
            w1, c1 = np.ascontiguousarray(ev1[:, 0]), np.ascontiguousarray(ev1[:, 1:])
            return lambda: emd_obj(w0, c0, w1, c1)
        runners.append(('Wass', wass_factory))

    return runners


def main():
    print_env()
    runners = build_runners()
    print(f'Timing implementations: {", ".join(label for label, _ in runners)}\n')

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    results = []

    for n in SIZES:
        ev0 = np.loadtxt(os.path.join(data_dir, f'event0_n{n}.csv'), delimiter=',')
        ev1 = np.loadtxt(os.path.join(data_dir, f'event1_n{n}.csv'), delimiter=',')
        # A 1-particle file loads as a flat array; keep everything 2-D.
        ev0 = np.atleast_2d(ev0)
        ev1 = np.atleast_2d(ev1)

        for setup_name, norm in SETUPS:
            for label, factory in runners:
                med, mn, reps = timed(factory(ev0, ev1, norm))
                results.append(dict(n=n, setup=setup_name, backend=label,
                                    median_s=med, min_s=mn, reps=reps))
                print(f'n={n:<5d} {setup_name:<18s} {label:<5s} '
                      f'median={med * 1e6:10.3f} µs  min={mn * 1e6:10.3f} µs  ({reps} reps)')

    os.makedirs('result', exist_ok=True)
    with open('result/single_emd_python.md', 'w') as handle:
        handle.write('# Single-Pair EMD Scaling — Python\n\n')
        handle.write('Median wall time for one EMD between two events of equal multiplicity\n')
        handle.write('n. POT = ot.lp.emd2 on a cost matrix built here; EF = Python\n')
        handle.write('EnergyFlow emd_pot; Wass = the wasserstein C++ library. Cost-matrix\n')
        handle.write('construction is included in every case.\n\n')
        write_env_block(handle)
        handle.write('| n | Setup | Backend | Median (s) | Min (s) | Reps |\n')
        handle.write('|---|---|---|---|---|---|\n')
        for r in results:
            handle.write(f"| {r['n']} | {r['setup']} | {r['backend']} | "
                         f"{r['median_s']:.9g} | {r['min_s']:.9g} | {r['reps']} |\n")
    print('\nResults saved to result/single_emd_python.md')


if __name__ == '__main__':
    main()
