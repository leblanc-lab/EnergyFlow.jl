# Sinkhorn accuracy/cost benchmark — Python (POT, ot.sinkhorn2)
# Usage: cd benchmark && python sinkhorn_benchmark_python.py
#
# Python counterpart to sinkhorn_benchmark.jl. Answers the question a timing of
# EnergyFlow.jl's `:sinkhorn` backend alone cannot: when the approximate solver
# turns out to be slower than the exact one, is that this implementation or is
# it Sinkhorn? Running POT's Sinkhorn over the same events and the same epsilon
# grid separates the two.
#
# The comparison is deliberately like-for-like:
#
#   * `method='sinkhorn_log'` — POT's log-domain implementation, the same
#     formulation as src/Sinkhorn.jl. POT's default 'sinkhorn' method works in
#     the kernel domain, which underflows at the epsilon values swept here, so
#     it would be neither numerically comparable nor a fair timing.
#   * `ot.sinkhorn2` returns <gamma, M>, the plain transport cost of the
#     regularised plan, which is exactly what emd_sinkhorn returns. Neither
#     includes the entropic term in the reported value. This is checked, not
#     assumed: converged values agree with EnergyFlow.jl's to ~6 decimals.
#   * Same cost matrix, same normalization, and the exact reference is POT's own
#     ot.lp.emd2 on that matrix, so each side is scored against its own solver.
#
# Two caveats, both recorded in the result file. POT does not anneal epsilon,
# so `Iters` counts iterations at the target epsilon only, while the Julia
# script's count sums over annealing levels — the columns are not comparable
# even though the timings are. And POT's stopping rule is a norm of the marginal
# violation where EnergyFlow.jl's is a max, so the two converge to slightly
# different points at the same nominal tolerance.

import os
import sys
import time
from statistics import median

import numpy as np
import ot
from ot.lp import emd2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from envinfo import print_env, write_env_block
# The same cost-matrix convention already validated against EnergyFlow.jl.
from emds_benchmark_python import euclidean_cost

# Must match sinkhorn_benchmark.jl — the two halves of the figure share axes.
EPSILONS = ([float(e) for e in os.environ['ENERGYFLOW_SINKHORN_EPS'].split(',')]
            if 'ENERGYFLOW_SINKHORN_EPS' in os.environ
            else [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002])
EPS_SWEEP_SIZES = ([int(s) for s in os.environ['ENERGYFLOW_SINKHORN_EPS_SIZES'].split(',')]
                   if 'ENERGYFLOW_SINKHORN_EPS_SIZES' in os.environ
                   else [50, 100, 200])
SCALING_SIZES = ([int(s) for s in os.environ['ENERGYFLOW_SINKHORN_SIZES'].split(',')]
                 if 'ENERGYFLOW_SINKHORN_SIZES' in os.environ
                 else [10, 50, 100, 200, 500])
SCALING_EPSILONS = [0.05, 0.01]

MAX_ITER = int(os.environ.get('ENERGYFLOW_SINKHORN_MAXITER', '100000'))
TOL = 1e-9

TARGET_SECONDS = float(os.environ.get('ENERGYFLOW_BENCH_TARGET', '2.0'))
MIN_REPS = 3
MAX_REPS = 10_000
LONG_SOLVE_SECONDS = 10.0
LONG_SOLVE_REPS = 1


def timed(fn):
    """Time fn() repeatedly; return (median_s, min_s, reps).

    Same adaptive scheme as sinkhorn_benchmark.jl, including its long-solve
    escape hatch: past LONG_SOLVE_SECONDS the pilot measurement is reported
    directly rather than running another repetition that would cost minutes to
    reproduce the same number. Both sides of the figure are measured the same
    way, so the timings remain comparable.
    """
    start = time.perf_counter()
    fn()
    est = time.perf_counter() - start
    if est > LONG_SOLVE_SECONDS:
        return est, est, LONG_SOLVE_REPS
    reps = int(min(max(TARGET_SECONDS / max(est, 1e-9), MIN_REPS), MAX_REPS))

    times = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return median(times), min(times), reps


def grid_points():
    """The (n, epsilon) points to measure — the union of the two sweeps."""
    points = set()
    for n in EPS_SWEEP_SIZES:
        for eps in EPSILONS:
            points.add((n, eps))
    for n in SCALING_SIZES:
        for eps in SCALING_EPSILONS:
            points.add((n, eps))
    return sorted(points)


def warmup():
    """Exercise both entry points once so no measured point pays first-call cost.

    Python does not compile, but POT and numpy do first-call work (dispatch
    setup, BLAS thread pool) that would otherwise land on whichever point ran
    first. `timed` no longer warms up per point — at two minutes a solve that
    would double the job — so it is done once here instead.
    """
    probe = np.array([[1.0, 0.0, 0.0], [0.5, 0.3, 0.4], [0.7, -0.2, 0.1]])
    a = probe[:, 0] / probe[:, 0].sum()
    M = euclidean_cost(probe[:, 1:], probe[:, 1:])
    emd2(a, a, M, numItermax=100000000)
    ot.sinkhorn2(a, a, M, 0.5, method='sinkhorn_log', numItermax=100, stopThr=TOL)


def main():
    print_env()
    warmup()

    points = grid_points()
    sizes = sorted({n for n, _ in points})
    print(f'Sinkhorn sweep: {len(points)} (n, ε) points over n ∈ {sizes}')
    print(f'numItermax = {MAX_ITER}, stopThr = {TOL}, method = sinkhorn_log, normalized\n')

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    results = []

    for n in sizes:
        ev0 = np.atleast_2d(np.loadtxt(os.path.join(data_dir, f'event0_n{n}.csv'), delimiter=','))
        ev1 = np.atleast_2d(np.loadtxt(os.path.join(data_dir, f'event1_n{n}.csv'), delimiter=','))
        a = ev0[:, 0] / ev0[:, 0].sum()
        b = ev1[:, 0] / ev1[:, 0].sum()

        # Reference: POT's own exact solver on the same cost matrix, so the
        # error attributed to Sinkhorn is not contaminated by any difference
        # between the two exact implementations.
        exact = float(emd2(a, b, euclidean_cost(ev0[:, 1:], ev1[:, 1:]),
                           numItermax=100000000))

        def exact_call():
            M = euclidean_cost(ev0[:, 1:], ev1[:, 1:])
            return emd2(a, b, M, numItermax=100000000)

        med, mn, reps = timed(exact_call)
        results.append(dict(n=n, epsilon=None, backend='POT_exact', median_s=med,
                            min_s=mn, reps=reps, value=exact, relerr=0.0,
                            iters=0, converged=True))
        print(f'n={n:<5d} exact  POT (emd2)      {med * 1e3:10.3f} ms  value={exact:.9g}')

        for eps in sorted({e for m, e in points if m == n}, reverse=True):
            # Cost-matrix construction is inside the timed call, matching the
            # Julia side where `emd_sinkhorn` builds its own.
            def sinkhorn_call(_eps=eps):
                M = euclidean_cost(ev0[:, 1:], ev1[:, 1:])
                return ot.sinkhorn2(a, b, M, _eps, method='sinkhorn_log',
                                    numItermax=MAX_ITER, stopThr=TOL)

            med, mn, reps = timed(sinkhorn_call)

            # One extra solve with log=True for the diagnostics. POT reports the
            # final marginal error, so convergence is that error against the
            # tolerance rather than a flag.
            M = euclidean_cost(ev0[:, 1:], ev1[:, 1:])
            val, log = ot.sinkhorn2(a, b, M, eps, method='sinkhorn_log',
                                    numItermax=MAX_ITER, stopThr=TOL, log=True)
            val = float(val)
            err = float(np.atleast_1d(log['err'])[-1]) if len(np.atleast_1d(log.get('err', []))) else float('nan')
            iters = int(log.get('niter', 0))
            converged = bool(err <= TOL)
            relerr = (val - exact) / exact

            results.append(dict(n=n, epsilon=eps, backend='POT_sinkhorn', median_s=med,
                                min_s=mn, reps=reps, value=val, relerr=relerr,
                                iters=iters, converged=converged))
            print(f'n={n:<5d} ε={eps:<7g} POT sinkhorn_log {med * 1e3:10.3f} ms  '
                  f'value={val:.9g}  relerr={relerr:+.3e}  iters={iters:<7d} '
                  f'{"converged" if converged else "NOT CONVERGED"}')
        print()

    n_sinkhorn = sum(1 for r in results if r['backend'] == 'POT_sinkhorn')
    unconverged = sum(1 for r in results
                      if r['backend'] == 'POT_sinkhorn' and not r['converged'])
    if unconverged:
        print(f'{unconverged} of {n_sinkhorn} POT Sinkhorn points hit the {MAX_ITER}-iteration cap;')
        print('their values are biased low and are flagged in the result table.\n')

    os.makedirs('result', exist_ok=True)
    with open('result/sinkhorn_python.md', 'w') as handle:
        handle.write('# Sinkhorn Accuracy vs Cost — Python (POT)\n\n')
        handle.write('POT `ot.sinkhorn2(method="sinkhorn_log")` against POT `ot.lp.emd2` on the\n')
        handle.write('same cost matrix, over entropic regularisation strength ε and multiplicity\n')
        handle.write(f'n. Euclidean metric, normalized weights, `numItermax={MAX_ITER}`, `stopThr={TOL}`.\n\n')
        handle.write('`RelErr` is (Sinkhorn − exact) / exact. `Converged` compares the final\n')
        handle.write('marginal error POT reports against `stopThr`; a `false` row stopped at the\n')
        handle.write('iteration cap and its value is biased low.\n\n')
        handle.write('Two columns are *not* comparable with the Julia table, though the timings\n')
        handle.write('are: POT does not anneal ε, so `Iters` counts iterations at the target ε\n')
        handle.write("only, while EnergyFlow.jl's sums over annealing levels; and POT's stopping\n")
        handle.write('rule is a norm of the marginal violation where EnergyFlow.jl uses a max.\n\n')
        write_env_block(handle)
        handle.write('| n | Epsilon | Backend | Median (s) | Min (s) | Reps | Value | RelErr | Iters | Converged |\n')
        handle.write('|---|---|---|---|---|---|---|---|---|---|\n')
        for r in results:
            eps_str = 'exact' if r['epsilon'] is None else f"{r['epsilon']:.9g}"
            handle.write(f"| {r['n']} | {eps_str} | {r['backend']} | {r['median_s']:.9g} | "
                         f"{r['min_s']:.9g} | {r['reps']} | {r['value']:.12g} | "
                         f"{r['relerr']:.6e} | {r['iters']} | {str(r['converged']).lower()} |\n")
    print('Results saved to result/sinkhorn_python.md')


if __name__ == '__main__':
    main()
