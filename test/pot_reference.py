# POT reference values for the EnergyFlow.jl EMD tests
# Usage: python pot_reference.py          (needs numpy + pot only)
#
# The Julia test suite (test/runtests.jl, test/test_emd.jl) asserts EMD values
# against hardcoded analytic numbers. This script recomputes those same small
# cases with POT's exact solver (ot.lp.emd2) and prints them next to the value
# the Julia tests expect, so the hardcoded numbers can be confirmed as
# POT-correct without adding a Python dependency to the CI test run.
#
# It reproduces EnergyFlow.jl's `emd` exactly: cost matrix (d_ij / R)**beta,
# balanced OT when norm=True, and a fictitious particle at ground cost 1 for
# the unbalanced norm=False case (see EMDSolver.jl / Distances.jl).

import math

import numpy as np
from ot.lp import emd2

_EPS = np.sqrt(np.finfo(np.float64).eps)


def ef_emd(ev0, ev1, R=1.0, beta=1.0, norm=True, cost=None):
    """EnergyFlow.jl `emd` via POT.

    ev0, ev1 are M×(1+g) arrays: column 0 = weight, columns 1: = coordinates.
    If `cost` (an n0×n1 matrix) is given it is used directly as the ground
    distance (PrecomputedMetric); otherwise the Euclidean distance over all
    coordinate columns is used.
    """
    ev0 = np.atleast_2d(np.asarray(ev0, dtype=np.float64))
    ev1 = np.atleast_2d(np.asarray(ev1, dtype=np.float64))
    w0, w1 = ev0[:, 0], ev1[:, 0]
    if cost is None:
        diff = ev0[:, np.newaxis, 1:] - ev1[np.newaxis, :, 1:]
        cost = np.sqrt(np.sum(diff**2, axis=2))
    M = (np.asarray(cost, dtype=np.float64) / R) ** beta

    if norm:
        a = w0 / w0.sum()
        b = w1 / w1.sum()
        return emd2(a, b, M, numItermax=100000000)

    total0, total1 = w0.sum(), w1.sum()
    max_total = max(total0, total1)
    a = w0 / max_total
    b = w1 / max_total
    d = total1 - total0
    if d > _EPS:
        a = np.append(a, d / max_total)
        M = np.vstack([M, np.ones((1, M.shape[1]))])
    elif d < -_EPS:
        b = np.append(b, -d / max_total)
        M = np.hstack([M, np.ones((M.shape[0], 1))])
    return emd2(a, b, M, numItermax=100000000) * max_total


def main():
    # (label, POT value, value asserted in the Julia tests)
    cases = []

    # emd_ns64 — identical distributions  (runtests.jl)
    cases.append(("identical",
                  ef_emd([[0.5, 0.0], [0.5, 1.0]], [[0.5, 0.0], [0.5, 1.0]]),
                  0.0))
    # emd_ns64 — shifted distributions
    cases.append(("shifted (beta=1,R=1)",
                  ef_emd([[1.0, 0.0]], [[1.0, 1.0]]),
                  1.0))
    # emd_ns64 — beta=2
    cases.append(("beta=2",
                  ef_emd([[1.0, 0.0]], [[1.0, 2.0]], beta=2.0),
                  4.0))
    # emd_ns64 — R scaling
    cases.append(("R=2",
                  ef_emd([[1.0, 0.0]], [[1.0, 1.0]], R=2.0),
                  0.5))
    # Float32 — beta=0.5 / R=0.5 (same analytic values as Float64)
    cases.append(("beta=0.5",
                  ef_emd([[1.0, 0.0]], [[1.0, 2.0]], beta=0.5),
                  math.sqrt(2.0)))
    cases.append(("R=0.5",
                  ef_emd([[1.0, 0.0]], [[1.0, 1.0]], R=0.5),
                  2.0))
    # gdim + custom metric — sliced 2D coords collapse to 0
    ev0 = [[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 2.0, 3.0]]
    ev1 = [[1.0, 0.0, 0.0, 9.0], [1.0, 1.0, 2.0, 8.0]]
    sliced = ef_emd([r[:3] for r in ev0], [r[:3] for r in ev1])   # gdim=2
    full = ef_emd(ev0, ev1)                                        # gdim=3
    cases.append(("gdim=2 sliced", sliced, 0.0))
    cases.append(("gdim=3 full (test: > sliced)", full, None))
    # PrecomputedMetric — costs = [0 2; 3 4]
    cases.append(("precomputed costs [0 2;3 4]",
                  ef_emd([[1.0, 0.0, 0.0], [1.0, 1.0, 2.0]],
                         [[1.0, 0.0, 0.0], [1.0, 1.0, 2.0]],
                         cost=[[0.0, 2.0], [3.0, 4.0]]),
                  2.0))
    # NetworkSimplex direct usage — costs [1 2; 3 4], uniform 0.5 weights
    cases.append(("network_simplex [1 2;3 4]",
                  ef_emd([[0.5, 0.0], [0.5, 1.0]], [[0.5, 0.0], [0.5, 1.0]],
                         cost=[[1.0, 2.0], [3.0, 4.0]]),
                  2.5))

    print("POT reference for EnergyFlow.jl EMD tests\n")
    print(f"{'case':<32s} {'POT':>14s} {'test expects':>14s} {'match':>7s}")
    print("-" * 70)
    ok = True
    for label, got, expected in cases:
        if expected is None:
            print(f"{label:<32s} {got:>14.9f} {'(>0)':>14s} {'--':>7s}")
            continue
        good = abs(got - expected) < 1e-9
        ok = ok and good
        print(f"{label:<32s} {got:>14.9f} {expected:>14.9f} {'OK' if good else 'FAIL':>7s}")

    # ns32/ns64 cross-validation events over the (beta, R, norm) grid — the
    # tests only assert ns32≈ns64, so here we just print POT's exact values.
    print("\ncross-validation events over (beta, R, norm) grid — POT exact values")
    xv0 = [[0.3, 0.0, 0.0], [0.7, 1.0, 2.0]]
    xv1 = [[0.4, 0.5, 0.5], [0.6, 1.5, 1.0]]
    print(f"{'beta':>5s} {'R':>5s} {'norm':>6s} {'POT':>14s}")
    for beta in (0.5, 1.0, 2.0):
        for R in (0.5, 1.0, 2.0):
            for norm in (True, False):
                v = ef_emd(xv0, xv1, R=R, beta=beta, norm=norm)
                print(f"{beta:>5.1f} {R:>5.1f} {str(norm):>6s} {v:>14.9f}")

    print("\nAll analytic cases match POT." if ok else "\nMISMATCH — see FAIL rows above.")


if __name__ == '__main__':
    main()
