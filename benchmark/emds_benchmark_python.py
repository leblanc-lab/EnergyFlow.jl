# Pairwise EMD Benchmark — Python (POT, ot.lp.emd2)
# Usage: python emds_benchmark_python.py
#
# Direct Julia-vs-POT comparison: reproduces EnergyFlow.jl's `emds` exactly
# (same ground distances, R/beta/norm handling, and the fictitious-particle
# construction for the unbalanced norm=false case) and times POT's exact
# network-simplex solver (`ot.lp.emd2`) over the same event pairs. Depends only
# on numpy + pot — no wasserstein / EnergyFlow install — matching
# isotropy_benchmark_python.py. Compare with isotropy-style side-by-side view:
#   julia --project=. emds_benchmark.jl     # writes result/emds_julia.md
#   python emds_benchmark_python.py          # writes result/emds_python.md
#   julia --project=. emds_compare.jl        # writes result/emds_compare.md

import os
import time

import numpy as np
from ot.lp import emd2

_EPS = np.sqrt(np.finfo(np.float64).eps)


def load_hepmc3(filepath, maxevents=100):
    events = []
    particles = []
    event_count = 0
    with open(filepath) as f:
        for line in f:
            if line.startswith('E '):
                if particles:
                    events.append(np.array(particles))
                    particles = []
                event_count += 1
                if maxevents > 0 and event_count > maxevents:
                    break
            elif line.startswith('P '):
                # HepMC3 P line: P id vertex_id pdg_id px py pz energy mass status
                parts = line.split()
                px, py, pz = float(parts[4]), float(parts[5]), float(parts[6])
                status = int(parts[9])
                if status == 1:
                    pt = np.sqrt(px**2 + py**2)
                    if pt <= 0.0:
                        continue
                    eta = np.arcsinh(pz / pt)
                    phi = np.arctan2(py, px)
                    particles.append([pt, eta, phi])
    if particles:
        events.append(np.array(particles))
    return events


# ── Ground distances (match src/GroundMetrics.jl) ──

def euclidean_cost(c0, c1):
    """(√Σ Δx²) over all coordinate columns; EnergyFlow EuclideanMetric."""
    diff = c0[:, np.newaxis, :] - c1[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))


def etaphi_cost(c0, c1):
    """√(Δη² + Δφ_wrap²) with φ wrapped to [0, π]; EnergyFlow EtaPhiMetric.
    Coordinate column 0 is η, column 1 is φ."""
    deta = c0[:, 0][:, np.newaxis] - c1[:, 0][np.newaxis, :]
    dphi_raw = np.abs(c0[:, 1][:, np.newaxis] - c1[:, 1][np.newaxis, :])
    dphi = np.pi - np.abs(np.mod(dphi_raw, 2 * np.pi) - np.pi)
    return np.sqrt(deta**2 + dphi**2)


def ef_emd(ev0, ev1, R, beta, norm, cost_fn):
    """EnergyFlow.jl `emd` reproduced with POT.

    Events are M×3 rows [pT, η, φ]; column 0 is the weight. The cost matrix is
    (d_ij / R)**beta. With norm=True both events are normalized to unit weight
    (balanced OT). With norm=False the lighter distribution gets a fictitious
    particle at ground cost 1 to absorb the weight difference, matching
    _emd_raw! / _fill_fict_costs! in EMDSolver.jl / Distances.jl.
    """
    w0 = ev0[:, 0].astype(np.float64)
    w1 = ev1[:, 0].astype(np.float64)
    M = (cost_fn(ev0[:, 1:], ev1[:, 1:]) / R) ** beta

    if norm:
        a = w0 / w0.sum()
        b = w1 / w1.sum()
        return emd2(a, b, M, numItermax=100000000)

    total0, total1 = w0.sum(), w1.sum()
    max_total = max(total0, total1)
    a = w0 / max_total
    b = w1 / max_total
    diff = total1 - total0
    if diff > _EPS:                       # target heavier: fictitious source
        a = np.append(a, diff / max_total)
        M = np.vstack([M, np.ones((1, M.shape[1]))])
    elif diff < -_EPS:                    # source heavier: fictitious target
        b = np.append(b, -diff / max_total)
        M = np.hstack([M, np.ones((M.shape[0], 1))])
    return emd2(a, b, M, numItermax=100000000) * max_total


def emds_pot(events_a, events_b, R, beta, norm, cost_fn):
    """Full na×nb pairwise EMD matrix via POT (mirrors EnergyFlow `emds`)."""
    na, nb = len(events_a), len(events_b)
    D = np.empty((na, nb))
    for i, ea in enumerate(events_a):
        for j, eb in enumerate(events_b):
            D[i, j] = ef_emd(ea, eb, R, beta, norm, cost_fn)
    return D


def main():
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sk_example_PU.hepmc')
    events = load_hepmc3(path, 100)
    print(f'Loaded {len(events)} events')

    splits = [("10v90", slice(0, 10), slice(10, 100)),
              ("50v50", slice(0, 50), slice(50, 100))]

    # (setup name, cost fn, norm) — matches the Julia benchmark setups.
    setups = [
        ("Euclidean_norm",   euclidean_cost, True),
        ("Euclidean_unnorm", euclidean_cost, False),
        ("EtaPhi_norm",      etaphi_cost,    True),
    ]

    results = []
    for split_name, sa, sb in splits:
        ea, eb = events[sa], events[sb]
        na, nb = len(ea), len(eb)
        for setup_name, cost_fn, norm in setups:
            emds_pot(ea[:2], eb[:2], 1.0, 1.0, norm, cost_fn)  # warmup
            t0 = time.perf_counter()
            emds_pot(ea, eb, 1.0, 1.0, norm, cost_fn)
            t1 = time.perf_counter()
            results.append(dict(split=split_name, setup=setup_name, backend="POT",
                                time_s=t1 - t0, pairs=na * nb))
            print(f"{split_name:6s} {setup_name:<18s} POT    {t1 - t0:8.2f} s  ({na * nb} pairs)")

    os.makedirs('result', exist_ok=True)
    with open('result/emds_python.md', 'w') as f:
        f.write("# Pairwise EMD Benchmark — Python (POT ot.lp.emd2)\n\n")
        f.write("| Split | Setup | Backend | Time (s) | Pairs |\n")
        f.write("|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['split']} | {r['setup']} | {r['backend']} | {r['time_s']:.2f} | {r['pairs']} |\n")
    print("\nResults saved to result/emds_python.md")


if __name__ == '__main__':
    main()
