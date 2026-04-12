# Pairwise EMD Benchmark — Python (using emds_pot / PairwiseEMD)
# Usage: python emds_benchmark_python.py
#        OMP_NUM_THREADS=8 python emds_benchmark_python.py   (for multi-threading)

import numpy as np
import time
import os
from energyflow.emd import emds_pot as ef_emds
import wasserstein


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
                    p_mag = np.sqrt(px**2 + py**2 + pz**2)
                    eta = np.arctanh(np.clip(pz/p_mag, -1+1e-15, 1-1e-15)) if p_mag > 0 else 0.0
                    phi = np.arctan2(py, px)
                    particles.append([pt, eta, phi])
    if particles:
        events.append(np.array(particles))
    return events


def main():
    events = load_hepmc3(os.path.join(os.path.dirname(__file__), '..', 'data', 'sk_example_PU.hepmc'), 100)
    print(f'Loaded {len(events)} events')

    # Number of threads (if supported). OMP_NUM_THREADS controls wasserstein's OpenMP threads.
    # n_jobs is used by energyflow's emds_pot (joblib convention).
    n_jobs = int(os.environ.get('OMP_NUM_THREADS', '1'))
    print(f'Using n_jobs={n_jobs} (also OMP_NUM_THREADS for wasserstein)')

    splits = [("10v90", slice(0, 10), slice(10, 100)), ("50v50", slice(0, 50), slice(50, 100))]

    results = []
    for split_name, sa, sb in splits:
        ea, eb = events[sa], events[sb]
        na, nb = len(ea), len(eb)

        # ── EnergyFlow POT pairwise (Euclidean) ──
        for norm in [True, False]:
            backend = "EF_POT"
            setup = f"Euclidean_{'norm' if norm else 'unnorm'}"
            # Warmup with tiny subset
            ef_emds(ea[:2], eb[:2], R=1.0, beta=1.0, norm=norm, n_jobs=n_jobs)
            t0 = time.perf_counter()
            D = ef_emds(ea, eb, R=1.0, beta=1.0, norm=norm, n_jobs=n_jobs)
            t1 = time.perf_counter()
            results.append(dict(split=split_name, setup=setup, backend=backend,
                                time_s=t1 - t0, pairs=na * nb))
            print(f"{split_name:6s} {setup:<20s} {backend:<8s} {t1 - t0:8.2f} s  ({na * nb} pairs)")

        # ── EnergyFlow POT pairwise (YPhi via periodic_phi=True) ──
        backend = "EF_POT"
        setup = "YPhi_norm"
        ef_emds(ea[:2], eb[:2], R=1.0, beta=1.0, norm=True, periodic_phi=True, n_jobs=n_jobs)  # warmup
        t0 = time.perf_counter()
        D = ef_emds(ea, eb, R=1.0, beta=1.0, norm=True, periodic_phi=True, n_jobs=n_jobs)
        t1 = time.perf_counter()
        results.append(dict(split=split_name, setup=setup, backend=backend,
                            time_s=t1 - t0, pairs=na * nb))
        print(f"{split_name:6s} {setup:<20s} {backend:<8s} {t1 - t0:8.2f} s  ({na * nb} pairs)")

        # ── wasserstein.PairwiseEMD (Euclidean distance) ──
        for norm in [True, False]:
            backend = "Wass"
            setup = f"Euclidean_{'norm' if norm else 'unnorm'}"
            pemd = wasserstein.PairwiseEMD(R=1.0, beta=1.0, norm=norm, num_threads=n_jobs)
            pemd(ea[:2], eb[:2])  # warmup
            t0 = time.perf_counter()
            pemd(ea, eb)
            D = pemd.emds()
            t1 = time.perf_counter()
            results.append(dict(split=split_name, setup=setup, backend=backend,
                                time_s=t1 - t0, pairs=na * nb))
            print(f"{split_name:6s} {setup:<20s} {backend:<8s} {t1 - t0:8.2f} s  ({na * nb} pairs)")

        # ── wasserstein.PairwiseEMDYPhi (YPhi distance with phi wrapping) ──
        backend = "Wass"
        setup = "YPhi_norm"
        pemd = wasserstein.PairwiseEMDYPhi(R=1.0, beta=1.0, norm=True, num_threads=n_jobs)
        pemd(ea[:2], eb[:2])  # warmup
        t0 = time.perf_counter()
        pemd(ea, eb)
        D = pemd.emds()
        t1 = time.perf_counter()
        results.append(dict(split=split_name, setup=setup, backend=backend,
                            time_s=t1 - t0, pairs=na * nb))
        print(f"{split_name:6s} {setup:<20s} {backend:<8s} {t1 - t0:8.2f} s  ({na * nb} pairs)")

    # Save results
    os.makedirs('result', exist_ok=True)
    with open('result/emds_python.md', 'w') as f:
        f.write(f"# Pairwise EMD Benchmark — Python (n_jobs={n_jobs})\n\n")
        f.write("| Split | Setup | Backend | Time (s) | Pairs |\n")
        f.write("|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['split']} | {r['setup']} | {r['backend']} | {r['time_s']:.2f} | {r['pairs']} |\n")
    print("\nResults saved to result/emds_python.md")


if __name__ == '__main__':
    # Required on Windows: multiprocessing uses 'spawn' and re-imports this
    # module in each worker. Without this guard, each worker would re-run
    # main() and spawn more workers → infinite process explosion.
    import multiprocessing
    multiprocessing.freeze_support()
    main()
