#!/usr/bin/env python3
"""Parse HepMC3 events, compute pairwise EMDs with energyflow, save to CSV."""
import os, time, sys
import numpy as np

DATADIR = os.path.join(os.path.dirname(__file__), "..", "data")
HEPMC_FILE = os.path.join(DATADIR, "sk_example_PU.hepmc")


def load_hepmc3_events(filepath, maxevents=-1):
    """Parse HepMC3 ASCII → list of Nx3 arrays [pt, eta, phi]."""
    events = []
    particles = []  # list of (px, py, pz)
    event_count = 0

    with open(filepath) as f:
        for line in f:
            if line.startswith("E "):
                if particles:
                    events.append(_to_ptEtaPhi(particles))
                    particles = []
                event_count += 1
                if 0 < maxevents < event_count:
                    break
            elif line.startswith("P "):
                tok = line.split()
                status = int(tok[9])
                if status == 1:
                    px, py, pz = float(tok[4]), float(tok[5]), float(tok[6])
                    particles.append((px, py, pz))

        if particles:
            events.append(_to_ptEtaPhi(particles))

    return events


def _to_ptEtaPhi(particles):
    arr = np.array(particles)  # Nx3: px, py, pz
    px, py, pz = arr[:, 0], arr[:, 1], arr[:, 2]
    pt = np.sqrt(px**2 + py**2)
    p = np.sqrt(px**2 + py**2 + pz**2)
    # eta = arctanh(pz/p)
    ratio = np.clip(pz / np.where(p > 0, p, 1.0), -1 + 1e-15, 1 - 1e-15)
    eta = np.arctanh(ratio)
    phi = np.arctan2(py, px)
    return np.column_stack([pt, eta, phi])


if __name__ == "__main__":
    import energyflow

    # Load events
    print("Loading HepMC3 events...")
    t0 = time.time()
    events = load_hepmc3_events(HEPMC_FILE, maxevents=100)
    print(f"Loaded {len(events)} events in {time.time()-t0:.3f} s")
    for i in range(min(5, len(events))):
        print(f"  Event {i+1}: {events[i].shape[0]} particles")

    # Define splits
    splits = [
        ("10v90", events[:10], events[10:100]),
        ("50v50", events[:50], events[50:100]),
    ]

    norms_list = [True, False]

    print("\n=== Computing EMDs (Python energyflow) ===")
    for split_name, evA, evB in splits:
        for norm_val in norms_list:
            norm_str = "norm" if norm_val else "unnorm"
            label = f"{split_name}_{norm_str}"
            print(f"\n--- {label} ---")

            nA, nB = len(evA), len(evB)
            D = np.zeros((nA, nB))

            t0 = time.time()
            for i in range(nA):
                for j in range(nB):
                    D[i, j] = energyflow.emd.emd(evA[i], evB[j], R=1.0, beta=1.0, norm=norm_val)
            elapsed = time.time() - t0

            print(f"  {label}: {elapsed:.4f} s  |  shape {D.shape}  |  mean={D.mean():.6e}  max={D.max():.6e}")

            csv_path = os.path.join(DATADIR, f"python_emds_{label}.csv")
            np.savetxt(csv_path, D, delimiter=",", fmt="%.15e")
            print(f"  Saved -> {csv_path}")

    print("\nDone.")
