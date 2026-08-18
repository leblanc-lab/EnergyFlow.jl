# Event Isotropy Benchmark — Python (POT, ot.lp.emd2)
# Usage: python isotropy_benchmark_python.py
#
# Computes the event isotropy (arXiv:2004.06125) of each event against
# quasi-uniform ring, cylinder (hadron-collider) and sphere (lepton-collider)
# references, calling POT's exact network simplex solver directly. The
# reference generation and ground distances reproduce C. Cesarotti's
# eventIsotropy package (github.com/caricesarotti/event_isotropy) so the
# numbers are directly comparable to isotropy_benchmark.jl.
#
# The sphere reference uses HEALPix pixel centers. spherGen.py uses
# astropy_healpix; the pure-numpy healpix_pix2vec_ring below reproduces
# astropy_healpix.healpy.pix2vec to ~1e-11, keeping this script's dependencies
# at numpy + pot.

import os
import time

import numpy as np
from ot.lp import emd2


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


def load_hepmc3_momenta(filepath, maxevents=100):
    """Load events as [E, px, py, pz] rows (input for spherical isotropy)."""
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
                parts = line.split()
                px, py, pz, e = (float(parts[4]), float(parts[5]),
                                 float(parts[6]), float(parts[7]))
                status = int(parts[9])
                if status == 1:
                    particles.append([e, px, py, pz])
    if particles:
        events.append(np.array(particles))
    return events


# ── Quasi-uniform references (cylGen.ringGen / cylGen.cylinderGen) ──

def ring_reference(n):
    """Ring of n equal-weight points at phi_j = 2*pi*(j+0.5)/n."""
    return np.array([2 * np.pi * (j + 0.5) / n for j in range(n)])


def cylinder_reference(nphi, ymax):
    """nphi x floor(ymax*nphi/pi) grid over |y| <= ymax; returns (y, phi) rows."""
    ny = int(np.floor(ymax * nphi / np.pi))
    phis = [2 * np.pi * (j + 0.5) / nphi for j in range(nphi)]
    ys = [-ymax + 2 * ymax * (i + 0.5) / ny for i in range(ny)]
    return np.array([[y, phi] for phi in phis for y in ys])


def healpix_pix2vec_ring(nside):
    """Unit vectors at HEALPix pixel centers (RING scheme); reproduces
    astropy_healpix.healpy.pix2vec to ~1e-11. nside must be a power of two."""
    npix = 12 * nside * nside
    ncap = 2 * nside * (nside - 1)
    vecs = np.empty((npix, 3))
    for p in range(npix):
        if p < ncap:                                   # north polar cap
            ph = (p + 1) / 2.0
            i = int(np.floor(np.sqrt(ph - np.sqrt(int(ph))))) + 1
            j = p + 1 - 2 * i * (i - 1)
            z = 1.0 - i * i / (3.0 * nside * nside)
            phi = (np.pi / (2 * i)) * (j - 0.5)
        elif p < npix - ncap:                          # equatorial belt
            pp = p - ncap
            i = pp // (4 * nside) + nside
            j = (pp % (4 * nside)) + 1
            s = (i - nside + 1) % 2
            z = 4.0 / 3.0 - 2.0 * i / (3.0 * nside)
            phi = (np.pi / (2 * nside)) * (j - s / 2.0)
        else:                                          # south polar cap
            pp = npix - p
            ph = pp / 2.0
            i = int(np.floor(np.sqrt(ph - np.sqrt(int(ph))))) + 1
            j = 4 * i + 1 - (pp - 2 * i * (i - 1))
            z = -1.0 + i * i / (3.0 * nside * nside)
            phi = (np.pi / (2 * i)) * (j - 0.5)
        sth = np.sqrt(1 - z * z)
        vecs[p] = (sth * np.cos(phi), sth * np.sin(phi), z)
    return vecs


def sphere_reference(nVal):
    """Equal-weight HEALPix unit sphere with 12*(2^nVal)^2 points."""
    return healpix_pix2vec_ring(2 ** nVal)


# ── Ground distances (emdVar._cdist_phicos / emdVar._cdist_phi_y) ──

def wrap_dphi(phi1, phi2):
    d = np.abs(phi1[:, np.newaxis] - phi2[np.newaxis, :])
    return np.pi - np.abs(np.mod(d, 2 * np.pi) - np.pi)


def cdist_ring_cos(phi1, phi2):
    return (np.pi / (np.pi - 2)) * (1 - np.cos(wrap_dphi(phi1, phi2)))


def cdist_cylinder(X, Y, ymax):
    """X, Y are (y, phi) rows; 12/(pi^2 + 16*ymax^2) * (dy^2 + dphi^2)."""
    dphi = wrap_dphi(X[:, 1], Y[:, 1])
    dy = X[:, 0, np.newaxis] - Y[:, 0]
    return (12.0 / (np.pi**2 + 16 * ymax**2)) * (dphi**2 + dy**2)


def cdist_sphere_cos(P, Q):
    """P is (px,py,pz) rows, Q unit vectors; 2*(1 - cos theta). emdVar._cdist_cos."""
    Pn = P / np.linalg.norm(P, axis=1, keepdims=True)
    Qn = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    cos = np.clip(Pn @ Qn.T, -1.0, 1.0)
    return 2.0 * (1.0 - cos)


def event_isotropy(weights, M, ref_weights):
    """EMD via POT between normalized weights over cost matrix M."""
    a = weights / weights.sum()
    return emd2(a, ref_weights, M, numItermax=100000000)


def select_events(events, ymax, min_particles=2):
    """Acceptance preselection: |y| <= ymax, drop events with < min_particles.

    Must match EnergyFlow.select_rapidity for a fair comparison.
    """
    out = []
    for ev in events:
        sel = ev[np.abs(ev[:, 1]) <= ymax]
        if len(sel) >= min_particles:
            out.append(sel)
    return out


def select_sphere_events(events, min_particles=2):
    """Drop zero-momentum particles and events with < min_particles.

    Must match EnergyFlow.select_sphere_events for a fair comparison.
    """
    out = []
    for ev in events:
        sel = ev[np.sum(ev[:, 1:4] ** 2, axis=1) > 0]
        if len(sel) >= min_particles:
            out.append(sel)
    return out


def main():
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sk_example_PU.hepmc')
    events = load_hepmc3(path, 100)
    ymax = 4.0
    selected = select_events(events, ymax)
    print(f'Loaded {len(events)} events, {len(selected)} after |y| <= {ymax} selection')

    # Sphere (lepton-collider) uses full 3-momenta with energy weighting.
    sphere_selected = select_sphere_events(load_hepmc3_momenta(path, 100))

    ring128 = ring_reference(128)
    ring2 = ring_reference(2)
    cyl16 = cylinder_reference(16, ymax)
    sph192 = sphere_reference(2)
    sph48 = sphere_reference(1)

    def w(ref):
        return np.full(len(ref), 1.0 / len(ref))

    # (name, event list, prep(ev) -> (weights, cost matrix, ref weights))
    setups = [
        ("ring128",   selected,        lambda ev: (ev[:, 0], cdist_ring_cos(ev[:, 2], ring128), w(ring128))),
        ("ring2",     selected,        lambda ev: (ev[:, 0], cdist_ring_cos(ev[:, 2], ring2), w(ring2))),
        ("cyl16",     selected,        lambda ev: (ev[:, 0], cdist_cylinder(ev[:, 1:], cyl16, ymax), w(cyl16))),
        ("sphere192", sphere_selected, lambda ev: (ev[:, 0], cdist_sphere_cos(ev[:, 1:4], sph192), w(sph192))),
        ("sphere48",  sphere_selected, lambda ev: (ev[:, 0], cdist_sphere_cos(ev[:, 1:4], sph48), w(sph48))),
    ]

    results = []
    for setup_name, evs, prep in setups:
        # Warmup
        for ev in evs[:2]:
            event_isotropy(*prep(ev))

        t0 = time.perf_counter()
        vals = []
        for ev in evs:
            vals.append(event_isotropy(*prep(ev)))
        t1 = time.perf_counter()
        elapsed = t1 - t0
        mean_iso = float(np.mean(vals))
        results.append(dict(setup=setup_name, backend="POT", time_s=elapsed,
                            events=len(evs), mean_iso=mean_iso))
        print(f"{setup_name:9s} POT    {elapsed:8.3f} s  ({len(evs)} events, mean isotropy {mean_iso:.6f})")

    # Save results
    os.makedirs('result', exist_ok=True)
    with open('result/isotropy_python.md', 'w') as f:
        f.write("# Event Isotropy Benchmark — Python (POT ot.lp.emd2)\n\n")
        ny = int(np.floor(ymax * 16 / np.pi))
        f.write(f"ring: N points in phi, (pi/(pi-2))*(1-cos dphi); cylinder: 16x{ny} grid, |y| <= {ymax}\n")
        f.write("sphere: HEALPix unit sphere (192/48 points), 2*(1-cos theta) on 3-momentum directions\n\n")
        f.write("| Setup | Backend | Time (s) | Events | Mean isotropy |\n")
        f.write("|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['setup']} | {r['backend']} | {r['time_s']:.3f} | {r['events']} | {r['mean_iso']:.6f} |\n")
    print("\nResults saved to result/isotropy_python.md")


if __name__ == '__main__':
    main()
