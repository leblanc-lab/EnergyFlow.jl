# Build the event-isotropy figure for the JOSS paper.
# Usage: cd benchmark && python isotropy_plot.py [--output ../paper/isotropy.png]
#
# The isotropy benchmark produces the only results in this directory that were
# never plotted — they lived in a comparison table, where the one thing a table
# is bad at showing is that a number disagrees with the reference it should
# match. Three panels:
#
#   (a) Wall time per reference geometry, for each Julia backend and POT. The
#       geometries span three orders of magnitude in cost, from a 2-point ring
#       to a 3072-point sphere, so this is a log axis and the ordering by cost
#       carries the information.
#
#   (b) Speedup over the single-threaded POT reference, at 1 and 8 Julia
#       threads. The 1-thread bars are the solver-vs-solver comparison; the
#       8-thread bars are the realistic workload but are not a fair
#       solver comparison, since ot.lp.emd2 has no threading.
#
#   (c) Agreement with POT on the physics value. Every backend should reproduce
#       POT's mean isotropy to floating-point precision; where a bar rises off
#       the floor, that backend and that geometry disagree, which matters more
#       than any timing on the other two panels.
#
# Inputs (produced by the benchmark scripts):
#   result/isotropy_julia_t{N}.md   isotropy_benchmark.jl, once per thread count
#   result/isotropy_python.md       isotropy_benchmark_python.py

import argparse
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scaling_plot import RESULT_DIR, parse_markdown_table, check_provenance

JULIA_HEADER = ['Setup', 'Backend', 'Time (s)', 'Min (s)', 'Reps', 'Events', 'Mean isotropy']
POT_HEADER = ['Setup', 'Backend', 'Time (s)', 'Events', 'Mean isotropy']

# Same hues as the other two figures: EnergyFlow.jl backends in the blue/green
# family, POT in vermillion. Float32 backends are the lighter step of their
# Float64 counterpart's hue, so precision reads as a shade and implementation as
# a hue. Checked for colour-vision separation, not eyeballed.
STYLES = {
    'ns64': ('#0072B2', 'EnergyFlow.jl :ns64'),
    'ot64': ('#009E73', 'EnergyFlow.jl :ot64'),
    'ns32': ('#56B4E9', 'EnergyFlow.jl :ns32'),
    'ot32': ('#66C2A5', 'EnergyFlow.jl :ot32'),
    'POT':  ('#D55E00', 'POT (ot.lp.emd2)'),
}
BACKENDS = ['ns64', 'ot64', 'ns32', 'ot32']

# Below this, a mean-isotropy difference is floating-point noise rather than
# disagreement. Bars are clipped here so a log axis has a floor to sit on.
AGREEMENT_FLOOR = 1e-9


def load_julia():
    """Return {threads: {(setup, backend): row}} from the thread-tagged files."""
    out = {}
    if not os.path.isdir(RESULT_DIR):
        return out
    for name in sorted(os.listdir(RESULT_DIR)):
        m = re.fullmatch(r'isotropy_julia_t(\d+)\.md', name)
        if not m:
            continue
        threads = int(m.group(1))
        rows = {}
        for row in parse_markdown_table(os.path.join(RESULT_DIR, name), JULIA_HEADER):
            rows[(row['Setup'], row['Backend'])] = dict(
                time_s=float(row['Time (s)']),
                mean_iso=float(row['Mean isotropy']),
                events=int(row['Events']))
        if rows:
            out[threads] = rows
    return out


def load_pot():
    """Return {setup: row} for the POT reference."""
    rows = {}
    for row in parse_markdown_table(os.path.join(RESULT_DIR, 'isotropy_python.md'), POT_HEADER):
        rows[row['Setup']] = dict(time_s=float(row['Time (s)']),
                                  mean_iso=float(row['Mean isotropy']),
                                  events=int(row['Events']))
    return rows


def order_setups(pot, julia):
    """Setups ordered by cost, so every panel reads left-to-right as harder."""
    setups = set(pot)
    for rows in julia.values():
        setups |= {s for s, _ in rows}
    ref = pot or next(iter(julia.values()), {})

    def cost(setup):
        if setup in pot:
            return pot[setup]['time_s']
        for rows in julia.values():
            if (setup, 'ns64') in rows:
                return rows[(setup, 'ns64')]['time_s']
        return 0.0

    return sorted(setups, key=cost)


def plot_times(ax, julia, pot, setups, threads):
    """(a) Wall time per geometry, grouped by backend."""
    rows = julia.get(threads, {})
    x = np.arange(len(setups))
    series = [(b, [rows.get((s, b), {}).get('time_s') for s in setups]) for b in BACKENDS]
    series.append(('POT', [pot.get(s, {}).get('time_s') for s in setups]))

    width = 0.8 / len(series)
    for i, (key, values) in enumerate(series):
        color, label = STYLES[key]
        offset = (i - (len(series) - 1) / 2) * width
        xs = [xi + offset for xi, v in zip(x, values) if v is not None]
        ys = [v for v in values if v is not None]
        ax.bar(xs, ys, width=width * 0.92, color=color, label=label, zorder=3)

    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(setups, rotation=30, ha='right', fontsize=7.5)
    ax.set_ylabel('Wall time for all events (s)')
    ax.set_title(f'(a) Event isotropy cost, {threads} thread(s)', loc='left', fontsize=10)
    ax.grid(True, axis='y', which='major', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, frameon=False, loc='upper left', ncol=2)


def plot_speedup(ax, julia, pot, setups):
    """(b) Speedup over single-threaded POT, per thread count."""
    thread_counts = sorted(julia)
    x = np.arange(len(setups))
    # One bar per (thread count, backend) would be unreadable; the panel shows
    # the default backend and the arc-mixing one, which is where the interesting
    # spread is, at each thread count.
    combos = [(t, b) for t in thread_counts for b in ('ns64', 'ot64')]
    width = 0.8 / max(len(combos), 1)

    for i, (threads, backend) in enumerate(combos):
        rows = julia[threads]
        color, _ = STYLES[backend]
        offset = (i - (len(combos) - 1) / 2) * width
        xs, ys = [], []
        for xi, setup in zip(x, setups):
            entry = rows.get((setup, backend))
            ref = pot.get(setup)
            if entry and ref and entry['time_s'] > 0:
                xs.append(xi + offset)
                ys.append(ref['time_s'] / entry['time_s'])
        # Thread count is encoded by alpha within a backend's hue, so the two
        # variables stay separable without inventing new colours.
        alpha = 0.55 if threads == min(thread_counts) else 1.0
        ax.bar(xs, ys, width=width * 0.92, color=color, alpha=alpha, zorder=3,
               label=f':{backend}, {threads} thread(s)')

    ax.axhline(1.0, color='0.35', linewidth=1.2, linestyle='--', zorder=2,
               label='POT parity')
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(setups, rotation=30, ha='right', fontsize=7.5)
    ax.set_ylabel('Speedup over POT ($>1$ faster)')
    ax.set_title('(b) Speedup over single-threaded POT', loc='left', fontsize=10)
    ax.grid(True, axis='y', which='major', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.5, frameon=False, loc='upper left', ncol=2)


def plot_agreement(ax, julia, pot, setups):
    """(c) |mean isotropy − POT| per backend and geometry.

    The panel that justifies the other two. A timing is only meaningful if the
    backend produced the right answer, and this is where that is checked: bars
    on the floor agree with POT to floating point, bars above it do not.
    """
    thread_counts = sorted(julia)
    x = np.arange(len(setups))
    width = 0.8 / len(BACKENDS)
    disagreements = []

    for i, backend in enumerate(BACKENDS):
        color, label = STYLES[backend]
        offset = (i - (len(BACKENDS) - 1) / 2) * width
        xs, ys = [], []
        for xi, setup in zip(x, setups):
            ref = pot.get(setup)
            if not ref:
                continue
            # Worst disagreement across thread counts. A backend whose value
            # depends on thread count is broken in a way an average would hide.
            diffs = []
            for threads in thread_counts:
                entry = julia[threads].get((setup, backend))
                if entry:
                    diffs.append(abs(entry['mean_iso'] - ref['mean_iso']))
            if not diffs:
                continue
            worst = max(diffs)
            xs.append(xi + offset)
            ys.append(max(worst, AGREEMENT_FLOOR))
            if worst > 1e-6:
                disagreements.append((setup, backend, worst))
        ax.bar(xs, ys, width=width * 0.92, color=color, label=label, zorder=3)

    ax.set_yscale('log')
    ax.set_ylim(bottom=AGREEMENT_FLOOR / 2)
    ax.axhline(1e-6, color='0.35', linewidth=1.2, linestyle='--', zorder=2,
               label='agreement threshold')
    ax.set_xticks(x)
    ax.set_xticklabels(setups, rotation=30, ha='right', fontsize=7.5)
    ax.set_ylabel('$|$mean isotropy $-$ POT$|$')
    ax.set_title('(c) Agreement with POT (worst over threads)', loc='left', fontsize=10)
    ax.grid(True, axis='y', which='major', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.5, frameon=False, loc='upper left', ncol=2)
    return disagreements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'paper', 'isotropy.png'))
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()

    inputs = [os.path.join(RESULT_DIR, 'isotropy_python.md')]
    if os.path.isdir(RESULT_DIR):
        inputs += [os.path.join(RESULT_DIR, n) for n in os.listdir(RESULT_DIR)
                   if re.fullmatch(r'isotropy_julia_t\d+\.md', n)]
    check_provenance(inputs)

    julia = load_julia()
    pot = load_pot()
    if not julia:
        sys.exit('No result/isotropy_julia_t*.md found. Run isotropy_benchmark.jl '
                 'first (see benchmark.md).')
    if not pot:
        print('No result/isotropy_python.md found: the speedup and agreement panels '
              'need the POT reference and will be empty. Run isotropy_benchmark_python.py.')

    setups = order_setups(pot, julia)
    # Panel (a) shows one thread count; the highest available is the one the
    # package would actually be run at.
    show_threads = max(julia)

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.0))
    plot_times(axes[0], julia, pot, setups, show_threads)
    plot_speedup(axes[1], julia, pot, setups)
    disagreements = plot_agreement(axes[2], julia, pot, setups)

    fig.tight_layout()
    output = os.path.normpath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches='tight')
    print(f'Figure written to {output}')

    if disagreements:
        print('\nBackends disagreeing with POT by more than 1e-6 in mean isotropy:')
        for setup, backend, diff in sorted(disagreements, key=lambda d: -d[2]):
            print(f'  {setup:<12s} :{backend:<6s} |Δ| = {diff:.3e}')
        print('These are exact solvers on identical problems, so a nonzero difference '
              'is a solver defect, not a tolerance.')


if __name__ == '__main__':
    main()
