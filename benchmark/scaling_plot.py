# Build the scaling figure for the JOSS paper.
# Usage: cd benchmark && python scaling_plot.py [--output ../paper/scaling.png]
#
# Reads the benchmark result tables and produces a two-panel figure:
#
#   (a) Single-pair EMD wall time against particle multiplicity, log-log, for
#       the Julia backends and each available Python implementation. This is
#       where the per-call overhead of the Python interface is visible: the
#       EnergyFlow wrapper curve is flat at its call overhead until the solver
#       finally dominates, while the Julia curves start three orders of
#       magnitude lower and track the same asymptotic slope.
#
#   (b) Pairwise EMD wall time against Julia thread count, with the
#       single-threaded POT time as a reference line. This is the parallel
#       scaling that the single-pair panel cannot show.
#
# Inputs (produced by the benchmark scripts):
#   result/single_emd_julia.md      single_emd_benchmark.jl
#   result/single_emd_python.md     single_emd_benchmark_python.py
#   result/emds_julia_t{N}.md       emds_benchmark.jl, once per thread count
#   result/emds_python.md           emds_benchmark_python.py
#
# Panel (b) is omitted if fewer than two thread counts are present, so the
# figure still builds from a laptop run.

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Override to plot an archived result set, e.g. one copied off a batch node.
RESULT_DIR = os.environ.get(
    'ENERGYFLOW_RESULT_DIR',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result'))

# One consistent colour and marker per implementation across both panels.
STYLES = {
    'ns64': ('#0072B2', 'o', 'EnergyFlow.jl :ns64'),
    'ot64': ('#009E73', 's', 'EnergyFlow.jl :ot64'),
    'ns32': ('#56B4E9', 'v', 'EnergyFlow.jl :ns32'),
    'ot32': ('#66C2A5', '^', 'EnergyFlow.jl :ot32'),
    'POT':  ('#D55E00', 'D', 'POT (ot.lp.emd2)'),
    'EF':   ('#CC79A7', 'X', 'Python EnergyFlow'),
    'Wass': ('#E69F00', 'P', 'wasserstein (C++)'),
}

# Which setup the figure shows. Both benchmarks record normalized and
# unnormalized variants; the figure uses one to stay legible.
SETUP = 'Euclidean_norm'


def parse_markdown_table(path, expected_header):
    """Yield result rows from a benchmark markdown file as dicts.

    The files contain two tables: a 2-column environment block and the results.
    Rows are matched against `expected_header` so the environment block, the
    separator rows, and any prose are skipped.
    """
    if not os.path.isfile(path):
        return []
    rows = []
    header = None
    for line in open(path):
        line = line.strip()
        if not line.startswith('|'):
            continue
        cells = [c.strip() for c in line.strip('|').split('|')]
        if all(set(c) <= set('-: ') for c in cells):
            continue                                   # separator row
        if cells == expected_header:
            header = cells
            continue
        if header is None or len(cells) != len(header):
            continue                                   # different table
        rows.append(dict(zip(header, cells)))
    return rows


def load_single_pair():
    """Return {backend: [(n, seconds), ...]} for the chosen setup."""
    header = ['n', 'Setup', 'Backend', 'Median (s)', 'Min (s)', 'Reps']
    series = {}
    for name in ('single_emd_julia.md', 'single_emd_python.md'):
        for row in parse_markdown_table(os.path.join(RESULT_DIR, name), header):
            if row['Setup'] != SETUP:
                continue
            series.setdefault(row['Backend'], []).append(
                (int(row['n']), float(row['Median (s)'])))
    for points in series.values():
        points.sort()
    return series


def load_thread_scaling():
    """Return ({threads: seconds}, pot_seconds) for one pairwise workload.

    Uses the 50v50 split, the larger of the two, so the measurement is not
    dominated by startup. Backends are summed to a single per-thread number by
    taking :ns64, the default.
    """
    header = ['Split', 'Setup', 'Backend', 'Time (s)', 'Pairs']
    by_threads = {}
    for name in sorted(os.listdir(RESULT_DIR)) if os.path.isdir(RESULT_DIR) else []:
        match = re.fullmatch(r'emds_julia_t(\d+)\.md', name)
        if not match:
            continue
        threads = int(match.group(1))
        for row in parse_markdown_table(os.path.join(RESULT_DIR, name), header):
            if row['Split'] == '50v50' and row['Setup'] == SETUP and row['Backend'] == 'ns64':
                by_threads[threads] = float(row['Time (s)'])

    pot = None
    for row in parse_markdown_table(os.path.join(RESULT_DIR, 'emds_python.md'), header):
        if row['Split'] == '50v50' and row['Setup'] == SETUP:
            pot = float(row['Time (s)'])
    return by_threads, pot


def plot_single_pair(ax, series):
    for backend, points in series.items():
        if backend not in STYLES:
            continue
        color, marker, label = STYLES[backend]
        ns = [p[0] for p in points]
        times = [p[1] * 1e6 for p in points]           # microseconds
        ax.plot(ns, times, marker=marker, color=color, label=label,
                linewidth=1.6, markersize=5)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Particles per event, $n$')
    ax.set_ylabel('Time per EMD (µs)')
    ax.set_title('(a) Single-pair EMD', loc='left', fontsize=10)
    ax.grid(True, which='major', alpha=0.3)
    ax.grid(True, which='minor', alpha=0.12)
    ax.legend(fontsize=7.5, frameon=False, loc='upper left')


def plot_thread_scaling(ax, by_threads, pot):
    threads = sorted(by_threads)
    times = [by_threads[t] for t in threads]
    color, marker, _ = STYLES['ns64']

    ax.plot(threads, times, marker=marker, color=color, linewidth=1.6,
            markersize=5, label='EnergyFlow.jl :ns64')

    # Ideal strong scaling from the measured single-thread point, so the reader
    # can see how far the real curve departs from linear speedup.
    if threads and threads[0] == 1:
        ideal = [times[0] / t for t in threads]
        ax.plot(threads, ideal, linestyle=':', color='0.45', linewidth=1.3,
                label='ideal linear scaling')

    if pot is not None:
        ax.axhline(pot, linestyle='--', color=STYLES['POT'][0], linewidth=1.3,
                   label='POT (single-threaded)')

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(threads)
    ax.set_xticklabels([str(t) for t in threads])
    ax.set_xlabel('Julia threads')
    ax.set_ylabel('Time for 2500 pairs (s)')
    ax.set_title('(b) Pairwise EMD scaling', loc='left', fontsize=10)
    ax.grid(True, which='major', alpha=0.3)
    ax.legend(fontsize=7.5, frameon=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'paper', 'scaling.png'))
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()

    series = load_single_pair()
    by_threads, pot = load_thread_scaling()

    if not series:
        sys.exit('No single-pair results found. Run single_emd_benchmark.jl and '
                 'single_emd_benchmark_python.py first (see benchmark.md).')

    two_panel = len(by_threads) >= 2
    if not two_panel:
        print('Fewer than two thread counts found in result/emds_julia_t*.md; '
              'building the single-pair panel only.')

    width = 9.0 if two_panel else 5.0
    fig, axes = plt.subplots(1, 2 if two_panel else 1, figsize=(width, 3.6))
    axes = axes if two_panel else [axes]

    plot_single_pair(axes[0], series)
    if two_panel:
        plot_thread_scaling(axes[1], by_threads, pot)

    fig.tight_layout()
    output = os.path.normpath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches='tight')
    print(f'Figure written to {output}')

    missing = [b for b in ('EF', 'Wass') if b not in series]
    if missing:
        print(f'Note: no data for {", ".join(missing)} — install energyflow / '
              'wasserstein and rerun single_emd_benchmark_python.py to include them.')


if __name__ == '__main__':
    main()
