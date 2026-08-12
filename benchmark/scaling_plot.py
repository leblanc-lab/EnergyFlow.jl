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


def parse_env_block(path):
    """Return the `## Environment` table of a result file as a dict."""
    if not os.path.isfile(path):
        return {}
    env = {}
    in_table = False
    for line in open(path):
        line = line.strip()
        if not line.startswith('|'):
            continue
        cells = [c.strip() for c in line.strip('|').split('|')]
        if len(cells) != 2 or all(set(c) <= set('-: ') for c in cells):
            continue
        if cells == ['Field', 'Value']:
            in_table = True
            continue
        if in_table:
            env[cells[0]] = cells[1]
    return env


def check_provenance(paths):
    """Warn if the inputs were not all produced on the same machine.

    The panels put Julia and Python timings on shared axes, so a figure built
    from a Julia run on one machine and a Python run on another is misleading
    in a way nothing else here would catch. A stale result file left behind by
    a failed run is the realistic way this happens.
    """
    seen = {}
    for path in paths:
        env = parse_env_block(path)
        cpu = env.get('CPU')
        if cpu:
            seen.setdefault(cpu, []).append(os.path.basename(path))
    if len(seen) > 1:
        print('WARNING: result files come from more than one machine. The figure '
              'will mix them:')
        for cpu, files in seen.items():
            print(f'  {cpu}: {", ".join(sorted(files))}')
        print('  Rerun the missing benchmarks on one machine before publishing.')
    return seen


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


def _thread_series(pattern, backend):
    """Collect {threads: seconds} from thread-tagged result files."""
    header = ['Split', 'Setup', 'Backend', 'Time (s)', 'Pairs']
    series = {}
    for name in sorted(os.listdir(RESULT_DIR)) if os.path.isdir(RESULT_DIR) else []:
        match = re.fullmatch(pattern, name)
        if not match:
            continue
        threads = int(match.group(1))
        for row in parse_markdown_table(os.path.join(RESULT_DIR, name), header):
            if (row['Split'] == '50v50' and row['Setup'] == SETUP
                    and row['Backend'] == backend):
                series[threads] = float(row['Time (s)'])
    return series


def load_thread_scaling():
    """Return (julia, wasserstein, pot_seconds) for the 50v50 workload.

    Both threaded implementations are collected per thread count, so the panel
    compares equal parallelism. POT is a single scalar: ot.lp.emd2 has no
    internal threading, so it is drawn as a one-thread reference rather than a
    curve, and is not a fair comparison at higher thread counts.
    """
    julia = _thread_series(r'emds_julia_t(\d+)\.md', 'ns64')
    wass = _thread_series(r'wass_pairwise_t(\d+)\.md', 'Wass')

    header = ['Split', 'Setup', 'Backend', 'Time (s)', 'Pairs']
    pot = None
    for row in parse_markdown_table(os.path.join(RESULT_DIR, 'emds_python.md'), header):
        if row['Split'] == '50v50' and row['Setup'] == SETUP:
            pot = float(row['Time (s)'])
    return julia, wass, pot


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


def plot_thread_scaling(ax, julia, wass, pot):
    all_threads = sorted(set(julia) | set(wass))

    for series, key, label in ((julia, 'ns64', 'EnergyFlow.jl :ns64'),
                               (wass, 'Wass', 'wasserstein (OpenMP)')):
        if not series:
            continue
        threads = sorted(series)
        color, marker, _ = STYLES[key]
        ax.plot(threads, [series[t] for t in threads], marker=marker, color=color,
                linewidth=1.6, markersize=5, label=label)

    # Ideal strong scaling from Julia's measured single-thread point, so the
    # reader can see how far the real curves depart from linear speedup.
    if julia and min(julia) == 1:
        ax.plot(all_threads, [julia[1] / t for t in all_threads], linestyle=':',
                color='0.45', linewidth=1.3, label='ideal linear scaling')

    if pot is not None:
        ax.axhline(pot, linestyle='--', color=STYLES['POT'][0], linewidth=1.3,
                   label='POT (no threading)')

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(all_threads)
    ax.set_xticklabels([str(t) for t in all_threads])
    ax.set_xlabel('Threads')
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

    inputs = [os.path.join(RESULT_DIR, name)
              for name in ('single_emd_julia.md', 'single_emd_python.md',
                           'emds_python.md')]
    if os.path.isdir(RESULT_DIR):
        inputs += [os.path.join(RESULT_DIR, name) for name in os.listdir(RESULT_DIR)
                   if re.fullmatch(r'(emds_julia|wass_pairwise)_t\d+\.md', name)]
    check_provenance(inputs)

    series = load_single_pair()
    julia_threads, wass_threads, pot = load_thread_scaling()
    by_threads = julia_threads or wass_threads

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
        plot_thread_scaling(axes[1], julia_threads, wass_threads, pot)
        if not wass_threads:
            print('No wass_pairwise_t*.md found: panel (b) shows Julia against a '
                  'single-threaded POT reference only, which is not a fair '
                  'threaded comparison. Run wass_pairwise_benchmark.py per thread count.')

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
