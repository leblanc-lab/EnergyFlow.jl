# Build the pivot-rule figure for the JOSS paper.
# Usage: cd benchmark && python pivot_plot.py [--output ../paper/pivot.png]
#
# Compares the three network-simplex entering-arc rules. The interesting result
# is that the two questions "which rule picks better arcs" and "which rule is
# faster" have different answers, so the figure has to show both:
#
#   (a) Pivot count against multiplicity. This is the algorithmic claim, free of
#       any threading or implementation effect: a rule that reaches the same
#       optimum in fewer pivots is choosing better entering arcs. Kara &
#       Ozturan's :parallel_block does, and the parallel Dantzig baseline does
#       best of all, which is what a full-scan rule should do.
#
#   (b) Wall time against thread count at fixed n. This is where the cost of
#       those better choices shows up. :serial ignores threads entirely and is
#       drawn as a flat reference; if a parallel rule never crosses below it,
#       the better pivot choices are not paying for their overhead.
#
#   (c) Wall time against multiplicity at the best thread count, so the two
#       effects can be read together at the sizes the package is used on.
#
# Inputs (produced by pivot_benchmark.jl, once per thread count):
#   result/pivot_julia_t{N}.md

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scaling_plot import RESULT_DIR, parse_markdown_table, check_provenance

HEADER = ['n', 'Mode', 'Median (s)', 'Min (s)', 'Reps', 'Value', 'Delta', 'Iters', 'ArcScans']

# Three categorical hues from the same palette as the other figures. :serial is
# the package default and carries the EnergyFlow.jl blue used elsewhere; the two
# parallel rules take the remaining validated steps. Line style doubles the
# encoding so the rules stay separable without relying on colour.
STYLES = {
    'serial':         ('#0072B2', 'o', '-',  ':serial (block search, default)'),
    'parallel_block': ('#009E73', 's', '--', ':parallel_block (Kara & Özturan)'),
    'full_parallel':  ('#E69F00', '^', ':',  ':full_parallel (parallel Dantzig)'),
}
MODES = ['serial', 'parallel_block', 'full_parallel']


def load():
    """Return {threads: [row, ...]} across the thread-tagged result files."""
    out = {}
    if not os.path.isdir(RESULT_DIR):
        return out
    for name in sorted(os.listdir(RESULT_DIR)):
        m = re.fullmatch(r'pivot_julia_t(\d+)\.md', name)
        if not m:
            continue
        rows = []
        for row in parse_markdown_table(os.path.join(RESULT_DIR, name), HEADER):
            rows.append(dict(n=int(row['n']), mode=row['Mode'],
                             time_s=float(row['Median (s)']),
                             iters=int(row['Iters']),
                             arc_scans=int(row['ArcScans'])))
        if rows:
            out[int(m.group(1))] = rows
    return out


def series(rows, mode, xkey, ykey):
    pts = sorted(((r[xkey], r[ykey]) for r in rows if r['mode'] == mode))
    return [p[0] for p in pts], [p[1] for p in pts]


def plot_pivots(ax, rows):
    """(a) Pivot count against n — the algorithmic comparison."""
    for mode in MODES:
        xs, ys = series(rows, mode, 'n', 'iters')
        if not xs:
            continue
        color, marker, style, label = STYLES[mode]
        ax.plot(xs, ys, marker=marker, linestyle=style, color=color, label=label,
                linewidth=1.6, markersize=5)

    ax.set_xscale('log')
    ax.set_yscale('log')
    # Decade labels only; over a narrow range matplotlib also labels the minor
    # ticks, which collide at this panel width.
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel('Particles per event, $n$')
    ax.set_ylabel('Pivots to optimum')
    ax.set_title('(a) Entering-arc quality', loc='left', fontsize=10)
    ax.grid(True, which='major', alpha=0.3)
    ax.grid(True, which='minor', alpha=0.12)
    ax.legend(fontsize=7, frameon=False, loc='upper left')


def plot_threads(ax, by_threads, n):
    """(b) Wall time against thread count at fixed n."""
    threads = sorted(by_threads)
    for mode in MODES:
        xs, ys = [], []
        for t in threads:
            match = [r for r in by_threads[t] if r['mode'] == mode and r['n'] == n]
            if match:
                xs.append(t)
                ys.append(match[0]['time_s'] * 1e3)
        if not xs:
            continue
        color, marker, style, label = STYLES[mode]
        # :serial does not use threads at all, so its across-invocation spread is
        # measurement noise, not scaling. Drawn flat and unmarked to say so.
        if mode == 'serial':
            ax.plot(xs, ys, linestyle='-', color=color, linewidth=1.5,
                    label=label + ', thread-independent', alpha=0.9)
        else:
            ax.plot(xs, ys, marker=marker, linestyle=style, color=color,
                    label=label, linewidth=1.6, markersize=5)

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(threads)
    ax.set_xticklabels([str(t) for t in threads])
    ax.set_xlabel('Threads')
    ax.set_ylabel('Time per EMD (ms)')
    ax.set_title(f'(b) Cost of parallel pivoting, $n={n}$', loc='left', fontsize=10)
    ax.grid(True, which='major', alpha=0.3)
    # Say so when a rule is absent here rather than leaving a silent gap: at the
    # sizes where parallel pivoting pays off, :full_parallel is past its cap.
    missing = [STYLES[m][3] for m in MODES
               if not any(r['mode'] == m and r['n'] == n
                          for rs in by_threads.values() for r in rs)]
    if missing:
        ax.plot([], [], ' ', label=f'not run at $n={n}$: ' + ', '.join(missing))
    ax.legend(fontsize=7, frameon=False, loc='upper left')


def plot_sizes(ax, by_threads):
    """(c) Wall time against n, each rule at its own best thread count.

    Not at the highest thread count, which panel (b) shows is the wrong
    operating point: :parallel_block's cost is U-shaped in threads and rises
    again past its optimum, so plotting 48 threads would show the parallel rules
    at their worst and contradict the comparison a user would actually get by
    picking a sensible setting. Each rule is therefore shown at whatever thread
    count is fastest for it at each size — the honest best-vs-best comparison.
    """
    best_threads = {}
    for mode in MODES:
        xs, ys, chosen = [], [], []
        sizes = sorted({r['n'] for rs in by_threads.values() for r in rs
                        if r['mode'] == mode})
        for n in sizes:
            candidates = [(r['time_s'], t) for t, rs in by_threads.items()
                          for r in rs if r['mode'] == mode and r['n'] == n]
            if not candidates:
                continue
            t_best, thr = min(candidates)
            xs.append(n)
            ys.append(t_best * 1e3)
            chosen.append(thr)
        if not xs:
            continue
        best_threads[mode] = chosen
        color, marker, style, label = STYLES[mode]
        ax.plot(xs, ys, marker=marker, linestyle=style, color=color,
                label=label, linewidth=1.6, markersize=5)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel('Particles per event, $n$')
    ax.set_ylabel('Time per EMD (ms)')
    ax.set_title('(c) Wall time, each rule at its best thread count',
                 loc='left', fontsize=10)
    ax.grid(True, which='major', alpha=0.3)
    ax.grid(True, which='minor', alpha=0.12)
    ax.legend(fontsize=7, frameon=False, loc='upper left')
    return best_threads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'paper', 'pivot.png'))
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()

    by_threads = load()
    if not by_threads:
        sys.exit('No result/pivot_julia_t*.md found. Run pivot_benchmark.jl first '
                 '(see benchmark.md).')

    inputs = [os.path.join(RESULT_DIR, n) for n in os.listdir(RESULT_DIR)
              if re.fullmatch(r'pivot_julia_t\d+\.md', n)]
    check_provenance(inputs)

    threads = max(by_threads)
    rows = by_threads[threads]

    # Panel (b) needs a size present at every thread count; the largest such is
    # where parallel pivoting has the best chance of paying off.
    common = None
    for t, rs in by_threads.items():
        sizes = {r['n'] for r in rs if r['mode'] != 'serial'}
        common = sizes if common is None else (common & sizes)
    panel_b_n = max(common) if common else max(r['n'] for r in rows)

    npanels = 3 if len(by_threads) >= 2 else 2
    if npanels == 2:
        print('Only one thread count found; omitting the thread panel. Run '
              'pivot_benchmark.jl at several thread counts to include it.')

    fig, axes = plt.subplots(1, npanels, figsize=(4.6 * npanels, 3.6))
    plot_pivots(axes[0], rows)
    idx = 1
    if npanels == 3:
        plot_threads(axes[idx], by_threads, panel_b_n)
        idx += 1
    best_threads = plot_sizes(axes[idx], by_threads)

    fig.tight_layout()
    output = os.path.normpath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches='tight')
    print(f'Figure written to {output}')

    # The headline comparison, printed so it does not have to be read off a log
    # axis: does the Kara & Ozturan rule ever beat the default in wall time?
    best = {}
    for t, rs in by_threads.items():
        for r in rs:
            key = (r['n'], r['mode'])
            if key not in best or r['time_s'] < best[key][1]:
                best[key] = (t, r['time_s'])
    print('\nBest wall time over all thread counts (ms), by size:')
    print(f"{'n':>6}  {'serial':>12}  {'par_block':>12}  {'at thr':>7}  "
          f"{'speedup':>8}  {'pivot ratio':>11}")
    for n in sorted({n for n, _ in best}):
        s = best.get((n, 'serial'))
        p = best.get((n, 'parallel_block'))
        if not (s and p):
            continue
        si = next((r['iters'] for r in rows if r['n'] == n and r['mode'] == 'serial'), None)
        pi = next((r['iters'] for r in rows if r['n'] == n and r['mode'] == 'parallel_block'), None)
        ratio = f'{pi / si:.2f}' if si and pi else '-'
        print(f'{n:>6}  {s[1] * 1e3:>12.3f}  {p[1] * 1e3:>12.3f}  {p[0]:>7}  '
              f'{s[1] / p[1]:>7.2f}x  {ratio:>11}')
    print('speedup > 1 means :parallel_block is faster; pivot ratio < 1 means it '
          'reaches the optimum in fewer pivots.')
    print('"at thr" is the thread count where :parallel_block was fastest — its cost is '
          'U-shaped in threads, so this is not simply the largest available.')


if __name__ == '__main__':
    main()
