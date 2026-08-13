# Build the Sinkhorn accuracy/cost figure for the JOSS paper.
# Usage: cd benchmark && python sinkhorn_plot.py [--output ../paper/sinkhorn.png]
#
# The exact backends can be judged on time alone; the approximate one cannot, so
# this figure never shows a Sinkhorn timing without the error of the value that
# timing bought. Three panels:
#
#   (a) Accuracy against the regularisation strength ε. The entropic bias falls
#       roughly linearly in ε, and EnergyFlow.jl's values sit on top of POT's —
#       the two implementations of the same algorithm agree to ~8 significant
#       figures, which is what makes the timing comparison in (b) and (c)
#       meaningful rather than a comparison of two different approximations.
#
#   (b) What that accuracy costs, at fixed multiplicity: wall time against the
#       error achieved, with both exact solvers as reference lines. This is the
#       panel that answers "should I use the approximate backend?" — at collider
#       multiplicities the exact network simplex is both faster and exact, and
#       the figure should say so rather than imply a tradeoff that is not there.
#
#   (c) How both scale with multiplicity at fixed ε, against the exact solvers.
#
# Inputs (produced by the benchmark scripts):
#   result/sinkhorn_julia.md    sinkhorn_benchmark.jl
#   result/sinkhorn_python.md   sinkhorn_benchmark_python.py
#
# The POT curves are optional: the figure builds from the Julia file alone, with
# the POT comparison omitted and a note printed.

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Same table parsing and provenance checking as the scaling figure — the result
# files share a format, and the "were these produced on one machine" question
# applies here for the same reason.
from scaling_plot import RESULT_DIR, parse_markdown_table, check_provenance

HEADER = ['n', 'Epsilon', 'Backend', 'Median (s)', 'Min (s)', 'Reps',
          'Value', 'RelErr', 'Iters', 'Converged']

# Two categorical hues only, carried from the scaling figure so the palette
# reads as one system across the paper: EnergyFlow.jl is blue, POT is vermillion
# in both. Everything else — multiplicity, ε, and exact-vs-approximate — is
# encoded by line style and marker, never by a third hue, so nothing depends on
# colour discrimination alone and the exact reference stays visibly attached to
# the implementation it came from. Checked for colour-vision separation rather
# than assumed: the pair is ΔE 31 in normal vision and ≥ 24 under protanopia,
# deuteranopia and tritanopia.
JULIA_C = '#0072B2'
POT_C = '#D55E00'

# Line style per multiplicity, in the order the sizes appear.
N_STYLES = ['-', '--', ':', '-.']

# Panel (b) is drawn at one multiplicity to stay legible; the middle of the
# ε-sweep sizes unless overridden.
PANEL_B_N = int(os.environ['ENERGYFLOW_SINKHORN_PLOT_N']) \
    if 'ENERGYFLOW_SINKHORN_PLOT_N' in os.environ else None


def load(path):
    """Return the result rows of one sinkhorn benchmark file as typed dicts."""
    rows = []
    for row in parse_markdown_table(path, HEADER):
        rows.append(dict(
            n=int(row['n']),
            # The exact rows carry 'exact' rather than a number in this column.
            epsilon=None if row['Epsilon'] == 'exact' else float(row['Epsilon']),
            backend=row['Backend'],
            time_s=float(row['Median (s)']),
            value=float(row['Value']),
            relerr=float(row['RelErr']),
            iters=int(row['Iters']),
            converged=row['Converged'].strip().lower() == 'true',
        ))
    return rows


def select(rows, backend, n=None):
    """Rows for one backend, optionally one multiplicity, sorted by ε."""
    out = [r for r in rows if r['backend'] == backend and (n is None or r['n'] == n)]
    return sorted(out, key=lambda r: (r['n'], r['epsilon'] if r['epsilon'] is not None else 0.0))


def exact_time(rows, backend, n):
    """Wall time of the exact solver at multiplicity n, or None."""
    for r in rows:
        if r['backend'] == backend and r['n'] == n:
            return r['time_s']
    return None


def _split_converged(points, xkey, ykey):
    """Split points into (converged, unconverged) x/y lists.

    Unconverged points are drawn hollow. Their error is not the entropic bias
    the panel is about — it is an iteration-cap artifact, and it is biased low,
    so plotting it as an ordinary accuracy measurement would misreport the
    method. Hollow markers keep them visible without asserting them.
    """
    ok_x = [p[xkey] for p in points if p['converged']]
    ok_y = [p[ykey] for p in points if p['converged']]
    no_x = [p[xkey] for p in points if not p['converged']]
    no_y = [p[ykey] for p in points if not p['converged']]
    return (ok_x, ok_y), (no_x, no_y)


def plot_accuracy(ax, julia, pot, sizes):
    """(a) Relative error against ε, both implementations."""
    for style, n in zip(N_STYLES, sizes):
        pts = [dict(p, abserr=abs(p['relerr']))
               for p in select(julia, 'sinkhorn', n)]
        if not pts:
            continue
        ax.plot([p['epsilon'] for p in pts], [p['abserr'] for p in pts],
                linestyle=style, color=JULIA_C, linewidth=1.6, zorder=3)
        (ok_x, ok_y), (no_x, no_y) = _split_converged(pts, 'epsilon', 'abserr')
        ax.plot(ok_x, ok_y, 'o', color=JULIA_C, markersize=5, zorder=4)
        ax.plot(no_x, no_y, 'o', markerfacecolor='white', markeredgecolor=JULIA_C,
                markersize=5, markeredgewidth=1.3, zorder=4)
        # Direct label instead of a third legend entry per multiplicity.
        last = pts[-1]
        ax.annotate(f'$n={n}$', (last['epsilon'], last['abserr']),
                    textcoords='offset points', xytext=(4, -9),
                    fontsize=7.5, color=JULIA_C)

        pot_pts = [dict(p, abserr=abs(p['relerr']))
                   for p in select(pot, 'POT_sinkhorn', n)]
        if pot_pts:
            ax.plot([p['epsilon'] for p in pot_pts], [p['abserr'] for p in pot_pts],
                    linestyle='none', marker='D', markerfacecolor='none',
                    markeredgecolor=POT_C, markersize=6, markeredgewidth=1.1, zorder=5)

    ax.set_xscale('log')
    ax.set_yscale('log')
    # Room on the right for the direct labels, which sit on the last point.
    left, right = ax.get_xlim()
    ax.set_xlim(left, right * 1.8)
    ax.set_xlabel('Regularisation strength, $\\varepsilon$')
    ax.set_ylabel('$|$relative error$|$ vs exact')
    ax.set_title('(a) Sinkhorn accuracy', loc='left', fontsize=10)
    ax.grid(True, which='major', alpha=0.3)
    ax.grid(True, which='minor', alpha=0.12)
    handles = [
        Line2D([], [], color=JULIA_C, marker='o', markersize=5, linewidth=1.6,
               label='EnergyFlow.jl :sinkhorn'),
        Line2D([], [], color=POT_C, marker='D', markerfacecolor='none', linestyle='none',
               markersize=6, label='POT sinkhorn_log'),
        Line2D([], [], color=JULIA_C, marker='o', markerfacecolor='white', linestyle='none',
               markersize=5, label='hit iteration cap'),
    ]
    ax.legend(handles=handles, fontsize=7.5, frameon=False, loc='lower right')


def plot_tradeoff(ax, julia, pot, n):
    """(b) Wall time against the accuracy achieved, at one multiplicity."""
    for rows, backend, color, marker, label in (
            (julia, 'sinkhorn', JULIA_C, 'o', 'EnergyFlow.jl :sinkhorn'),
            (pot, 'POT_sinkhorn', POT_C, 'D', 'POT sinkhorn_log')):
        # In ε order, not error order. The two coincide while Sinkhorn behaves —
        # smaller ε costs more and buys accuracy — so a curve that visibly
        # doubles back is a point where spending more did not buy more, which is
        # exactly what the panel should not hide by re-sorting.
        pts = [dict(p, abserr=abs(p['relerr']), time_ms=p['time_s'] * 1e3)
               for p in select(rows, backend, n)]
        if not pts:
            continue
        ax.plot([p['abserr'] for p in pts], [p['time_ms'] for p in pts],
                linestyle='-', color=color, linewidth=1.6, label=label, zorder=3)
        (ok_x, ok_y), (no_x, no_y) = _split_converged(pts, 'abserr', 'time_ms')
        ax.plot(ok_x, ok_y, marker, color=color, markersize=5, linestyle='none', zorder=4)
        ax.plot(no_x, no_y, marker, markerfacecolor='white', markeredgecolor=color,
                markersize=5, markeredgewidth=1.3, linestyle='none', zorder=4)

    # The exact solvers, which reach zero error and so cannot be drawn as points
    # on a log error axis — they are the floor the tradeoff is measured against.
    for rows, backend, color, style, label in (
            (julia, 'ns64', JULIA_C, ':', 'EnergyFlow.jl :ns64 (exact)'),
            (pot, 'POT_exact', POT_C, ':', 'POT ot.lp.emd2 (exact)')):
        t = exact_time(rows, backend, n)
        if t is not None:
            ax.axhline(t * 1e3, linestyle=style, color=color, linewidth=1.3,
                       alpha=0.9, label=label, zorder=2)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.invert_xaxis()                     # more accurate to the right
    ax.set_xlabel('$|$relative error$|$  (more accurate $\\rightarrow$)')
    ax.set_ylabel('Time per EMD (ms)')
    ax.set_title(f'(b) Cost of accuracy, $n={n}$', loc='left', fontsize=10)
    ax.grid(True, which='major', alpha=0.3)
    ax.grid(True, which='minor', alpha=0.12)
    # Upper left: the curves climb to the right on this inverted axis, so the
    # high-time/high-error corner is the empty one.
    ax.legend(fontsize=7.5, frameon=False, loc='upper left')


def plot_scaling(ax, julia, pot, epsilons):
    """(c) Wall time against multiplicity, at fixed ε, against the exact solvers.

    POT is drawn at the tightest ε only. Panel (b) already establishes the gap
    between the two Sinkhorn implementations at every ε; repeating it here for
    each ε would double the legend to say the same thing, and the panel's job is
    the shape of the curves against n.
    """
    pot_eps = min(epsilons) if epsilons else None
    for style, eps in zip(N_STYLES, epsilons):
        pts = sorted((dict(p, time_ms=p['time_s'] * 1e3)
                      for p in julia if p['backend'] == 'sinkhorn' and p['epsilon'] == eps),
                     key=lambda p: p['n'])
        if pts:
            ax.plot([p['n'] for p in pts], [p['time_ms'] for p in pts],
                    linestyle=style, color=JULIA_C, linewidth=1.6, zorder=3,
                    label=f'EnergyFlow.jl :sinkhorn, $\\varepsilon={eps:g}$')
            (ok_x, ok_y), (no_x, no_y) = _split_converged(pts, 'n', 'time_ms')
            ax.plot(ok_x, ok_y, 'o', color=JULIA_C, markersize=5, zorder=4)
            ax.plot(no_x, no_y, 'o', markerfacecolor='white', markeredgecolor=JULIA_C,
                    markersize=5, markeredgewidth=1.3, zorder=4)

        if eps != pot_eps:
            continue
        pot_pts = sorted((dict(p, time_ms=p['time_s'] * 1e3)
                          for p in pot if p['backend'] == 'POT_sinkhorn' and p['epsilon'] == eps),
                         key=lambda p: p['n'])
        if pot_pts:
            ax.plot([p['n'] for p in pot_pts], [p['time_ms'] for p in pot_pts],
                    linestyle=style, color=POT_C, linewidth=1.6, marker='D',
                    markersize=5, zorder=3,
                    label=f'POT sinkhorn_log, $\\varepsilon={eps:g}$')

    # Exact reference curves.
    for rows, backend, color, style, marker, label in (
            (julia, 'ns64', JULIA_C, ':', 's', 'EnergyFlow.jl :ns64 (exact)'),
            (pot, 'POT_exact', POT_C, ':', 'v', 'POT ot.lp.emd2 (exact)')):
        pts = sorted((r for r in rows if r['backend'] == backend), key=lambda r: r['n'])
        if pts:
            ax.plot([p['n'] for p in pts], [p['time_s'] * 1e3 for p in pts],
                    linestyle=style, color=color, linewidth=1.4, marker=marker,
                    markersize=4, label=label, zorder=2)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Particles per event, $n$')
    ax.set_ylabel('Time per EMD (ms)')
    ax.set_title('(c) Scaling at fixed $\\varepsilon$', loc='left', fontsize=10)
    ax.grid(True, which='major', alpha=0.3)
    ax.grid(True, which='minor', alpha=0.12)
    # Lower right: every curve rises with n, so the fast/large-n corner is free.
    ax.legend(fontsize=6.8, frameon=False, loc='lower right')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'paper', 'sinkhorn.png'))
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()

    julia_path = os.path.join(RESULT_DIR, 'sinkhorn_julia.md')
    pot_path = os.path.join(RESULT_DIR, 'sinkhorn_python.md')
    check_provenance([julia_path, pot_path])

    julia = load(julia_path)
    pot = load(pot_path)
    if not julia:
        sys.exit('No Sinkhorn results found. Run sinkhorn_benchmark.jl first '
                 '(see benchmark.md).')
    if not pot:
        print('No result/sinkhorn_python.md found: the figure will show '
              'EnergyFlow.jl only. Run sinkhorn_benchmark_python.py to include '
              'the POT comparison, which is what shows the two implementations '
              'agree on the values being timed.')

    # The ε sweep runs at the multiplicities with the most ε points; the
    # scaling panel uses the ε values measured at the most multiplicities. Both
    # are derived from the data rather than hardcoded, so overriding the sweep
    # in the benchmark scripts does not silently produce an empty panel.
    eps_counts = {}
    n_counts = {}
    for r in julia:
        if r['backend'] != 'sinkhorn':
            continue
        eps_counts[r['n']] = eps_counts.get(r['n'], 0) + 1
        n_counts[r['epsilon']] = n_counts.get(r['epsilon'], 0) + 1
    sweep_sizes = sorted(n for n, c in eps_counts.items() if c == max(eps_counts.values()))
    scaling_eps = sorted((e for e, c in n_counts.items() if c == max(n_counts.values())),
                         reverse=True)

    panel_b_n = PANEL_B_N if PANEL_B_N is not None else sweep_sizes[len(sweep_sizes) // 2]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.6))
    plot_accuracy(axes[0], julia, pot, sweep_sizes)
    plot_tradeoff(axes[1], julia, pot, panel_b_n)
    plot_scaling(axes[2], julia, pot, scaling_eps)

    fig.tight_layout()
    output = os.path.normpath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches='tight')
    print(f'Figure written to {output}')

    capped = [r for r in julia if r['backend'] == 'sinkhorn' and not r['converged']]
    if capped:
        print(f'{len(capped)} Julia Sinkhorn point(s) hit the iteration cap and are drawn '
              'hollow; their error is a truncation artifact, not the entropic bias.')


if __name__ == '__main__':
    main()
