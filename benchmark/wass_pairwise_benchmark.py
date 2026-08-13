# Threaded pairwise EMD benchmark — wasserstein.PairwiseEMD
# Usage:
#   python wass_pairwise_benchmark.py --threads 8
#   python wass_pairwise_benchmark.py --threads 48 --sample result/large_sample.csv
#
# This exists to make the pairwise comparison fair. `ot.lp.emd2` is
# single-threaded and is called from a Python loop, so comparing an N-thread
# Julia `emds` against it measures Julia's threading against Python having none.
# The `wasserstein` library — the compiled backend behind Python EnergyFlow —
# has a genuinely multithreaded pairwise driver (OpenMP, `num_threads`), so
# running it at the same thread count as Julia is the like-for-like test:
# two compiled network-simplex implementations, equal parallelism, same events.
#
# Writes result/wass_pairwise_t{N}.md, matching the layout of
# result/emds_julia_t{N}.md so scaling_plot.py can overlay them.
#
# With --sample, instead runs one large self-pairwise matrix over the resampled
# event file written by make_large_sample.jl, for the large-matrix table.

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from envinfo import print_env, write_env_block
from emds_benchmark_python import load_hepmc3

REPS = 3
# Matches LONG_SOLVE_SECONDS in pairs_scaling_benchmark.jl, so both halves of
# the size sweep switch to single-run timing at the same point.
LONG_SOLVE_SECONDS = 30.0


def load_sample_csv(path):
    """Read the shared resampled event file (event_id, weight, y, phi).

    Both languages read the same file so they solve byte-identical problems;
    generating the sample independently on each side would not guarantee that.
    """
    raw = np.loadtxt(path, delimiter=',', skiprows=1)
    events = []
    for eid in np.unique(raw[:, 0]):
        rows = raw[raw[:, 0] == eid][:, 1:]
        events.append(np.ascontiguousarray(rows, dtype=np.float64))
    return events


def time_pairwise(events_a, events_b, norm, threads, long_solve_seconds=None):
    """Median wall time of a full pairwise EMD via wasserstein, in seconds.

    `verbose=0` suppresses the per-EMD progress output, which would otherwise
    be tens of thousands of lines in a batch log.

    `long_solve_seconds` enables the adaptive path used by the size sweep: past
    that pilot time, the pilot is reported instead of running REPS repetitions,
    because a multi-million-pair matrix takes minutes to reproduce a number that
    is not timer-noise limited. Left as None — the default — the function keeps
    its original fixed-REPS behaviour, so the split and large-matrix benchmarks
    that already published numbers are measured exactly as before.
    """
    import wasserstein

    def once():
        pw = wasserstein.PairwiseEMD(R=1.0, beta=1.0, norm=norm,
                                     num_threads=threads, verbose=0)
        if events_b is None:
            pw(events_a)
        else:
            pw(events_a, events_b)
        return np.asarray(pw.emds())

    once()                                    # warm up: first call sets up threads
    if long_solve_seconds is not None:
        start = time.perf_counter()
        once()
        est = time.perf_counter() - start
        if est > long_solve_seconds:
            return est, est
        times = [est]
        for _ in range(REPS - 1):
            start = time.perf_counter()
            once()
            times.append(time.perf_counter() - start)
        times.sort()
        return times[len(times) // 2], min(times)

    times = []
    for _ in range(REPS):
        start = time.perf_counter()
        once()
        times.append(time.perf_counter() - start)
    times.sort()
    return times[len(times) // 2], min(times)


def run_splits(threads):
    """The same 10v90 / 50v50 splits and setups as emds_benchmark.jl."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                        'data', 'sk_example_PU.hepmc')
    events = load_hepmc3(path, 100)
    events = [np.ascontiguousarray(e, dtype=np.float64) for e in events]
    print(f'Loaded {len(events)} events')

    # wasserstein's own Euclidean ground distance on the (y, phi) columns
    # matches EnergyFlow.jl's EuclideanMetric; the EtaPhi (periodic) setup is
    # left out because it would need a different measure on this side and the
    # figure only uses Euclidean_norm.
    setups = [('Euclidean_norm', True), ('Euclidean_unnorm', False)]
    splits = [('10v90', slice(0, 10), slice(10, 100)),
              ('50v50', slice(0, 50), slice(50, 100))]

    results = []
    for split_name, sa, sb in splits:
        ea, eb = events[sa], events[sb]
        for setup_name, norm in setups:
            med, mn = time_pairwise(ea, eb, norm, threads)
            results.append(dict(split=split_name, setup=setup_name,
                                backend='Wass', time_s=med, min_s=mn,
                                pairs=len(ea) * len(eb)))
            print(f'{split_name:6s} {setup_name:<18s} Wass  {med:8.2f} s '
                  f'(min {mn:6.2f} s, {len(ea) * len(eb)} pairs, {threads} threads)')
    return results


def run_large(sample_path, threads):
    """One large self-pairwise matrix over the shared resampled sample."""
    events = load_sample_csv(sample_path)
    n = len(events)
    pairs = n * (n - 1) // 2
    print(f'Loaded {n} events from {sample_path} ({pairs} unique pairs)')

    results = []
    for setup_name, norm in [('Euclidean_norm', True), ('Euclidean_unnorm', False)]:
        med, mn = time_pairwise(events, None, norm, threads)
        results.append(dict(split=f'{n}x{n}', setup=setup_name, backend='Wass',
                            time_s=med, min_s=mn, pairs=pairs))
        print(f'{n}x{n} {setup_name:<18s} Wass  {med:8.2f} s '
              f'(min {mn:6.2f} s, {pairs} pairs, {threads} threads)')
    return results


def run_sizes(sample_path, sizes, threads):
    """Self-pairwise matrices over increasing prefixes of the sample.

    The Python half of pairs_scaling_benchmark.jl. Prefixes of one file rather
    than separately generated samples, so the size-N run here and the size-N run
    in Julia solve byte-identical problems; make_large_sample.jl draws from a
    seeded stream in order, which is what makes prefixes well defined.

    Only the normalized setup is swept: it is the one the figure plots, and the
    unnormalized variant would double a sweep whose largest points take minutes.
    """
    events = load_sample_csv(sample_path)
    available = len(events)
    usable = [n for n in sizes if n <= available]
    if len(usable) < len(sizes):
        print(f'Sample holds {available} events; skipping sizes '
              f'{[n for n in sizes if n > available]}.')
    if not usable:
        sys.exit(f'No requested size fits in the {available}-event sample.')
    print(f'Loaded {available} events from {sample_path}; sweeping {usable}')

    results = []
    for n in usable:
        subset = events[:n]
        pairs = n * (n - 1) // 2
        med, mn = time_pairwise(subset, None, True, threads,
                                long_solve_seconds=LONG_SOLVE_SECONDS)
        results.append(dict(split=f'{n}x{n}', setup='Euclidean_norm', backend='Wass',
                            time_s=med, min_s=mn, pairs=pairs))
        print(f'{n}x{n} Euclidean_norm     Wass  {med:9.3f} s '
              f'(min {mn:9.3f} s, {pairs} pairs, {pairs / med:8.0f} pairs/s, '
              f'{threads} threads)', flush=True)
    return results


def write_results(path, title, results, threads):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as handle:
        handle.write(f'# {title}\n\n')
        handle.write(f'wasserstein.PairwiseEMD with num_threads={threads}. '
                     f'Time is the median over {REPS} repetitions after a warmup.\n\n')
        write_env_block(handle)
        handle.write('| Split | Setup | Backend | Time (s) | Pairs |\n')
        handle.write('|---|---|---|---|---|\n')
        for r in results:
            handle.write(f"| {r['split']} | {r['setup']} | {r['backend']} | "
                         f"{r['time_s']:.2f} | {r['pairs']} |\n")
    print(f'\nResults saved to {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, required=True,
                        help='OpenMP threads for wasserstein (match the Julia run)')
    parser.add_argument('--sample', default=None,
                        help='resampled event CSV; switches to the large-matrix run')
    parser.add_argument('--sizes', default=None,
                        help='comma-separated event counts; with --sample, sweeps '
                             'self-pairwise matrices over prefixes of it instead of '
                             'running the single large matrix')
    args = parser.parse_args()

    if args.sizes and not args.sample:
        parser.error('--sizes needs --sample to take prefixes of.')

    print_env()
    try:
        import wasserstein  # noqa: F401
    except Exception as exc:
        sys.exit(f'wasserstein unavailable ({exc.__class__.__name__}): '
                 'this benchmark needs it for the threaded comparison.')

    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')
    if args.sizes:
        sizes = [int(s) for s in args.sizes.split(',')]
        results = run_sizes(args.sample, sizes, args.threads)
        write_results(os.path.join(result_dir, f'pairs_scaling_wass_t{args.threads}.md'),
                      'Pairwise EMD Scaling in Pair Count — wasserstein',
                      results, args.threads)
    elif args.sample:
        results = run_large(args.sample, args.threads)
        write_results(os.path.join(result_dir, 'large_pairwise_python.md'),
                      'Large Pairwise EMD — wasserstein', results, args.threads)
    else:
        results = run_splits(args.threads)
        write_results(os.path.join(result_dir, f'wass_pairwise_t{args.threads}.md'),
                      'Pairwise EMD — wasserstein (threaded)', results, args.threads)


if __name__ == '__main__':
    main()
