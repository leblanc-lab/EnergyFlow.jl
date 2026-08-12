# Environment capture for EnergyFlow.jl benchmarks (Python side).
#
# Mirrors envinfo.jl so that the Julia and Python result files carry the same
# provenance fields and can be compared meaningfully. Every Python benchmark
# imports this and calls write_env_block(f) when writing its result table.

import os
import platform
import subprocess
import sys
from datetime import datetime

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def _probe(fn, fallback='unknown'):
    """Best-effort metadata probe: never let missing provenance fail a run."""
    try:
        value = fn()
        if value is None:
            return fallback
        text = str(value).strip()
        return text if text else fallback
    except Exception:
        return fallback


def _cpu_model():
    """Human-readable CPU model. platform.processor() is empty on most Linux,
    so read /proc/cpuinfo there and sysctl on macOS."""
    if sys.platform == 'darwin':
        return _probe(lambda: subprocess.check_output(
            ['sysctl', '-n', 'machdep.cpu.brand_string'], text=True))
    if sys.platform.startswith('linux'):
        def from_proc():
            with open('/proc/cpuinfo') as handle:
                for line in handle:
                    if line.startswith('model name'):
                        return line.split(':', 1)[1]
            return None
        return _probe(from_proc)
    return _probe(platform.processor)


def _git_describe():
    commit = _probe(lambda: subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD'], cwd=_REPO_ROOT, text=True,
        stderr=subprocess.DEVNULL))
    dirty = _probe(lambda: subprocess.check_output(
        ['git', 'status', '--porcelain'], cwd=_REPO_ROOT, text=True,
        stderr=subprocess.DEVNULL), fallback=None)
    if dirty:
        commit += ' (modified)'
    return commit


def _package_version(name):
    def lookup():
        from importlib.metadata import version
        return version(name)
    return _probe(lookup, fallback='not installed')


def env_pairs():
    """Environment description as a list of (field, value) pairs."""
    pairs = [
        ('Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
        ('CPU', _cpu_model()),
        ('CPU threads', _probe(lambda: os.cpu_count())),
        ('OS', f'{platform.system()} {platform.release()} ({platform.machine()})'),
        ('Python', platform.python_version()),
        ('numpy', _package_version('numpy')),
        ('POT', _package_version('pot')),
        ('wasserstein', _package_version('wasserstein')),
        ('energyflow', _package_version('energyflow')),
        ('Commit', _git_describe()),
    ]
    for var, label in (('SLURM_JOB_ID', 'SLURM job'),
                       ('SLURM_JOB_NODELIST', 'SLURM node'),
                       ('SLURM_CPUS_ON_NODE', 'SLURM CPUs')):
        if var in os.environ:
            pairs.append((label, os.environ[var]))
    return pairs


def write_env_block(handle):
    """Write the environment as a markdown table into an open result file."""
    handle.write('## Environment\n\n')
    handle.write('| Field | Value |\n')
    handle.write('|---|---|\n')
    for key, value in env_pairs():
        handle.write(f'| {key} | {value} |\n')
    handle.write('\n')


def print_env():
    """Echo provenance to stdout so job logs match the result files."""
    for key, value in env_pairs():
        print(f'{key:<14} {value}')
    print()
