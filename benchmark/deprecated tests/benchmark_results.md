# EnergyFlow.jl Backend Benchmark Results

Date: 2026-04-08T05:25:15.912
Julia: 1.12.5, Threads: 1

## Parameters: R=1.0, beta=1.0, norm=true

| n | Backend | Median (µs) | Min (µs) | Max (µs) |
|---|---------|-------------|----------|----------|
| 2 | ns64 | 1.1 | 0.5 | 5.4 |
| 2 | ot64 | 0.6 | 0.5 | 2.4 |
| 2 | ns32 | 0.8 | 0.6 | 6.6 |
| 2 | ot32 | 0.7 | 0.6 | 2.2 |
| 100 | ns64 | 282.8 | 265.2 | 415.7 |
| 100 | ot64 | 283.3 | 248.0 | 521.1 |
| 100 | ns32 | 256.8 | 245.9 | 334.3 |
| 100 | ot32 | 284.4 | 264.3 | 481.7 |
| 1000 | ns64 | 36823.6 | 33779.5 | 39209.5 |
| 1000 | ot64 | 51429.6 | 48724.6 | 56791.8 |
| 1000 | ns32 | 32517.7 | 31022.6 | 38254.1 |
| 1000 | ot32 | 44143.2 | 41877.9 | 45849.8 |
| 2000 | ns64 | 194141.6 | 185516.3 | 199860.5 |
| 2000 | ot64 | 254163.6 | 244908.5 | 261858.8 |
| 2000 | ns32 | 198894.9 | 193767.9 | 204783.9 |
| 2000 | ot32 | 228920.7 | 223618.7 | 242304.8 |

## Notes
- Float32 backends (ns32, ot32) use the same NetworkSimplex algorithm with Float32 arithmetic
- Arc mixing (ot variants) uses POT-style strided arc interleaving
- Benchmark uses allocating API (fresh workspace per call)
