# SABA: sampling-aware vectorized random walks

Research code accompanying **Accelerating Spanning-Centrality Approximation
with Sampling-Aware Vectorized Random Walks**. SABA groups walks initialized on an
ordered uniform grid into SIMD-width bouquets and integrates this scheduling into TGT+ for
approximate edge spanning centrality.

## Build

Requires x86-64 Linux, CMake 3.16+, Intel oneAPI's C++20 compiler (`icpx`),
and OpenMP for the AESC programs. The primary implementation uses native
AVX2 and Intel SVML intrinsics; no SIMD compatibility shims are included.
The `edgeutils` dataset utility is compiler-agnostic: it builds with any
C++20 compiler and fetches Eigen, Spectra, cpp-httplib, and miniz at
configure time.

```sh
cmake -S . -B build/oneapi -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release
cmake --build build/oneapi -j
```

On Apple Silicon, enter an amd64 oneAPI container:

```sh
container run -it --rm --arch amd64 --rosetta --name oneapi-dev \
  -v "<TGTP PATH>" intel/oneapi:latest /bin/bash
```

Use the build commands above inside the container. Emulated execution is
useful for verification; collect publication timings on native hardware.
All timed implementations compile with the paper's `-Ofast -march=native`
flags. Record the compiler, CPU, and thread count with reported timings.

## Run spanning-centrality approximation

```sh
OMP_NUM_THREADS=1 ./build/oneapi/tgtp datasets Facebook 0.05 128 10 scores.bin
./build/oneapi/benchmark Facebook 0.05,0.01,0.005 1,2,4,8,16
./build/oneapi/benchmark_baseline Facebook 0.05,0.01,0.005 1
```

The runner arguments are `folder graph epsilon omega gamma outfile`.
Input files are `folder/graph/graph.txt` and
`folder/graph/sorted_eigens_<omega>.txt`. See
[`datasets/README.md`](datasets/README.md) for their formats. The benchmark
accepts `dataset [eps_csv] [threads_csv] [repeats] [data_root]
[omega] [gamma] [warmups]`. Defaults are three measured
repeats, `./datasets`, omega 128, gamma 10, and one warmup. Output is TSV
with a header; its final column is a double-precision sum of computed scores.
Timing includes `tgtp()` but excludes graph/eigenpair loading and the checksum.
`benchmark_baseline` accepts the same argument positions, runs with one
thread, and requires its thread argument to be `1` (the default).

## Run synthetic benchmarks

The three executables use the same arguments: dataset root, dataset name,
number of hops, walks per vertex, and optional thread count (default 1).
Each walk performs exactly the specified number of hops and increments a
counter for every reached vertex, excluding its starting vertex.

```sh
./build/oneapi/rw_experiment datasets Facebook 5 2048 16
./build/oneapi/rw_experiment_simd datasets Facebook 5 2048 16
./build/oneapi/rw_experiment_saba datasets Facebook 5 2048 16
```

Use hop counts 5, 10, and 15 and walk counts 2048 and 16384 for the paper
settings. SIMD and SABA require full eight-lane bouquets. The regular and
SIMD baselines draw from MT19937; SABA uses the same grid, vertex hash,
streaming update, and whole-bouquet SplitMix32 fallback as the AESC kernel.
SABA grid initialization and all visited-vertex counter updates are included
in `t_rw_s`; graph loading is excluded. Results are written to `results/rw/`
and printed as TSV rows with these columns:

```text
variant dataset depth n_walks threads n_vertices n_edges t_load_s t_rw_s total_s total_walks walks_per_sec visit_count checksum
```

The legacy fields `total_walks` and `walks_per_sec` count transitions and
transitions per second. `visit_count` sums the vertex counters; `checksum`
weights each counter by its vertex ID plus one. Both consume the outputs.
For comparable runs, choose the same compiler flags and OpenMP environment
for all variants and report the minimum of three measured repetitions.

## Validate accuracy

`exact_sc` computes reference scores, and `scripts/paper_accuracy.py`
compares saved AESC scores with a grounded-Laplacian reference. The Python
analysis and tests require NumPy and SciPy.

```sh
python3 scripts/paper_accuracy.py datasets/Facebook/graph.txt accuracy.tsv \
  --scores 0.05=scores.bin
```

## Repository and provenance

- `src/`: primary implementation, benchmarks, and preprocessing utility.
- `datasets/`: input format documentation and download URLs; preparation
  uses the C++ `edgeutils` executable.
- `paper/`: manuscript and figures.

The original AESC implementation and its reference-only benchmark wrappers
are not distributed here. Comparisons against original TGT+ require obtaining
and running that implementation separately; this repository's benchmark runs
only this project's implementation. Historical results are left locally in
`results/` and excluded from version control, as are downloaded datasets and
build products.

The repository retains its existing AGPL-3.0 license in [`LICENSE`](LICENSE).
The upstream AESC eigenpair-preparation provenance is retained in
`src/eigenprep.h`.
