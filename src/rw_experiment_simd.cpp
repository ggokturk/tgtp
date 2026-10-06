// Conventional AVX2 MT19937 baseline.
// Usage: rw_experiment_simd <datasets_folder> <dataset> <depth> <n_walks> [threads]
// Every walk performs exactly depth hops; counters consume each reached vertex.
// Graph loading is excluded from t_rw_s; all walk computation is included.
// Build with Intel oneAPI C++20, -Ofast -march=native -qopenmp.

#include "graph.h"

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;

// This executable's variant label. Embedded in the result filename so that
// "regular" and "simd" runs do not overwrite each other's output files.
static const char* const VARIANT = "simd";

// ---- AVX2 helper types (copied from src/aesc.h) ---------------------------
union alignas(32) avx2_t {
  __m256i i;
  uint32_t is[8];
  avx2_t() = default;
  avx2_t(__m256i rhs) : i(rhs) {}
};

// Conventional AVX2 baseline: full bouquets, exactly len_walk hops each.
// MT19937 draws are scalar; unsigned remainder uses Intel's SVML intrinsic.
static inline void simd_randomwalk(uint32_t src, size_t len_walk,
                                   size_t n_walk, float* pvec,
                                   fuser::graph_t<uint32_t, uint32_t>& g,
                                   std::mt19937& rng)
{
  const int BLOCKSIZE = 8;
  for (size_t ii = 0; ii < n_walk; ii += BLOCKSIZE) {
    avx2_t cur;
    cur.i = _mm256_set1_epi32((int)src);
    for (size_t len = 0; len < len_walk; len++) {
      avx2_t rnd, ds, k;
      for (int lane = 0; lane < BLOCKSIZE; ++lane)
        rnd.is[lane] = rng();
      ds = _mm256_set_epi32(
          (int)g.degree(cur.is[7]), (int)g.degree(cur.is[6]),
          (int)g.degree(cur.is[5]), (int)g.degree(cur.is[4]),
          (int)g.degree(cur.is[3]), (int)g.degree(cur.is[2]),
          (int)g.degree(cur.is[1]), (int)g.degree(cur.is[0]));
      k.i = _mm256_rem_epu32(rnd.i, ds.i);
      cur = _mm256_set_epi32(
          (int)g.begin(cur.is[7])[k.is[7]],
          (int)g.begin(cur.is[6])[k.is[6]],
          (int)g.begin(cur.is[5])[k.is[5]],
          (int)g.begin(cur.is[4])[k.is[4]],
          (int)g.begin(cur.is[3])[k.is[3]],
          (int)g.begin(cur.is[2])[k.is[2]],
          (int)g.begin(cur.is[1])[k.is[1]],
          (int)g.begin(cur.is[0])[k.is[0]]);
      pvec[cur.is[0]] += 1.0f;
      pvec[cur.is[1]] += 1.0f;
      pvec[cur.is[2]] += 1.0f;
      pvec[cur.is[3]] += 1.0f;
      pvec[cur.is[4]] += 1.0f;
      pvec[cur.is[5]] += 1.0f;
      pvec[cur.is[6]] += 1.0f;
      pvec[cur.is[7]] += 1.0f;
    }
  }
}

int
main(int argc, char** argv)
{
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0]
              << " <datasets_folder> <dataset> <depth> <n_walks> [threads]\n";
    return 1;
  }

  const std::string folder = argv[1];
  const std::string dataset = argv[2];
  const size_t depth = std::strtoull(argv[3], nullptr, 10);
  const size_t n_walks = std::strtoull(argv[4], nullptr, 10);
  if (n_walks % 8 != 0) {
    std::cerr << "n_walks must be a multiple of 8\n";
    return 1;
  }
  int threads = (argc > 5) ? std::atoi(argv[5]) : 1;
  if (threads < 1) threads = 1;

#ifdef _OPENMP
  omp_set_num_threads(threads);
#endif

  const fs::path graph_path = fs::path(folder) / dataset / "graph.txt";

  // ---- Load graph (timed) ------------------------------------------------
  fuser::graph_t<uint32_t, uint32_t> g;
  auto t0 = std::chrono::high_resolution_clock::now();
  g.load_txt(graph_path.string());
  auto t1 = std::chrono::high_resolution_clock::now();
  const double t_load = std::chrono::duration<double>(t1 - t0).count();

  const size_t n = g.n;
  const size_t m = g.m;

  // ---- SIMD random walk phase (timed) ------------------------------------
  auto t2 = std::chrono::high_resolution_clock::now();

  const bool do_walks = (n_walks > 0);

  double visit_count = 0.0, checksum = 0.0;

#pragma omp parallel reduction(+:visit_count,checksum)
  {
#ifdef _OPENMP
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    std::mt19937 rng(42u + (unsigned)tid);
    std::vector<float> pvec(n, 0.0f);

#pragma omp for schedule(dynamic, 1024)
    for (long long src = 0; src < (long long)n; ++src) {
      const uint32_t src_u = (uint32_t)src;
      if (g.degree(src_u) == 0)
        continue;

      rng.seed(42u + src_u); // matches source's per-source seeding

      if (do_walks)
        simd_randomwalk(src_u, depth, n_walks, pvec.data(), g, rng);
    }
    for (size_t vertex = 0; vertex < n; ++vertex) {
      visit_count += pvec[vertex];
      checksum += double(vertex + 1) * pvec[vertex];
    }
  }

  auto t3 = std::chrono::high_resolution_clock::now();
  const double t_rw = std::chrono::duration<double>(t3 - t2).count();
  const double total = std::chrono::duration<double>(t3 - t0).count();

  const double total_walks = (double)n * (double)n_walks * (double)depth;
  const double walks_per_sec = (t_rw > 0.0) ? (total_walks / t_rw) : 0.0;

  g.free();

  fs::create_directories("results/rw");
  const std::string resfile = "results/rw/" + std::string(VARIANT) + "_" +
                              dataset + "_d" + std::to_string(depth) + "_w" +
                              std::to_string(n_walks) + ".txt";
  {
    std::ofstream out(resfile);
    out << "variant " << VARIANT << "\n";
    out << "dataset " << dataset << "\n";
    out << "depth " << depth << "\n";
    out << "n_walks " << n_walks << "\n";
    out << "threads " << threads << "\n";
    out << "n_vertices " << n << "\n";
    out << "n_edges " << m << "\n";
    out << "t_load_s " << t_load << "\n";
    out << "t_rw_s " << t_rw << "\n";
    out << "total_s " << total << "\n";
    out << "total_walks " << total_walks << "\n";
    out << "walks_per_sec " << walks_per_sec << "\n";
    out << "visit_count " << visit_count << "\n";
    out << "checksum " << checksum << "\n";
  }

  std::cout << "simd\t" << dataset << "\t" << depth << "\t" << n_walks << "\t"
            << threads << "\t" << n << "\t" << m << "\t" << t_load << "\t"
            << t_rw << "\t" << total << "\t" << total_walks << "\t"
            << walks_per_sec << "\t" << visit_count << "\t" << checksum << "\n";

  return 0;
}
