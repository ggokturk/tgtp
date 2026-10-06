// Scalar MT19937 baseline.
// Usage: rw_experiment <datasets_folder> <dataset> <depth> <n_walks> [threads]
// Every walk performs exactly depth hops; counters consume each reached vertex.
// Graph loading is excluded from t_rw_s; all walk computation is included.
// Build with Intel oneAPI C++20, -Ofast -march=native -qopenmp.

#include "graph.h"

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
static const char* const VARIANT = "regular";

// Regular baseline: n_walk independent walks, each with exactly len_walk hops.
// Consume every reached vertex, excluding the starting vertex.
static inline void scalar_randomwalk(uint32_t src, size_t len_walk,
                                     size_t n_walk, float* pvec,
                                     fuser::graph_t<uint32_t, uint32_t>& g,
                                     std::mt19937& rng)
{
  for (size_t i = 0; i < n_walk; i++) {
    uint32_t cur = src;
    for (size_t len = 0; len < len_walk; len++) {
      const uint32_t k = rng() % g.degree(cur);
      cur = g.begin(cur)[k];
      pvec[cur] += 1.0f;
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

  // ---- Regular random walk phase (timed) --------------------------------
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
        continue;            // isolated vertex -> no edges (avoid div by zero)

      rng.seed(42u + src_u); // matches source's per-source seeding

      if (do_walks)
        scalar_randomwalk(src_u, depth, n_walks, pvec.data(), g, rng);
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

  std::cout << "regular\t" << dataset << "\t" << depth << "\t" << n_walks << "\t"
            << threads << "\t" << n << "\t" << m << "\t" << t_load << "\t"
            << t_rw << "\t" << total << "\t" << total_walks << "\t"
            << walks_per_sec << "\t" << visit_count << "\t" << checksum << "\n";

  return 0;
}
