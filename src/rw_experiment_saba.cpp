// Ordered uniform-grid AVX2 streaming with SplitMix32 fallback.
// Usage: rw_experiment_saba <datasets_folder> <dataset> <depth> <n_walks> [threads]
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

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;

// This executable's variant label. Embedded in the result filename so that
// "regular", "simd" and "saba" runs do not overwrite each other.
static const char* const VARIANT = "saba";

// ---- AVX2 helper types (copied from src/aesc.h) ---------------------------
union alignas(32) avx2_t {
  __m256i i;
  uint32_t is[8];
  avx2_t() = default;
  avx2_t(__m256i rhs) : i(rhs) {}
};

// MurmurHash3 x86_32 finalizer (fmix32), vectorized lane-wise over 8 walks.
// Bit-exact equivalent of calling the scalar fmix32 on each of the 8 lanes:
// the ops (xor / shift / 32-bit multiply) map 1:1 onto AVX2 integer intrinsics.
static inline __m256i murmur3_avx2(__m256i h) {
  h = _mm256_xor_si256(h, _mm256_srli_epi32(h, 16));
  h = _mm256_mullo_epi32(h, _mm256_set1_epi32((int)0x85ebca6bU));
  h = _mm256_xor_si256(h, _mm256_srli_epi32(h, 13));
  h = _mm256_mullo_epi32(h, _mm256_set1_epi32((int)0xc2b2ae35U));
  h = _mm256_xor_si256(h, _mm256_srli_epi32(h, 16));
  return h;
}

// The same eight-lane SplitMix32 mixer used by the AESC fallback.
static inline __m256i splitmix32_avx2(__m256i& state) {
  state = _mm256_add_epi32(state, _mm256_set1_epi32(0x9e3779b9));
  __m256i z = _mm256_xor_si256(state, _mm256_srli_epi32(state, 16));
  z = _mm256_mullo_epi32(z, _mm256_set1_epi32(0x21f0aaad));
  z = _mm256_xor_si256(z, _mm256_srli_epi32(z, 15));
  z = _mm256_mullo_epi32(z, _mm256_set1_epi32(0x735a2d97));
  return _mm256_xor_si256(z, _mm256_srli_epi32(z, 15));
}

// SABA: uniform grid, ordered bouquets, and whole-bouquet fallback.
// Initialization is inside the timed kernel; visits are consumed at each hop.
static inline void saba_randomwalk(uint32_t src, size_t len_walk,
                                   size_t n_walk, float* pvec,
                                   fuser::graph_t<uint32_t, uint32_t>& g)
{
  if (n_walk == 0) [[unlikely]] return;
  const int BLOCKSIZE = 8;
  const uint32_t grid_step = UINT32_MAX / static_cast<uint32_t>(n_walk);
  for (size_t ii = 0; ii < n_walk; ii += BLOCKSIZE) {
    avx2_t cur;
    cur.i = _mm256_set1_epi32((int)src);
    __m256i secret = _mm256_mullo_epi32(
        _mm256_add_epi32(_mm256_set1_epi32(static_cast<int>(ii)),
                         _mm256_set_epi32(8, 7, 6, 5, 4, 3, 2, 1)),
        _mm256_set1_epi32(static_cast<int>(grid_step)));
    __m256i random_state = secret;

    size_t len = 0;
    for (; len < len_walk; len++) {
      const __m256i cmp = _mm256_cmpeq_epi32(secret, _mm256_setzero_si256());
      const int mask = _mm256_movemask_epi8(cmp);
      if (mask != 0) [[unlikely]] break;
      avx2_t rnd, ds, k;
      ds = _mm256_set_epi32(
          (int)g.degree(cur.is[7]), (int)g.degree(cur.is[6]),
          (int)g.degree(cur.is[5]), (int)g.degree(cur.is[4]),
          (int)g.degree(cur.is[3]), (int)g.degree(cur.is[2]),
          (int)g.degree(cur.is[1]), (int)g.degree(cur.is[0]));
      rnd.i = _mm256_xor_si256(secret, murmur3_avx2(cur.i));
      const __m256i even = _mm256_mul_epu32(rnd.i, ds.i);
      const __m256i odd = _mm256_mul_epu32(
          _mm256_srli_epi64(rnd.i, 32), _mm256_srli_epi64(ds.i, 32));
      k.i = _mm256_blend_epi32(_mm256_srli_epi64(even, 32), odd, 0xaa);
      cur = _mm256_set_epi32(
          (int)g.begin(cur.is[7])[k.is[7]],
          (int)g.begin(cur.is[6])[k.is[6]],
          (int)g.begin(cur.is[5])[k.is[5]],
          (int)g.begin(cur.is[4])[k.is[4]],
          (int)g.begin(cur.is[3])[k.is[3]],
          (int)g.begin(cur.is[2])[k.is[2]],
          (int)g.begin(cur.is[1])[k.is[1]],
          (int)g.begin(cur.is[0])[k.is[0]]);
      secret = _mm256_mullo_epi32(secret, ds.i);
      pvec[cur.is[0]] += 1.0f;
      pvec[cur.is[1]] += 1.0f;
      pvec[cur.is[2]] += 1.0f;
      pvec[cur.is[3]] += 1.0f;
      pvec[cur.is[4]] += 1.0f;
      pvec[cur.is[5]] += 1.0f;
      pvec[cur.is[6]] += 1.0f;
      pvec[cur.is[7]] += 1.0f;
    }
    // Resume at len with the same vertices; never check for zero again.
    for (; len < len_walk; len++) {
      avx2_t rnd, ds, k;
      rnd.i = splitmix32_avx2(random_state);
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
  if (n_walks % 8 != 0 || n_walks > UINT32_MAX) {
    std::cerr << "n_walks must be a multiple of 8 within the 32-bit grid domain\n";
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

  // ---- SABA random walk phase (timed) ------------------------------------
  auto t2 = std::chrono::high_resolution_clock::now();

  const bool do_walks = (n_walks > 0);

  double visit_count = 0.0, checksum = 0.0;

#pragma omp parallel reduction(+:visit_count,checksum)
  {
    float* const pvec = new float[n]();

#pragma omp for schedule(dynamic, 1024)
    for (long long src = 0; src < (long long)n; ++src) {
      const uint32_t src_u = (uint32_t)src;
      if (g.degree(src_u) == 0)
        continue;

      if (do_walks)
        saba_randomwalk(src_u, depth, n_walks, pvec, g);
    }
    for (size_t vertex = 0; vertex < n; ++vertex) {
      visit_count += pvec[vertex];
      checksum += double(vertex + 1) * pvec[vertex];
    }
    delete[] pvec;
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

  std::cout << VARIANT << "\t" << dataset << "\t" << depth << "\t" << n_walks << "\t"
            << threads << "\t" << n << "\t" << m << "\t" << t_load << "\t"
            << t_rw << "\t" << total << "\t" << total_walks << "\t"
            << walks_per_sec << "\t" << visit_count << "\t" << checksum << "\n";

  return 0;
}
