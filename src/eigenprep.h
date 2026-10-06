#pragma once
// Eigenpair preparation for spanning centrality: reads <folder>/graph.txt,
// computes the top-omega eigenpairs of the symmetrically normalized adjacency
// matrix A[i,j]/sqrt(deg_i*deg_j) with a Lanczos solver, back-transforms the
// vectors by sqrt(2m/deg_i) (isolated vertices get zero), sorts by descending
// absolute eigenvalue and writes sorted_eigens_<omega>.txt and stat.txt.
//
// This file is based on calEigen.py from the upstream AESC project:
// https://github.com/jeremyzhangsq/AESC. It is solely used for regularizing
// input files across implementations; refer to the original author for the
// license.
//
// Replaces calEigen.py. The D^0.5 * P * D^-0.5 construction used there
// simplifies exactly to A[i,j]/sqrt(deg_i*deg_j) because the 0.5/m scaling
// of the diagonal factors cancels against the un-normalization transform.

#include <Eigen/SparseCore>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/SymEigsSolver.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dataset {

struct EdgeList {
  std::vector<std::pair<uint32_t, uint32_t>> edges;  // unique, undirected
  uint32_t n = 0;  // distinct vertices
  size_t m = 0;    // unique edges; self-loops count once
};

inline EdgeList read_edges(const std::filesystem::path& graph_path) {
  std::ifstream in(graph_path);
  if (!in) throw std::runtime_error("cannot open graph: " + graph_path.string());
  std::vector<std::pair<uint32_t, uint32_t>> raw;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line.at(0) == '#' || line.at(0) == '%') continue;
    if (line.back() == '\r') line.pop_back();
    std::istringstream ss(line);
    ss >> std::ws;
    if (ss.eof() || ss.peek() == '#' || ss.peek() == '%') continue;
    int64_t s, t;
    if (!(ss >> s >> t) || s < 0 || t < 0 ||
        s > (int64_t)std::numeric_limits<uint32_t>::max() - 1 ||
        t > (int64_t)std::numeric_limits<uint32_t>::max() - 1)
      throw std::runtime_error("invalid edge in " + graph_path.string() + ": " +
                               line);
    raw.emplace_back((uint32_t)s, (uint32_t)t);
  }
  if (raw.empty()) throw std::runtime_error("empty graph: " + graph_path.string());
  // Rank-map vertex IDs by sorted order, mirroring the sorted-nodelist
  // semantics of networkx in calEigen.py (identity for contiguous IDs).
  std::vector<uint32_t> ids;
  ids.reserve(raw.size() * 2);
  for (auto& e : raw) {
    ids.push_back(e.first);
    ids.push_back(e.second);
  }
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  auto rank = [&ids](uint32_t v) {
    return (uint32_t)(std::lower_bound(ids.begin(), ids.end(), v) - ids.begin());
  };
  EdgeList result;
  result.n = (uint32_t)ids.size();
  result.edges.reserve(raw.size());
  for (auto& e : raw) {
    auto key = e.first <= e.second
                   ? std::make_pair(rank(e.first), rank(e.second))
                   : std::make_pair(rank(e.second), rank(e.first));
    result.edges.push_back(key);
  }
  std::sort(result.edges.begin(), result.edges.end());
  result.edges.erase(std::unique(result.edges.begin(), result.edges.end()),
                     result.edges.end());
  result.m = result.edges.size();
  return result;
}

// Computes and writes the eigenpair files for one dataset folder.
inline void eigenprep_folder(const std::filesystem::path& folder, size_t omega) {
  using Eigen::Index;
  const auto graph_path = folder / "graph.txt";
  EdgeList g = read_edges(graph_path);
  if (omega < 2 || omega >= g.n)
    throw std::runtime_error("omega must satisfy 2 <= omega < n; got " +
                             std::to_string(omega) + " for n=" +
                             std::to_string(g.n));
  const double n = g.n, m = g.m;
  std::cout << "f=" << folder.string() << ", n=" << g.n << ", m=" << g.m
            << ", ad=" << std::fixed << std::setprecision(6) << (2.0 * m / n)
            << std::endl;

  // Row sums of the undirected adjacency (networkx convention: a self-loop
  // contributes once to the adjacency diagonal).
  std::vector<double> rs(g.n, 0.0);
  for (auto& [u, v] : g.edges) {
    rs[u] += 1.0;
    if (u != v) rs[v] += 1.0;
  }

  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(g.edges.size() * 2);
  for (auto& [u, v] : g.edges) {
    const double val = 1.0 / std::sqrt(rs[u] * rs[v]);
    triplets.emplace_back(u, v, val);
    if (u != v) triplets.emplace_back(v, u, val);
  }
  Eigen::SparseMatrix<double> p((Index)g.n, (Index)g.n);
  p.setFromTriplets(triplets.begin(), triplets.end());
  triplets.clear();
  triplets.shrink_to_fit();

  const double tol = g.n > 1000000 ? 1e-4 : 1e-6;
  const Index ncv = (std::min)(p.cols(),
                               (Index)std::max<size_t>(2 * omega + 1, 20));
  const auto start = std::chrono::steady_clock::now();
  Spectra::SparseSymMatProd<double> op(p);
  Spectra::SymEigsSolver<Spectra::SparseSymMatProd<double>> eigs(
      op, (Index)omega, ncv);
  eigs.init();
  const Index nconv =
      eigs.compute(Spectra::SortRule::LargestMagn, 10 * p.cols(), tol,
                   Spectra::SortRule::LargestMagn);
  const double secs =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  if (eigs.info() != Spectra::CompInfo::Successful || nconv < (Index)omega)
    throw std::runtime_error("eigensolver did not converge for " +
                             graph_path.string() + ": converged " +
                             std::to_string(nconv) + "/" +
                             std::to_string(omega));
  std::cout << "omega:" << omega << ",tol:" << tol << ",ncv:" << ncv
            << ",time:" << std::setprecision(12) << secs << std::endl;

  Eigen::VectorXd vals = eigs.eigenvalues();
  Eigen::MatrixXd vecs = eigs.eigenvectors();

  // Back-transform: x_i = v_i * sqrt(2m / rowsum_i), zero for isolated
  // vertices (matching Dnegsqrt * vecs in calEigen.py).
  Eigen::MatrixXd x(vecs.rows(), vecs.cols());
  const double norm2m = std::sqrt(2.0 * m);
  for (Index j = 0; j < x.cols(); j++)
    for (Index i = 0; i < x.rows(); i++)
      x(i, j) = rs[i] > 0 ? vecs(i, j) * norm2m / std::sqrt(rs[i]) : 0.0;

  std::vector<size_t> order(omega);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&vals](size_t a, size_t b) {
    return std::abs(vals[a]) > std::abs(vals[b]);
  });

  {
    std::ofstream sf(folder / "stat.txt");
    if (!sf) throw std::runtime_error("cannot write stat.txt in " + folder.string());
    char statbuf[128];
    std::snprintf(statbuf, sizeof(statbuf), "n=%d\nm=%zu\nl=%f", g.n, g.m,
                  vals[(Index)order[0]]);
    sf << statbuf;
  }
  {
    const auto out_path =
        folder / ("sorted_eigens_" + std::to_string(omega) + ".txt");
    std::ofstream of(out_path, std::ios::out);
    if (!of) throw std::runtime_error("cannot write " + out_path.string());
    of << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (size_t j : order) {
      of << vals[(Index)j];
      for (Index i = 0; i < x.rows(); i++) of << ' ' << x(i, (Index)j);
      of << '\n';
    }
  }
  std::cout << "wrote " << folder.string() << "/sorted_eigens_" << omega
            << ".txt" << std::endl;
}

// Runs eigenprep for every immediate subdirectory of root that does not
// already carry sorted_eigens_<omega>.txt.
inline void eigenprep_batch(const std::filesystem::path& root, size_t omega) {
  std::vector<std::filesystem::path> folders;
  for (auto& entry : std::filesystem::directory_iterator(root))
    if (entry.is_directory()) folders.push_back(entry.path());
  std::sort(folders.begin(), folders.end());
  for (auto& folder : folders) {
    const auto outfile = folder / ("sorted_eigens_" + std::to_string(omega) + ".txt");
    std::cout << outfile.string() << std::endl;
    if (std::filesystem::exists(outfile)) {
      std::cout << "Exists: " << outfile.string() << std::endl;
      continue;
    }
    try {
      eigenprep_folder(folder, omega);
    } catch (const std::exception& e) {
      std::cerr << "error: " << folder.string() << ": " << e.what() << std::endl;
    }
  }
}

}  // namespace dataset
