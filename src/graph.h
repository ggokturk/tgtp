#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <vector>
namespace fuser {
template<typename idx_t=uint32_t, typename vert_t=uint32_t>
struct graph_t
{
  idx_t* __restrict idx = NULL;
  vert_t* __restrict adj = NULL;
  size_t n = 0, m = 0;
  vert_t degree(vert_t vertex)
  {
    return end(vertex) - begin(vertex);
  }
  vert_t* begin(vert_t vertex) { return adj + idx[vertex]; }
  vert_t* end(vert_t vertex) { return adj + idx[vertex + 1]; }
  void free()
  {
    if (idx != NULL)
      delete[] idx;
    if (adj != NULL)
      delete[] adj;
  }
  graph_t& load_txt(std::string filename,
                    bool directed = false,
                    size_t skip = 0)
  {
    std::ifstream in(filename, std::fstream::in);
    if (!in) throw std::runtime_error("cannot open graph: " + filename);
    in.sync_with_stdio(false);
    std::vector<std::vector<vert_t>> adjlist;
    vert_t s, v;
    size_t n = 0, m = 0;
    std::string line;
    while (std::getline(in, line)) {
      if (line.length() > 0 && (line.at(0) == '#' || line.at(0) == '%'))
        continue;
      if (skip > 0) {
        skip--;
        continue;
      }
      std::stringstream ss(line);

      ss >> std::ws;
      if (ss.eof() || ss.peek() == '#' || ss.peek() == '%') continue;
      int64_t source, target;
      if (!(ss >> source >> target) || source < 0 || target < 0 ||
          source >= std::numeric_limits<vert_t>::max() ||
          target >= std::numeric_limits<vert_t>::max())
        throw std::runtime_error("invalid edge in " + filename + ": " + line);
      s = static_cast<vert_t>(source);
      v = static_cast<vert_t>(target);

      if (adjlist.size() < ((std::max)(s, v) + 1)) {
        adjlist.resize((std::max)(s, v) + 1);
      }

      adjlist[s].push_back(v);

      if (!directed)
        adjlist[v].push_back(s);
      n = (std::max)(size_t((std::max)(s, v) + 1), n);
    }
    if (adjlist.empty()) throw std::runtime_error("empty graph: " + filename);
    for (size_t i = 0; i < adjlist.size(); i++) {
      m += adjlist[i].size();
    }
    if (m > std::numeric_limits<idx_t>::max())
      throw std::runtime_error("graph exceeds CSR index capacity");
#pragma omp parallel for
    for (int64_t i = 0; i < (int64_t)adjlist.size(); i++) {
      sort(adjlist[i].begin(), adjlist[i].end());
    }

    this->n = n;
    this->m = m;
    this->idx = new idx_t[n + 1];
    this->adj = new vert_t[m];
    idx_t pos = 0;
    this->idx[0] = 0;
    for (size_t i = 0; i < n; i++) {
      pos += adjlist[i].size();
      this->idx[i + 1] = pos;
    }
    pos = 0;
    for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < adjlist[i].size(); j++) {
        this->adj[pos++] = adjlist[i][j];
      }
    }
    return *this;
  }
};
}