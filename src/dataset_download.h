#pragma once
// Dataset acquisition: downloads SNAP-style archives over plain HTTP,
// decompresses them, remaps vertex names to contiguous zero-based IDs and
// writes <name>/graph.txt. The zip path replaces commas with spaces
// (twitch_gamers); undirected sources are deduplicated symmetrically,
// directed ones exactly.

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "httplib.h"
#include "miniz.h"

// httplib.h pulls in glibc's resolv.h, which defines `_res` as a macro; it
// collides with Eigen's template parameter names. Drop it after use.
#ifdef _res
#undef _res
#endif

namespace dataset {

struct UrlSpec {
  std::string name;
  std::string url;
  bool directed = false;
  bool skip_first = false;
};

// Parses `name url [undirected|directed] [skip-first]` lines; blank lines and
// #-comments are ignored.
inline std::vector<UrlSpec> read_url_list(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open url list: " + path);
  std::vector<UrlSpec> specs;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && (line.at(0) == '#' || line.at(0) == '%')) continue;
    std::istringstream ss(line);
    UrlSpec spec;
    std::string flag;
    if (!(ss >> spec.name >> spec.url)) continue;
    while (ss >> flag) {
      if (flag == "directed") spec.directed = true;
      else if (flag == "undirected") spec.directed = false;
      else if (flag == "skip-first") spec.skip_first = true;
      else throw std::runtime_error("unknown flag '" + flag + "' in " + path);
    }
    specs.push_back(spec);
  }
  if (specs.empty()) throw std::runtime_error("no urls in " + path);
  return specs;
}

// Streams the response body to dest. Only plain http is supported; https
// urls are rewritten with a warning (no TLS stack is linked).
inline void http_get(const std::string& url, const std::filesystem::path& dest) {
  std::string effective = url;
  if (effective.rfind("https://", 0) == 0) {
    effective = "http://" + effective.substr(8);
    std::cout << "rewriting to plain http (no TLS in edgeutils): " << effective
              << std::endl;
  }
  if (effective.rfind("http://", 0) != 0)
    throw std::runtime_error("only plain http urls are supported: " + url);

  const std::string rest = effective.substr(7);
  const size_t slash = rest.find('/');
  if (slash == std::string::npos)
    throw std::runtime_error("malformed url: " + url);
  std::string host = rest.substr(0, slash);
  const std::string target = rest.substr(slash);
  int port = 80;
  const size_t colon = host.rfind(':');
  if (colon != std::string::npos) {
    port = std::stoi(host.substr(colon + 1));
    host = host.substr(0, colon);
  }

  httplib::Client cli(host, port);
  cli.set_follow_location(true);
  cli.set_read_timeout(600, 0);
  cli.set_write_timeout(600, 0);
  std::ofstream out(dest, std::ios::binary);
  if (!out) throw std::runtime_error("cannot open " + dest.string());
  int status = 0;
  const auto res = cli.Get(target, [&](const char* data, size_t len) {
    out.write(data, (std::streamsize)len);
    return bool(out);
  });
  out.close();
  if (res) status = res->status;
  const auto err = res.error();
  if (err != httplib::Error::Success || status != 200) {
    std::filesystem::remove(dest);
    throw std::runtime_error("download failed for " + url + " (status " +
                             std::to_string(status) + ", error " +
                             httplib::to_string(err) + ")");
  }
}

// Reads a gzip stream: skips the header fields and inflates the raw deflate
// payload. The CRC32/ISIZE trailer is not verified, matching download.py.
inline std::string gunzip(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open archive: " + path.string());
  std::string compressed((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
  in.close();
  if (compressed.size() < 18 || (unsigned char)compressed[0] != 0x1f ||
      (unsigned char)compressed[1] != 0x8b || compressed[2] != 8)
    throw std::runtime_error("not a deflate gzip stream: " + path.string());
  const unsigned char flg = compressed[3];
  size_t pos = 10;
  if (flg & 4) {  // FEXTRA
    if (compressed.size() < pos + 2)
      throw std::runtime_error("truncated gzip header");
    const size_t xlen = (unsigned char)compressed[pos] |
                        ((unsigned char)compressed[pos + 1] << 8);
    pos += 2 + xlen;
  }
  if (flg & 8) {  // FNAME
    const size_t end = compressed.find('\0', pos);
    if (end == std::string::npos)
      throw std::runtime_error("truncated gzip header");
    pos = end + 1;
  }
  if (flg & 16) {  // FCOMMENT
    const size_t end = compressed.find('\0', pos);
    if (end == std::string::npos)
      throw std::runtime_error("truncated gzip header");
    pos = end + 1;
  }
  if (flg & 2) pos += 2;  // FHCRC
  if (pos + 8 >= compressed.size())
    throw std::runtime_error("truncated gzip stream: " + path.string());

  mz_stream stream{};
  stream.next_in = (const unsigned char*)compressed.data() + pos;
  stream.avail_in = (unsigned int)(compressed.size() - pos - 8);
  if (mz_inflateInit2(&stream, -MZ_DEFAULT_WINDOW_BITS) != MZ_OK)
    throw std::runtime_error("miniz inflate init failed");
  std::string out;
  std::vector<unsigned char> buf(1u << 16);
  for (;;) {
    stream.next_out = buf.data();
    stream.avail_out = (mz_uint)buf.size();
    const int ret = mz_inflate(&stream, MZ_NO_FLUSH);
    out.append(reinterpret_cast<const char*>(buf.data()),
               buf.size() - stream.avail_out);
    if (ret == MZ_STREAM_END) break;
    if (ret != MZ_OK && ret != MZ_BUF_ERROR) {
      mz_inflateEnd(&stream);
      throw std::runtime_error("miniz inflate failed: " + path.string());
    }
    if (ret == MZ_BUF_ERROR && stream.avail_in == 0)  // truncated stream
      break;
  }
  mz_inflateEnd(&stream);
  return out;
}

// Extracts the first file entry of a zip archive (z.infolist()[0] parity).
inline std::string unzip_first(const std::filesystem::path& path) {
  mz_zip_archive za{};
  if (!mz_zip_reader_init_file(&za, path.string().c_str(), 0))
    throw std::runtime_error("cannot open zip archive: " + path.string());
  const mz_uint nfiles = mz_zip_reader_get_num_files(&za);
  mz_uint idx = 0;
  for (; idx < nfiles; idx++)
    if (!mz_zip_reader_is_file_a_directory(&za, idx)) break;
  if (idx >= nfiles) {
    mz_zip_reader_end(&za);
    throw std::runtime_error("zip archive contains no files: " + path.string());
  }
  size_t size = 0;
  char* data = (char*)mz_zip_reader_extract_to_heap(&za, idx, &size, 0);
  mz_zip_reader_end(&za);
  if (!data) throw std::runtime_error("zip extraction failed: " + path.string());
  std::string text(data, size);
  mz_free(data);
  std::replace(text.begin(), text.end(), ',', ' ');
  return text;
}

inline std::string lowercase_suffix(const std::string& url) {
  const size_t dot = url.rfind('.');
  if (dot == std::string::npos) return "";
  std::string ext = url.substr(dot);
  for (char& c : ext) c = (char)std::tolower((unsigned char)c);
  return ext;
}

// Extracts and writes <folder>/graph.txt: remaps vertex names to contiguous
// zero-based IDs by first appearance and deduplicates edges (exactly for
// directed sources, symmetrically otherwise).
inline void write_graph_file(const std::filesystem::path& folder,
                             const UrlSpec& spec, const std::string& text) {
  std::unordered_map<std::string, uint32_t> index;
  index.reserve(1u << 20);
  uint32_t next_id = 0;
  std::vector<std::pair<uint32_t, uint32_t>> pairs;
  bool skip_pending = spec.skip_first;

  size_t start = 0;
  while (start < text.size()) {
    size_t end = text.find('\n', start);
    if (end == std::string::npos) end = text.size();
    size_t len = end - start;
    if (len > 0 && text[start + len - 1] == '\r') len--;
    const std::string_view line(text.data() + start, len);
    start = end + 1;

    size_t p = line.find_first_not_of(" \t");
    if (p == std::string_view::npos) continue;
    if (line[p] == '#' || line[p] == '%') continue;
    if (skip_pending) {
      skip_pending = false;
      continue;
    }
    size_t q = line.find_first_of(" \t", p);
    if (q == std::string_view::npos) continue;
    size_t r = line.find_first_not_of(" \t", q);
    if (r == std::string_view::npos) continue;
    size_t s = line.find_first_of(" \t", r);
    if (s == std::string_view::npos) s = line.size();
    const std::string src(line.substr(p, q - p));
    const std::string dst(line.substr(r, s - r));
    size_t extra = line.find_first_not_of(" \t", s);
    if (extra != std::string_view::npos)
      std::cout << "err: " << line << std::endl;  // download.py parity
    auto it = index.find(src);
    if (it == index.end()) it = index.emplace(src, next_id++).first;
    auto it2 = index.find(dst);
    if (it2 == index.end()) it2 = index.emplace(dst, next_id++).first;
    pairs.emplace_back(it->second, it2->second);
  }

  std::ofstream out(folder / "graph.txt");
  if (!out) throw std::runtime_error("cannot write " + (folder / "graph.txt").string());
  std::unordered_set<uint64_t> seen;
  seen.reserve(pairs.size() * 2);
  size_t m = 0;
  for (auto& [u, v] : pairs) {
    const uint64_t key =
        spec.directed ? ((uint64_t)u << 32) | v
                      : ((uint64_t)(std::min)(u, v) << 32) | (std::max)(u, v);
    if (!seen.insert(key).second) continue;
    out << (spec.directed ? u : (std::min)(u, v)) << '\t'
        << (spec.directed ? v : (std::max)(u, v)) << '\n';
    m++;
  }
  std::cout << spec.name << ": n=" << next_id << ", m=" << m << std::endl;
}

inline std::string extract_archive(const std::filesystem::path& archive,
                                   const std::string& url) {
  const std::string ext = lowercase_suffix(url);
  if (ext == ".zip") return unzip_first(archive);
  if (ext == ".gz") return gunzip(archive);
  std::ifstream in(archive, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read archive: " + archive.string());
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

// Downloads every `name url` pair of the list into <outroot>/<name>/
// graph.txt, skipping datasets that already carry a graph.
inline void download_datasets(const std::string& urls_path,
                              const std::filesystem::path& outroot) {
  const auto specs = read_url_list(urls_path);
  for (auto& spec : specs) {
    std::cout << spec.name << " " << spec.url << std::endl;
    const auto folder = outroot / spec.name;
    std::filesystem::create_directories(folder);
    const auto graph_path = folder / "graph.txt";
    if (std::filesystem::exists(graph_path)) {
      std::cout << "File " << graph_path.string() << " exists" << std::endl;
      continue;
    }
    const auto tmp = folder / ".archive_download";
    try {
      http_get(spec.url, tmp);
      const std::string text = extract_archive(tmp, spec.url);
      std::filesystem::remove(tmp);
      write_graph_file(folder, spec, text);
    } catch (...) {
      std::error_code ec;
      std::filesystem::remove(tmp, ec);
      throw;
    }
  }
}

}  // namespace dataset
