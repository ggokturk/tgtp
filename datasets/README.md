# Dataset preparation

Store each dataset in `<name>/graph.txt`. The C++ loader expects a whitespace
separated undirected edge list with zero-based, contiguous vertex IDs and
no isolated vertices. List each undirected edge once. Blank lines and lines
beginning with `#` or `%` are accepted. Use simple, connected, non-bipartite
graphs for the spanning-centrality pipeline; the eigenvalue-based truncation
assumes mixing of the random walk. Do not add a vertex/edge-count header.

For spanning centrality, prepare `sorted_eigens_<omega>.txt`: one eigenvalue
followed by its `n` eigenvector components per row, with `omega` rows sorted
by descending absolute eigenvalue. Components follow vertex-ID order.

The `edgeutils` executable performs both preparation stages from this
repository's `src/`:

```sh
edgeutils -d datasets/urls.txt datasets   # download and remap to graph.txt
edgeutils -c datasets/Facebook 128        # sorted_eigens_128.txt + stat.txt
edgeutils -call datasets 128              # every dataset missing its eigens
```

Choose `2 <= omega < n` for the sparse eigensolver. The eigen computation
uses Eigen and Spectra (Lanczos, largest-magnitude rule) in double
precision; the tolerance is 1e-4 above one million vertices and 1e-6 below,
as before.

## URLs and download options

`urls.txt` carries one dataset per line with optional flag columns:

```
name url [undirected|directed] [skip-first]
```

- `undirected` (default) canonicalizes each edge to `(min,max)` and
  deduplicates symmetrically; `directed` keeps the source orientation and
  deduplicates exact pairs. Both stages keep vertex IDs contiguous by
  remapping names in order of first appearance.
- `skip-first` drops the first non-comment line, for sources whose first
  line is a vertex/edge-count header rather than an edge. `Delaunay-N21`
  is a MatrixMarket pattern-symmetric `.mtx` (a `%%` banner plus a
  dimension line, then 1-based row/col pairs): the banner is dropped as a
  `%` comment and `skip-first` drops the dimension line, so the plain
  pipeline remaps it like any other source.
- Downloads go over plain http where possible; the C++ tool links OpenSSL
  and follows http->https redirects (e.g. nrvis.com).

Downloaded dataset directories, caches, and experiment outputs are
intentionally ignored by Git.
