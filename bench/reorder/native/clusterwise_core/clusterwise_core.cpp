#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <new>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace fs = std::filesystem;

#include "CSR.h"
#include "Triple.h"
#include "hash_mult.h"

namespace py = pybind11;

namespace {

using Index = int64_t;
using Value = double;
using ClusterMap = std::map<Index, std::vector<Index>>;
using ClosePairs = std::map<std::pair<Index, Index>, Value>;
using Item = std::pair<Value, std::pair<Index, Index>>;

double now_ms() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::milli>(
             clock::now().time_since_epoch())
      .count();
}

CSR<Index, Value> make_csr(
    const std::vector<Index>& indptr,
    const std::vector<Index>& indices,
    const std::vector<Value>& values,
    Index nrows) {
  if (indptr.size() != static_cast<size_t>(nrows + 1)) {
    throw std::invalid_argument("indptr length must be nrows + 1");
  }
  if (indices.size() != values.size()) {
    throw std::invalid_argument("indices and values must have the same length");
  }
  std::vector<Triple<Index, Value>> triples;
  triples.reserve(indices.size());
  for (Index row = 0; row < nrows; ++row) {
    for (Index pos = indptr[row]; pos < indptr[row + 1]; ++pos) {
      triples.emplace_back(row, indices[pos], values[pos]);
    }
  }
  return CSR<Index, Value>(triples.data(), static_cast<Index>(triples.size()), nrows, nrows);
}

CSR<Index, Value> make_transpose_csr(
    const std::vector<Index>& indptr,
    const std::vector<Index>& indices,
    const std::vector<Value>& values,
    Index nrows) {
  std::vector<Triple<Index, Value>> triples;
  triples.reserve(indices.size());
  for (Index row = 0; row < nrows; ++row) {
    for (Index pos = indptr[row]; pos < indptr[row + 1]; ++pos) {
      triples.emplace_back(indices[pos], row, values[pos]);
    }
  }
  return CSR<Index, Value>(triples.data(), static_cast<Index>(triples.size()), nrows, nrows);
}

ClosePairs topk_close_pairs(CSR<Index, Value>& a, CSR<Index, Value>& b, Index topk) {
  // Match GenerateCandidatePairs.cpp: sortOutput=false, then sortValues().
  const bool sort_output = false;
  CSR<Index, Value> c;
  c.is_value_sorted = sort_output;

  a.Sorted();
  b.Sorted();
  a.ResetValues();
  b.ResetValues();

  HashSpGEMMTopK<sort_output>(
      a, b, c, std::multiplies<Value>(), std::plus<Value>(), topk, "");

  if (!sort_output) {
    c.sortValues();
  }

  ClosePairs close_pairs;
  for (Index row = 0; row < c.rows; ++row) {
    Index count = 0;
    for (Index pos = c.rowptr[row]; pos < c.rowptr[row + 1]; ++pos) {
      Index col = c.colids[pos];
      if (col == row) {
        continue;
      }
      close_pairs[std::make_pair(row, col)] = c.values[pos];
      ++count;
      if (count == topk) {
        break;
      }
    }
  }
  c.make_empty();
  return close_pairs;
}

ClusterMap hierarchical_clustering(
    CSR<Index, Value>& sp,
    ClosePairs& close_pairs,
    int cluster_size) {
  auto cmp = [](const Item& a, const Item& b) { return a.first < b.first; };
  std::priority_queue<Item, std::vector<Item>, decltype(cmp)> sims(cmp);

  for (auto& p : close_pairs) {
    sims.push(std::make_pair(p.second, p.first));
  }

  std::vector<Index> clusters(sp.rows);
  std::vector<Index> sz(sp.rows);
  std::vector<Index> valid(sp.rows);
  int nclusters = static_cast<int>(sp.rows);
  for (Index i = 0; i < sp.rows; ++i) {
    clusters[i] = i;
    sz[i] = 1;
    valid[i] = 1;
  }

  while (!sims.empty() && nclusters != 0) {
    Item s = sims.top();
    sims.pop();
    Index i = s.second.first;
    Index j = s.second.second;
    if (clusters[i] == i && clusters[j] == j) {
      if (!valid[i] || !valid[j]) {
        continue;
      }
      nclusters--;
      if (sz[i] < sz[j]) {
        clusters[i] = j;
        sz[j] += sz[i];
        if (sz[j] >= cluster_size) {
          valid[j] = 0;
          nclusters--;
        }
      } else {
        clusters[j] = i;
        sz[i] += sz[j];
        if (sz[i] >= cluster_size) {
          valid[i] = 0;
          nclusters--;
        }
      }
    } else {
      while (i != clusters[i]) {
        clusters[i] = clusters[clusters[i]];
        i = clusters[i];
      }
      while (j != clusters[j]) {
        clusters[j] = clusters[clusters[j]];
        j = clusters[j];
      }
      if (!valid[i] || !valid[j]) {
        continue;
      }
      if (i != j) {
        auto p = std::make_pair(i, j);
        if (close_pairs.find(p) == close_pairs.end()) {
          Value score = sp.jaccard_similarity(i, j);
          sims.push(std::make_pair(score, p));
          close_pairs[p] = score;
        }
      }
    }
  }

  ClusterMap reordered_dict;
  for (Index i = 0; i < static_cast<Index>(clusters.size()); ++i) {
    Index j = i;
    while (j != clusters[j]) {
      j = clusters[j];
    }
    reordered_dict[j].push_back(i);
  }
  return reordered_dict;
}

py::tuple reorder(
    const std::vector<Index>& indptr,
    const std::vector<Index>& indices,
    const std::vector<Value>& values,
    Index nrows,
    Index topk,
    int cluster_size) {
  if (topk <= 0) {
    throw std::invalid_argument("topk must be positive");
  }
  if (cluster_size <= 0) {
    throw std::invalid_argument("cluster_size must be positive");
  }

  CSR<Index, Value> a = make_csr(indptr, indices, values, nrows);
  CSR<Index, Value> b = make_transpose_csr(indptr, indices, values, nrows);

  double t0 = now_ms();
  ClosePairs close_pairs;
  try {
    close_pairs = topk_close_pairs(a, b, topk);
  } catch (const std::bad_alloc&) {
    a.make_empty();
    b.make_empty();
    throw std::runtime_error(
        "clusterwise-hier OOM during HashSpGEMMTopK (A*A^T topk); "
        "rows=" +
        std::to_string(nrows) + ", topk=" + std::to_string(topk));
  } catch (const std::runtime_error&) {
    a.make_empty();
    b.make_empty();
    throw;
  }
  double topk_ms = now_ms() - t0;

  ClusterMap clusters = hierarchical_clustering(a, close_pairs, cluster_size);

  a.make_empty();
  b.make_empty();

  py::dict parts;
  parts["topk_ms"] = topk_ms;
  return py::make_tuple(clusters, parts);
}

std::vector<Index> clusters_to_row_perm(const ClusterMap& clusters) {
  std::vector<Index> row_perm;
  for (const auto& item : clusters) {
    row_perm.insert(row_perm.end(), item.second.begin(), item.second.end());
  }
  return row_perm;
}

}  // namespace

PYBIND11_MODULE(clusterwise_reorder_core, m) {
  m.doc() = "pybind11 wrapper for clusterwise-spgemm TopK + hierarchical clustering";
  m.def("reorder", &reorder, py::arg("indptr"), py::arg("indices"), py::arg("values"),
        py::arg("nrows"), py::arg("topk"), py::arg("cluster_size"));
  m.def("clusters_to_row_perm", &clusters_to_row_perm, py::arg("clusters"));
}
