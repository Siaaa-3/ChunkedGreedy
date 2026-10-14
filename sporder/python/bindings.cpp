#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cuda_runtime.h>
#include <sporder.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

template <typename T>
using gpu_in =
    nb::ndarray<const T, nb::ndim<1>, nb::device::cuda, nb::c_contig>;

template <typename T>
using gpu_ret = nb::ndarray<T, nb::ndim<1>, nb::device::cuda>;

struct py_context {
  sporder_context_t handle;

  py_context() : handle(sporder_context_create()) {}
  ~py_context() {
    if (handle) {
      sporder_context_destroy(handle);
    }
  }

  py_context(const py_context &) = delete;
  py_context &operator=(const py_context &) = delete;
  py_context(py_context &&) = delete;
  py_context &operator=(py_context &&) = delete;
};

py_context &default_context() {
  static py_context *ctx = new py_context();
  return *ctx;
}

template <typename T>
nb::list to_list(const std::vector<T> &values) {
  nb::list out;
  for (const T &value : values) {
    out.append(value);
  }
  return out;
}

template <typename T>
nb::capsule cuda_alloc(int64_t count) {
  T *data = nullptr;
  cudaError_t err = cudaMalloc(&data, static_cast<size_t>(count) * sizeof(T));
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMalloc failed: ") +
                             cudaGetErrorString(err));
  }
  return nb::capsule(data, [](void *p) noexcept { cudaFree(p); });
}

template <typename T>
nb::object as_gpu_array(T *data, int64_t size, nb::capsule owner) {
  int device_id = 0;
  cudaGetDevice(&device_id);
  return nb::cast(gpu_ret<T>(data, {static_cast<size_t>(size)}, owner, {},
                             nb::dtype<T>(), nb::device::cuda::value,
                             device_id));
}

nb::object hierarchical_matching(gpu_in<int32_t> ptr, gpu_in<int32_t> ind,
                                 bool is_symmetric, int32_t num_levels,
                                 int32_t k, float oversampling,
                                 int32_t max_rounds, bool with_timing,
                                 bool with_stats, py_context *ctx_arg) {
  py_context &ctx = ctx_arg ? *ctx_arg : default_context();
  if (ptr.size() < 2) {
    throw std::invalid_argument("ptr must have length n+1 (at least 2)");
  }
  if (num_levels < 1 || num_levels > 30) {
    throw std::invalid_argument("num_levels must be in [1, 30]");
  }
  int64_t n = static_cast<int64_t>(ptr.size()) - 1;
  int64_t nnz = static_cast<int64_t>(ind.size());
  if (n > INT32_MAX || nnz > INT32_MAX) {
    throw std::invalid_argument("matrix too large for int32 indexing");
  }

  sporder_options_t opts{k, oversampling, max_rounds};

  nb::capsule owner = cuda_alloc<int32_t>(n); // freed with the returned array
  auto *d_new_order = static_cast<int32_t *>(owner.data());

  std::vector<float> matching_per_level(num_levels, 0.0f);
  std::vector<float> advance_per_level(num_levels, 0.0f);
  sporder_hm_timing_t timing{};
  timing.matching_per_level_ms = matching_per_level.data();
  timing.advance_per_level_ms = advance_per_level.data();

  std::vector<int32_t> level_size(num_levels, 0);
  std::vector<int32_t> matched_pairs(num_levels, 0);
  std::vector<int32_t> rounds(num_levels, 0);
  sporder_hm_stats_t stats{};
  stats.level_size = level_size.data();
  stats.matched_pairs = matched_pairs.data();
  stats.rounds = rounds.data();

  {
    nb::gil_scoped_release release;
    sporder_hierarchical_matching(
        ctx.handle, static_cast<int32_t>(n), static_cast<int32_t>(nnz),
        ptr.data(), ind.data(), is_symmetric, num_levels, &opts, d_new_order,
        with_timing ? &timing : nullptr, with_stats ? &stats : nullptr);
  }

  nb::object new_order = as_gpu_array(d_new_order, n, owner);

  nb::object timing_info = nb::none();
  if (with_timing) {
    nb::dict d;
    d["wall_ms"] = timing.wall_ms;
    d["csr2csc_ms"] = timing.csr2csc_ms;
    d["similarity_ms"] = timing.similarity_ms;
    d["matching_ms"] = timing.matching_ms;
    d["advance_ms"] = timing.advance_ms;
    d["matching_per_level_ms"] = to_list(matching_per_level);
    d["advance_per_level_ms"] = to_list(advance_per_level);
    timing_info = d;
  }

  nb::object stats_info = nb::none();
  if (with_stats) {
    nb::dict d;
    d["k_used"] = stats.k_used;
    d["m_used"] = stats.m_used;
    d["avg_degree"] = stats.avg_degree;
    d["avg_edge_weight"] = stats.avg_edge_weight;
    d["zero_degree_rows"] = stats.zero_degree_rows;
    d["level_size"] = to_list(level_size);
    d["matched_pairs"] = to_list(matched_pairs);
    d["rounds"] = to_list(rounds);
    stats_info = d;
  }

  return nb::make_tuple(new_order, timing_info, stats_info);
}

nb::object reorder(gpu_in<int32_t> ptr, gpu_in<int32_t> ind, nb::object val,
                   gpu_in<int32_t> new_order, bool symmetric,
                   py_context *ctx_arg) {
  py_context &ctx = ctx_arg ? *ctx_arg : default_context();
  if (ptr.size() < 2) {
    throw std::invalid_argument("ptr must have length n+1 (at least 2)");
  }
  int64_t n = static_cast<int64_t>(ptr.size()) - 1;
  int64_t nnz = static_cast<int64_t>(ind.size());
  if (static_cast<int64_t>(new_order.size()) != n) {
    throw std::invalid_argument("new_order must have length n");
  }
  if (n > INT32_MAX || nnz > INT32_MAX) {
    throw std::invalid_argument("matrix too large for int32 indexing");
  }

  gpu_in<float> val_in =
      val.is_none() ? gpu_in<float>() : nb::cast<gpu_in<float>>(val);
  if (!val.is_none() && (val_in.device_type() != ptr.device_type() ||
                         val_in.device_id() != ptr.device_id())) {
    throw std::invalid_argument("val must live on the same device as ptr");
  }

  nb::capsule ptr_owner = cuda_alloc<int32_t>(n + 1);
  nb::capsule ind_owner = cuda_alloc<int32_t>(std::max<int64_t>(nnz, 1));
  auto *d_ptr_new = static_cast<int32_t *>(ptr_owner.data());
  auto *d_ind_new = static_cast<int32_t *>(ind_owner.data());

  nb::capsule val_owner;
  float *d_val_new = nullptr;
  if (!val.is_none()) {
    val_owner = cuda_alloc<float>(std::max<int64_t>(nnz, 1));
    d_val_new = static_cast<float *>(val_owner.data());
  }

  {
    nb::gil_scoped_release release;
    sporder_reorder(ctx.handle, static_cast<int32_t>(n),
                    static_cast<int32_t>(nnz), ptr.data(), ind.data(),
                    val.is_none() ? nullptr : val_in.data(), new_order.data(),
                    symmetric ? SPORDER_REORDER_SYMMETRIC
                              : SPORDER_REORDER_ROWS,
                    d_ptr_new, d_ind_new, d_val_new);
  }

  nb::tuple result = nb::make_tuple(as_gpu_array(d_ptr_new, n + 1, ptr_owner),
                                    as_gpu_array(d_ind_new, nnz, ind_owner));
  if (!val.is_none()) {
    result += nb::make_tuple(as_gpu_array(d_val_new, nnz, val_owner));
  }
  return result;
}

int32_t count_pillars(gpu_in<int32_t> ptr, gpu_in<int32_t> ind, int32_t height,
                      py_context *ctx_arg) {
  py_context &ctx = ctx_arg ? *ctx_arg : default_context();
  if (ptr.size() < 2) {
    throw std::invalid_argument("ptr must have length n+1 (at least 2)");
  }
  if (height < 1) {
    throw std::invalid_argument("height must be >= 1");
  }
  int64_t n = static_cast<int64_t>(ptr.size()) - 1;
  int64_t nnz = static_cast<int64_t>(ind.size());
  if (n > INT32_MAX || nnz > INT32_MAX) {
    throw std::invalid_argument("matrix too large for int32 indexing");
  }

  int32_t count = 0;
  {
    nb::gil_scoped_release release;
    sporder_count_pillars(ctx.handle, static_cast<int32_t>(n),
                          static_cast<int32_t>(nnz), ptr.data(), ind.data(),
                          height, &count);
  }
  return count;
}

float bench_cusparse_spmm(gpu_in<int32_t> ptr, gpu_in<int32_t> ind,
                          int32_t cols, int32_t warmup, int32_t repeat,
                          uint64_t seed, py_context *ctx_arg) {
  py_context &ctx = ctx_arg ? *ctx_arg : default_context();
  if (ptr.size() < 2) {
    throw std::invalid_argument("ptr must have length n+1 (at least 2)");
  }
  if (warmup < 0 || repeat < 0) {
    throw std::invalid_argument("warmup and repeat must be >= 0");
  }
  if (cols < 1) {
    throw std::invalid_argument("cols must be >= 1");
  }
  int64_t n = static_cast<int64_t>(ptr.size()) - 1;
  int64_t nnz = static_cast<int64_t>(ind.size());
  if (n > INT32_MAX || nnz > INT32_MAX) {
    throw std::invalid_argument("matrix too large for int32 indexing");
  }

  float wall_ms = 0.0f;
  {
    nb::gil_scoped_release release;
    sporder_bench_cusparse_spmm(
        ctx.handle, static_cast<int32_t>(n), static_cast<int32_t>(nnz),
        ptr.data(), ind.data(), cols, warmup, repeat, seed, &wall_ms);
  }
  return wall_ms;
}

nb::object chunk_greedy(gpu_in<int32_t> ptr, gpu_in<int32_t> ind,
                        bool is_symmetric, int32_t k, float oversampling,
                        int32_t chunk_size, bool with_timing, bool with_stats,
                        py_context *ctx_arg) {
  py_context &ctx = ctx_arg ? *ctx_arg : default_context();
  if (ptr.size() < 2) {
    throw std::invalid_argument("ptr must have length n+1 (at least 2)");
  }
  int64_t n = static_cast<int64_t>(ptr.size()) - 1;
  int64_t nnz = static_cast<int64_t>(ind.size());
  if (n > INT32_MAX || nnz > INT32_MAX) {
    throw std::invalid_argument("matrix too large for int32 indexing");
  }

  nb::capsule owner = cuda_alloc<int32_t>(n);
  auto *d_new_order = static_cast<int32_t *>(owner.data());

  sporder_cg_timing_t timing = {};
  sporder_cg_stats_t stats = {};

  {
    nb::gil_scoped_release release;
    sporder_chunk_greedy(ctx.handle, static_cast<int32_t>(n),
                         static_cast<int32_t>(nnz), ptr.data(), ind.data(),
                         is_symmetric, k, oversampling, chunk_size, d_new_order,
                         with_timing ? &timing : nullptr,
                         with_stats ? &stats : nullptr);
  }

  nb::object new_order = as_gpu_array(d_new_order, n, owner);

  nb::object timing_info = nb::none();
  if (with_timing) {
    nb::dict t;
    t["wall_ms"] = timing.wall_ms;
    t["csr2csc_ms"] = timing.csr2csc_ms;
    t["similarity_ms"] = timing.similarity_ms;
    t["chunk_greedy_ms"] = timing.chunk_greedy_ms;
    timing_info = t;
  }

  nb::object stats_info = nb::none();
  if (with_stats) {
    nb::dict s;
    s["k_used"] = stats.k_used;
    s["m_used"] = stats.m_used;
    s["avg_degree"] = stats.avg_degree;
    s["avg_edge_weight"] = stats.avg_edge_weight;
    s["zero_degree_rows"] = stats.zero_degree_rows;
    s["chunk_size_used"] = stats.chunk_size_used;
    s["chain_restarts"] = stats.chain_restarts;
    s["avg_followed_weight"] = stats.avg_followed_weight;
    s["cut_weight_fraction"] = stats.cut_weight_fraction;
    stats_info = s;
  }

  return nb::make_tuple(new_order, timing_info, stats_info);
}

nb::object chunk_similarity_greedy(gpu_in<int32_t> ptr, gpu_in<int32_t> ind,
                                   bool is_symmetric, int32_t k,
                                   float oversampling, int32_t chunk_size,
                                   bool with_timing, bool with_stats,
                                   py_context *ctx_arg) {
  py_context &ctx = ctx_arg ? *ctx_arg : default_context();
  if (ptr.size() < 2) {
    throw std::invalid_argument("ptr must have length n+1 (at least 2)");
  }
  int64_t n = static_cast<int64_t>(ptr.size()) - 1;
  int64_t nnz = static_cast<int64_t>(ind.size());
  if (n > INT32_MAX || nnz > INT32_MAX) {
    throw std::invalid_argument("matrix too large for int32 indexing");
  }

  nb::capsule owner = cuda_alloc<int32_t>(n);
  auto *d_new_order = static_cast<int32_t *>(owner.data());

  sporder_csg_timing_t timing = {};
  sporder_csg_stats_t stats = {};

  {
    nb::gil_scoped_release release;
    sporder_chunk_similarity_greedy(
        ctx.handle, static_cast<int32_t>(n), static_cast<int32_t>(nnz),
        ptr.data(), ind.data(), is_symmetric, k, oversampling, chunk_size,
        d_new_order, with_timing ? &timing : nullptr,
        with_stats ? &stats : nullptr);
  }

  nb::object new_order = as_gpu_array(d_new_order, n, owner);

  nb::object timing_info = nb::none();
  if (with_timing) {
    nb::dict t;
    t["wall_ms"] = timing.wall_ms;
    t["csr2csc_ms"] = timing.csr2csc_ms;
    t["similarity_ms"] = timing.similarity_ms;
    t["chunk_greedy_ms"] = timing.chunk_greedy_ms;
    timing_info = t;
  }

  nb::object stats_info = nb::none();
  if (with_stats) {
    nb::dict s;
    s["k_used"] = stats.k_used;
    s["m_used"] = stats.m_used;
    s["avg_degree"] = stats.avg_degree;
    s["avg_edge_weight"] = stats.avg_edge_weight;
    s["zero_degree_rows"] = stats.zero_degree_rows;
    s["chunk_size_used"] = stats.chunk_size_used;
    s["chain_restarts"] = stats.chain_restarts;
    s["avg_followed_weight"] = stats.avg_followed_weight;
    stats_info = s;
  }

  return nb::make_tuple(new_order, timing_info, stats_info);
}

nb::object similarity_graph(gpu_in<int32_t> ptr, gpu_in<int32_t> ind,
                            bool is_symmetric, int32_t k, float oversampling,
                            bool with_timing, bool with_stats,
                            py_context *ctx_arg) {
  py_context &ctx = ctx_arg ? *ctx_arg : default_context();
  if (ptr.size() < 2) {
    throw std::invalid_argument("ptr must have length n+1 (at least 2)");
  }
  int64_t n = static_cast<int64_t>(ptr.size()) - 1;
  int64_t nnz = static_cast<int64_t>(ind.size());
  if (n > INT32_MAX || nnz > INT32_MAX) {
    throw std::invalid_argument("matrix too large for int32 indexing");
  }

  if (k <= 0) {
    k = std::min(32, std::max(16, 2 * static_cast<int32_t>(nnz / n)));
  }

  int64_t nk = static_cast<int64_t>(n) * k;
  nb::capsule ind_owner = cuda_alloc<int32_t>(nk);
  nb::capsule weight_owner = cuda_alloc<float>(nk);

  auto *d_indices = static_cast<int32_t *>(ind_owner.data());
  auto *d_weights = static_cast<float *>(weight_owner.data());

  sporder_sim_timing_t timing = {};
  sporder_sim_stats_t stats = {};

  {
    nb::gil_scoped_release release;
    sporder_similarity_graph(ctx.handle, static_cast<int32_t>(n),
                             static_cast<int32_t>(nnz), ptr.data(), ind.data(),
                             is_symmetric, k, oversampling, d_indices,
                             d_weights, with_timing ? &timing : nullptr,
                             with_stats ? &stats : nullptr);
  }

  nb::object idx_obj = as_gpu_array(d_indices, nk, ind_owner);
  nb::object wgt_obj = nb::cast(
      gpu_ret<float>(d_weights, {static_cast<size_t>(nk)}, weight_owner, {},
                     nb::dtype<float>(), nb::device::cuda::value, 0));

  nb::object timing_info = nb::none();
  if (with_timing) {
    nb::dict t;
    t["wall_ms"] = timing.wall_ms;
    t["csr2csc_ms"] = timing.csr2csc_ms;
    t["similarity_ms"] = timing.similarity_ms;
    t["unpack_ms"] = timing.unpack_ms;
    timing_info = t;
  }

  nb::object stats_info = nb::none();
  if (with_stats) {
    nb::dict s;
    s["k_used"] = stats.k_used;
    s["m_used"] = stats.m_used;
    s["avg_degree"] = stats.avg_degree;
    s["avg_edge_weight"] = stats.avg_edge_weight;
    s["zero_degree_rows"] = stats.zero_degree_rows;
    stats_info = s;
  }

  return nb::make_tuple(idx_obj, wgt_obj, timing_info, stats_info);
}

nb::object similarity_graph_exact(gpu_in<int32_t> ptr, gpu_in<int32_t> ind,
                                  bool is_symmetric, int32_t k,
                                  bool with_timing, bool with_stats,
                                  py_context *ctx_arg) {
  py_context &ctx = ctx_arg ? *ctx_arg : default_context();
  if (ptr.size() < 2) {
    throw std::invalid_argument("ptr must have length n+1 (at least 2)");
  }
  int64_t n = static_cast<int64_t>(ptr.size()) - 1;
  int64_t nnz = static_cast<int64_t>(ind.size());
  if (n > INT32_MAX || nnz > INT32_MAX) {
    throw std::invalid_argument("matrix too large for int32 indexing");
  }

  if (k <= 0) {
    k = std::min(32, std::max(16, 2 * static_cast<int32_t>(nnz / n)));
  }

  int64_t nk = static_cast<int64_t>(n) * k;
  nb::capsule ind_owner = cuda_alloc<int32_t>(nk);
  nb::capsule weight_owner = cuda_alloc<float>(nk);

  auto *d_indices = static_cast<int32_t *>(ind_owner.data());
  auto *d_weights = static_cast<float *>(weight_owner.data());

  sporder_sim_exact_timing_t timing = {};
  sporder_sim_exact_stats_t stats = {};

  {
    nb::gil_scoped_release release;
    sporder_similarity_graph_exact(
        ctx.handle, static_cast<int32_t>(n), static_cast<int32_t>(nnz),
        ptr.data(), ind.data(), is_symmetric, k, d_indices, d_weights,
        with_timing ? &timing : nullptr, with_stats ? &stats : nullptr);
  }

  nb::object idx_obj = as_gpu_array(d_indices, nk, ind_owner);
  nb::object wgt_obj = nb::cast(
      gpu_ret<float>(d_weights, {static_cast<size_t>(nk)}, weight_owner, {},
                     nb::dtype<float>(), nb::device::cuda::value, 0));

  nb::object timing_info = nb::none();
  if (with_timing) {
    nb::dict t;
    t["wall_ms"] = timing.wall_ms;
    t["csr2csc_ms"] = timing.csr2csc_ms;
    t["spgemm_ms"] = timing.spgemm_ms;
    t["dice_ms"] = timing.dice_ms;
    t["sort_ms"] = timing.sort_ms;
    t["select_ms"] = timing.select_ms;
    timing_info = t;
  }

  nb::object stats_info = nb::none();
  if (with_stats) {
    nb::dict s;
    s["k_used"] = stats.k_used;
    s["c_nnz"] = static_cast<int64_t>(stats.c_nnz);
    s["avg_degree"] = stats.avg_degree;
    s["avg_edge_weight"] = stats.avg_edge_weight;
    s["zero_degree_rows"] = stats.zero_degree_rows;
    stats_info = s;
  }

  return nb::make_tuple(idx_obj, wgt_obj, timing_info, stats_info);
}

} // namespace

NB_MODULE(_sporder, m) {
  m.doc() = "GPU sparse matrix reordering via hierarchical matching";

  nb::class_<py_context>(m, "Context",
                         "CUDA execution context owning a stream and "
                         "cuSPARSE handle")
      .def(nb::init<>());

  m.def("hierarchical_matching", &hierarchical_matching, "ptr"_a, "ind"_a,
        "is_symmetric"_a = false, "num_levels"_a = 1, "k"_a = 0,
        "oversampling"_a = 0.0f, "max_rounds"_a = 0, "with_timing"_a = false,
        "with_stats"_a = false, "ctx"_a.none() = nb::none(),
        "Low-level entry point; use sporder.hierarchical_matching instead.\n"
        "Returns (new_order, timing, stats): a freshly allocated int32 CUDA\n"
        "array exposing DLPack, and two dicts that are None unless the\n"
        "corresponding with_* flag is set.");

  m.def("reorder", &reorder, "ptr"_a, "ind"_a, "val"_a = nb::none(),
        "new_order"_a, "symmetric"_a = false, "ctx"_a.none() = nb::none(),
        "Low-level entry point; use sporder.reorder instead.\n"
        "Returns (ptr_new, ind_new) as freshly allocated int32 CUDA\n"
        "arrays exposing DLPack.");

  m.def("count_pillars", &count_pillars, "ptr"_a, "ind"_a, "height"_a,
        "ctx"_a.none() = nb::none(),
        "Low-level entry point; use sporder.count_pillars instead.\n"
        "Returns the total number of nonzero columns over all row\n"
        "windows of the given height.");

  m.def("bench_cusparse_spmm", &bench_cusparse_spmm, "ptr"_a, "ind"_a, "cols"_a,
        "warmup"_a = 3, "repeat"_a = 10, "seed"_a = 0,
        "ctx"_a.none() = nb::none(),
        "Benchmark cuSPARSE SpMM on the given pattern.\n"
        "Returns the average wall-clock time per repeat in millseconds.");

  m.def("similarity_graph", &similarity_graph, "ptr"_a, "ind"_a,
        "is_symmetric"_a = false, "k"_a = 0, "oversampling"_a = 0.0f,
        "with_timing"_a = false, "with_stats"_a = false,
        "ctx"_a.none() = nb::none(),
        "Build the weighted top-k Dice-similarity graph.\n"
        "Returns (indices, weights, timing, stats): two freshly allocated\n"
        "CUDA arrays exposing DLPack, and two dicts that are None unless\n"
        "the corresponding with_* flag is set.");

  m.def("chunk_greedy", &chunk_greedy, "ptr"_a, "ind"_a,
        "is_symmetric"_a = false, "k"_a = 0, "oversampling"_a = 0.0f,
        "chunk_size"_a = 0, "with_timing"_a = false, "with_stats"_a = false,
        "ctx"_a.none() = nb::none(),
        "One-shot chunked greedy: similarity -> chunks -> permutation.\n"
        "Returns (new_order, timing, stats): a freshly allocated int32 CUDA\n"
        "array exposing DLPack, and two dicts that are None unless the\n"
        "corresponding with_* flag is set.");

  m.def("chunk_similarity_greedy", &chunk_similarity_greedy, "ptr"_a, "ind"_a,
        "is_symmetric"_a = false, "k"_a = 0, "oversampling"_a = 0.0f,
        "chunk_size"_a = 0, "with_timing"_a = false, "with_stats"_a = false,
        "ctx"_a.none() = nb::none(),
        "Per-chunk similarity + greedy seriation.\n"
        "Similarity is computed within each chunk independently, reducing\n"
        "work from O(n d^2) to O(chunk_size d^2).\n"
        "Returns (new_order, timing, stats): a freshly allocated int32 CUDA\n"
        "array exposing DLPack, and two dicts that are None unless the\n"
        "corresponding with_* flag is set.");

  m.def("similarity_graph_exact", &similarity_graph_exact, "ptr"_a, "ind"_a,
        "is_symmetric"_a = false, "k"_a = 0, "with_timing"_a = false,
        "with_stats"_a = false, "ctx"_a.none() = nb::none(),
        "Exact top-k Dice-similarity graph via cuSPARSE SpGEMM.\n"
        "Returns (indices, weights, timing, stats): two freshly allocated\n"
        "CUDA arrays exposing DLPack, and two dicts that are None unless\n"
        "the corresponding with_* flag is set.");
}
