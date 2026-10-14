/**
 * PyBind11 wrapper for AccSpMM prepare/run benchmark API.
 */
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <torch/extension.h>

#include "acc_spmm_api.h"
#include "common.hpp"

namespace py = pybind11;

py::capsule acc_spmm_prepare_py(py::array_t<int> row_ptr, py::array_t<int> col_idx,
                                py::array_t<float> values, int num_nodes,
                                int num_edges) {
  AccSpMMHandle* handle = acc_spmm_prepare(
      row_ptr.data(0), col_idx.data(0), values.data(0), num_nodes, num_edges, -1);
  return py::capsule(handle, [](void* ptr) {
    acc_spmm_release(reinterpret_cast<AccSpMMHandle*>(ptr));
  });
}

py::capsule acc_spmm_prepare_with_balance_py(
    py::array_t<int> row_ptr, py::array_t<int> col_idx,
    py::array_t<float> values, int num_nodes, int num_edges,
    bool force_balance) {
  AccSpMMHandle* handle = acc_spmm_prepare(
      row_ptr.data(0), col_idx.data(0), values.data(0), num_nodes, num_edges,
      force_balance ? 1 : 0);
  return py::capsule(handle, [](void* ptr) {
    acc_spmm_release(reinterpret_cast<AccSpMMHandle*>(ptr));
  });
}

float acc_spmm_run_py(py::capsule handle_capsule, torch::Tensor B) {
  TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
  TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
  auto* handle = reinterpret_cast<AccSpMMHandle*>(handle_capsule.get_pointer());
  return acc_spmm_run(handle, B.data_ptr<float>(), static_cast<int>(B.size(1)));
}

PYBIND11_MODULE(acc_spmm_bench, m) {
  m.doc() = "AccSpMM benchmark wrapper";
  m.attr("ROW_WINDOW") = ROW_WINDOW;
  m.attr("COL_WINDOW") = COL_WINDOW;
  m.def("prepare", &acc_spmm_prepare_py,
        "Prepare AccSpMM sparse GPU buffers for a matrix");
  m.def("prepare_with_balance", &acc_spmm_prepare_with_balance_py,
        py::arg("row_ptr"), py::arg("col_idx"), py::arg("values"),
        py::arg("num_nodes"), py::arg("num_edges"),
        py::arg("force_balance"),
        "Prepare AccSpMM and force the balanced or unbalanced format");
  m.def("run", &acc_spmm_run_py,
        "Run AccSpMM kernel once; returns kernel time in ms");
}
