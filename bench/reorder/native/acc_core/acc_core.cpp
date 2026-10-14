#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "my_order.hpp"

namespace py = pybind11;

namespace {

void clear_acc_globals() {
  all_vertex_nbrs.clear();
  std::map<vint, std::vector<vint>>().swap(all_vertex_nbrs);
  comm_map.clear();
  std::map<vint, std::vector<vint>>().swap(comm_map);
  v2comm.clear();
  std::vector<vint>().swap(v2comm);
}

std::vector<vint> reorder_vertex_py(const adjacency_list& adj) {
  clear_acc_globals();
  std::deque<vint> order = re2order_vertex(adj);
  std::vector<vint> out(order.begin(), order.end());
  clear_acc_globals();
  return out;
}

// Rabbit aggregate + trace_com only. Returns communities sorted by root
// (comm_id), each as (root, members). Root is the community representative
// and is always included in members when present in the graph.
std::vector<std::pair<vint, std::vector<vint>>> rabbit_communities_py(
    const adjacency_list& adj) {
  clear_acc_globals();
  get_comm_map(adj);

  std::vector<std::pair<vint, std::vector<vint>>> out;
  out.reserve(comm_map.size());
  for (auto& item : comm_map) {
    out.emplace_back(item.first, std::move(item.second));
  }
  clear_acc_globals();
  return out;
}

}  // namespace

PYBIND11_MODULE(acc_reorder_core, m) {
  m.doc() = "pybind11 wrapper for Acc_spmm_reorder re2order_vertex";
  m.def(
      "reorder_vertex",
      &reorder_vertex_py,
      py::arg("adj"),
      "Run Acc re2order_vertex on an adjacency list and return new-to-old row order.");
  m.def(
      "rabbit_communities",
      &rabbit_communities_py,
      py::arg("adj"),
      "Rabbit community partition: list of (root, members), sorted by root id.");
}
