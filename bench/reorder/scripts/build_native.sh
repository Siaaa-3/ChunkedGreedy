#!/usr/bin/env bash
# Build the native reordering extensions used in the paper.
# Usage: ./scripts/build_native.sh [acc|clusterwise|all]
#   acc          AccReorder and rabbit-local-greedy
#   clusterwise  ClusterHier
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${1:-all}"
PYTHON="${PYTHON:-python}"

python_ext_suffix() {
  "$PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))'
}

pybind11_includes() {
  "$PYTHON" -m pybind11 --includes
}

conda_flags() {
  conda_include=()
  conda_lib=()
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    conda_include=(-I"$CONDA_PREFIX/include")
    conda_lib=(-L"$CONDA_PREFIX/lib")
    local t
    for t in "$CONDA_PREFIX"/targets/*-linux; do
      [[ -d "$t/include" ]] && conda_include+=(-I"$t/include")
      [[ -d "$t/lib" ]] && conda_lib+=(-L"$t/lib")
    done
  fi
}

build_acc() {
  echo "[build_native] acc: building acc_reorder_core"
  local src_dir="$ROOT/native/acc_core"
  local build_dir="$src_dir/build"
  local sanitized_dir="$build_dir/sanitized_order"
  local upstream_order="$ROOT/Reordering_methods/Acc_spmm_reorder/order"
  local output
  output="$ROOT/reorder_bench/acc_reorder_core$(python_ext_suffix)"

  rm -rf "$sanitized_dir"
  mkdir -p "$sanitized_dir"
  cp "$upstream_order"/*.hpp "$sanitized_dir"/
  # drop the ''' ... ''' blocks (commented-out Python) from my_order.hpp
  "$PYTHON" - "$upstream_order/my_order.hpp" "$sanitized_dir/my_order.hpp" <<'PY'
from pathlib import Path
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
skip_doc_block = False
out = []
for line in src.read_text(encoding="utf-8").splitlines(keepends=True):
    if line.strip() == "'''":
        skip_doc_block = not skip_doc_block
        continue
    if skip_doc_block:
        continue
    out.append(line)
dst.write_text("".join(out), encoding="utf-8")
PY

  conda_flags
  # shellcheck disable=SC2046
  g++ -O3 -std=c++17 -shared -fPIC -fopenmp \
    $(pybind11_includes) \
    "${conda_include[@]}" \
    -I"$sanitized_dir" \
    "$src_dir/acc_core.cpp" \
    -o "$output" \
    "${conda_lib[@]}" \
    -lnuma
  echo "[build_native] acc: wrote $output"
}

build_clusterwise() {
  echo "[build_native] clusterwise: building clusterwise_reorder_core"
  local src_dir="$ROOT/native/clusterwise_core"
  local shim_dir="$src_dir/build/simde_shim"
  local upstream="$ROOT/Reordering_methods/clusterwise-spgemm"
  local output
  output="$ROOT/reorder_bench/clusterwise_reorder_core$(python_ext_suffix)"

  # the upstream code includes x86 intrinsics; SIMDe maps them on other architectures
  mkdir -p "$shim_dir"
  cat > "$shim_dir/immintrin.h" <<'EOF'
#pragma once
#define SIMDE_ENABLE_NATIVE_ALIASES
#include <simde/x86/avx2.h>
#ifndef _popcnt32
#define _popcnt32(x) __builtin_popcount(static_cast<unsigned int>(x))
#endif
EOF
  cat > "$shim_dir/x86intrin.h" <<'EOF'
#pragma once
#include "immintrin.h"
EOF

  conda_flags
  # shellcheck disable=SC2046
  g++ -O3 -std=c++17 -shared -fPIC -fopenmp \
    $(pybind11_includes) \
    -I"$shim_dir" \
    "${conda_include[@]}" \
    -I"$upstream" \
    -I"$upstream/GTgraph/sprng2.0-lite/include" \
    "$src_dir/clusterwise_core.cpp" \
    -o "$output" \
    "${conda_lib[@]}" \
    -ltbb -ltbbmalloc
  echo "[build_native] clusterwise: wrote $output"
}

case "$TARGET" in
  acc) build_acc ;;
  clusterwise) build_clusterwise ;;
  all) build_acc; build_clusterwise ;;
  *)
    echo "Unknown target: $TARGET" >&2
    exit 1
    ;;
esac
