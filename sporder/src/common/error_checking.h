#pragma once

#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include <source_location>
#include <string>

namespace sporder {

template <typename error_t>
struct error_traits {
  static constexpr char *name();
  static bool is_error(error_t err);
  static std::string message(error_t err);

  static_assert(sizeof(error_t) == 0, "unknown error_t");
};

template <typename error_t>
void ensure(error_t err,
            std::source_location loc = std::source_location::current()) {
  if (error_traits<error_t>::is_error(err)) {
    std::cerr << loc.file_name() << ':' << loc.line() << ':' << loc.column()
              << ": ensure failed - " << error_traits<error_t>::name()
              << " error: " << error_traits<error_t>::message(err) << '\n';
    std::abort();
  }
}

template <>
struct error_traits<cudaError_t> {
  static constexpr const char *name() { return "CUDA"; }

  static bool is_error(cudaError_t err) { return err != cudaSuccess; }

  static std::string message(cudaError_t err) {
    return cudaGetErrorString(err);
  }
};

template <>
struct error_traits<cusparseStatus_t> {
  static constexpr const char *name() { return "cuSPARSE"; }

  static bool is_error(cusparseStatus_t err) {
    return err != CUSPARSE_STATUS_SUCCESS;
  }

  static std::string message(cusparseStatus_t err) {
    return cusparseGetErrorString(err);
  }
};

inline void expect(bool expected, const std::string &msg,
                   std::source_location loc = std::source_location::current()) {
  if (!expected) {
    std::cerr << loc.file_name() << ':' << loc.line() << ':' << loc.column()
              << ": expect failed - " << msg << '\n';
    std::abort();
  }
}

} // namespace sporder
