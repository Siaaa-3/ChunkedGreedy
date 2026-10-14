#pragma once
#include <bit>
#include <cstdint>

namespace sporder {

class weighted_index {
  uint64_t data_;

public:
  __device__ __forceinline__ weighted_index() : data_(0) {}

  __device__ __forceinline__ weighted_index(uint64_t data) : data_(data) {}

  __device__ __forceinline__ weighted_index(int32_t index, float weight)
      : data_((static_cast<uint64_t>(std::bit_cast<uint32_t>(weight)) << 32) |
              (static_cast<uint32_t>(index + 1))) {}

  __device__ __forceinline__ operator uint64_t() const { return data_; }

  __device__ __forceinline__ int32_t index() const {
    return static_cast<int32_t>(data_ & 0xFFFFFFFF) - 1;
  }

  __device__ __forceinline__ float weight() const {
    return std::bit_cast<float>(static_cast<uint32_t>(data_ >> 32));
  }

  __device__ __forceinline__ bool
  operator==(const weighted_index &other) const {
    return data_ == other.data_;
  }
  __device__ __forceinline__ bool
  operator!=(const weighted_index &other) const {
    return data_ != other.data_;
  }
  __device__ __forceinline__ bool operator<(const weighted_index &other) const {
    return data_ < other.data_;
  }
  __device__ __forceinline__ bool operator>(const weighted_index &other) const {
    return data_ > other.data_;
  }
  __device__ __forceinline__ bool
  operator<=(const weighted_index &other) const {
    return data_ <= other.data_;
  }
  __device__ __forceinline__ bool
  operator>=(const weighted_index &other) const {
    return data_ >= other.data_;
  }

  __device__ __forceinline__ weighted_index
  atomic_update_if_greater(weighted_index other) {
    static_assert(sizeof(uint64_t) == sizeof(unsigned long long));
    return weighted_index(
        atomicMax(reinterpret_cast<unsigned long long *>(&data_),
                  static_cast<unsigned long long>(other.data_)));
  }

  __device__ __forceinline__ weighted_index replace_index(int32_t index) {
    return weighted_index((data_ & 0xFFFFFFFF00000000ULL) |
                          static_cast<uint32_t>(index + 1));
  }
};

} // namespace sporder
