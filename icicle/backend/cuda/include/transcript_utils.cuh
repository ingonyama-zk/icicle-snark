#pragma once

#include <cuda.h>

// Device function to convert a byte vector to a field (reducing hash result to field)
template <typename F>
__launch_bounds__(1) __global__
  void reduce_hash_result_to_field(F* alpha, std::byte* hash_result, size_t hash_result_size)
{
  alpha[0] = F::from(hash_result, hash_result_size);
}

// Device function to append a bytes vector to another bytes vector
__device__ inline void append_data(std::byte* byte_vec, std::byte* label, const size_t label_size, int& start_idx)
{
  // Copy label elements to the end of byte_vec
  for (size_t i = 0; i < label_size; ++i) {
    byte_vec[start_idx + i] = label[i];
  }
  start_idx += label_size;
}

// Device function to append a uint32_t as bytes to a device_vector<byte>
__device__ inline void append_u32(std::byte* byte_vec, const uint32_t data, int& start_idx)
{
  // Convert uint32_t to bytes
  const std::byte* data_bytes = reinterpret_cast<const std::byte*>(&data);

  // Copy the bytes to the end of the byte_vec
  for (int i = 0; i < sizeof(uint32_t); ++i)
    byte_vec[start_idx + i] = data_bytes[i];

  start_idx += sizeof(uint32_t);
}

// Device function to append a field element to the byte vector
template <typename F>
__device__ inline void append_field(std::byte* byte_vec, const F& field, int& start_idx)
{
  const std::byte* data_bytes = reinterpret_cast<const std::byte*>(&field);

  // Copy the bytes of the field to the byte vector
  for (int i = 0; i < sizeof(F); ++i)
    byte_vec[start_idx + i] = data_bytes[i];

  start_idx += sizeof(F);
}
