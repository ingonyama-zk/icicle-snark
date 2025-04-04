#pragma once

#include "cuda.h"
#include "gpu-utils/error_handler.h"

template <typename E>
static E* allocate_on_device(size_t byte_size, cudaStream_t cuda_stream)
{
  E* device_mem = nullptr;
  cudaMallocAsync(&device_mem, byte_size, cuda_stream);
  CHK_LAST();

  return device_mem;
}

template <typename E>
static const E* allocate_and_copy_to_device(const E* host_mem, size_t byte_size, cudaStream_t cuda_stream)
{
  E* device_mem = nullptr;
  cudaMallocAsync(&device_mem, byte_size, cuda_stream);
  cudaMemcpyAsync(device_mem, host_mem, byte_size, cudaMemcpyHostToDevice, cuda_stream);
  CHK_LAST();

  return device_mem;
}