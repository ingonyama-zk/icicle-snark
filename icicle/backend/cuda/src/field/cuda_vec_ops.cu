#include <cuda.h>
#include <cub/cub.cuh>
#include <stdexcept>
#include <thrust/device_vector.h>

#include "icicle/errors.h"
#include "icicle/backend/vec_ops_backend.h"
#include "gpu-utils/error_handler.h"
#include "error_translation.h"
#include "gpu-utils/utils.h"

#include "icicle/fields/field_config.h"
using namespace field_config;

#define MAX_THREADS_PER_BLOCK 256

template <typename E, typename F, void (*Kernel)(const E*, const F*, uint64_t, E*)>
cudaError_t vec_op(
  const E* a, const F* b, uint64_t size_a, uint64_t size_b, const VecOpsConfig& config, E* result, uint64_t size_res)
{
  CHK_INIT_IF_RETURN();

  size_a *= config.batch_size;
  size_b *= config.batch_size;
  size_res *= config.batch_size;

  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(config.stream);

  // allocate device memory and copy if input/output not on device already
  a = config.is_a_on_device ? a : allocate_and_copy_to_device(a, size_a * sizeof(E), cuda_stream);
  b = config.is_b_on_device ? b : allocate_and_copy_to_device(b, size_b * sizeof(E), cuda_stream);
  E* d_result = config.is_result_on_device ? result : allocate_on_device<E>(size_res * sizeof(E), cuda_stream);

  // Call the kernel to perform element-wise operation
  uint64_t num_threads = MAX_THREADS_PER_BLOCK;
  uint64_t num_blocks = (size_res + num_threads - 1) / num_threads;
  Kernel<<<num_blocks, num_threads, 0, cuda_stream>>>(a, b, size_res, d_result);

  // copy back result to host if need to
  if (!config.is_result_on_device) {
    CHK_IF_RETURN(cudaMemcpyAsync(result, d_result, size_res * sizeof(E), cudaMemcpyDeviceToHost, cuda_stream));
    CHK_IF_RETURN(cudaFreeAsync(d_result, cuda_stream));
  }

  // release device memory, if allocated
  // the cast is ugly but it makes the code more compact
  if (!config.is_a_on_device) { CHK_IF_RETURN(cudaFreeAsync((void*)a, cuda_stream)); }
  if (!config.is_b_on_device) { CHK_IF_RETURN(cudaFreeAsync((void*)b, cuda_stream)); }

  // wait for stream to empty is not async
  if (!config.is_async) return CHK_STICKY(cudaStreamSynchronize(cuda_stream));

  return CHK_LAST();
}

template <typename E, void (*Kernel)(const E*, const E*, uint64_t, uint64_t, bool, E*)>
cudaError_t vec_scalar_op(
  const E* scalar,
  const E* vec,
  uint64_t scalar_size,
  uint64_t vec_size,
  const VecOpsConfig& config,
  E* result,
  uint64_t res_size)
{
  CHK_INIT_IF_RETURN();

  res_size *= config.batch_size;

  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(config.stream);

  // allocate device memory and copy if input/output not on device already
  scalar = config.is_a_on_device
             ? scalar
             : allocate_and_copy_to_device(scalar, scalar_size * (config.batch_size) * sizeof(E), cuda_stream);
  vec = config.is_b_on_device ? vec
                              : allocate_and_copy_to_device(vec, vec_size * config.batch_size * sizeof(E), cuda_stream);
  E* d_result = config.is_result_on_device ? result : allocate_on_device<E>(res_size * sizeof(E), cuda_stream);

  // Call the kernel to perform element-wise operation
  uint64_t num_threads = MAX_THREADS_PER_BLOCK;
  uint64_t num_blocks = (res_size + num_threads - 1) / num_threads;
  Kernel<<<num_blocks, num_threads, 0, cuda_stream>>>(
    scalar, vec, vec_size, config.batch_size, config.columns_batch, d_result);

  // copy back result to host if need to
  if (!config.is_result_on_device) {
    CHK_IF_RETURN(cudaMemcpyAsync(result, d_result, res_size * sizeof(E), cudaMemcpyDeviceToHost, cuda_stream));
    CHK_IF_RETURN(cudaFreeAsync(d_result, cuda_stream));
  }

  // release device memory, if allocated
  // the cast is ugly but it makes the code more compact
  if (!config.is_a_on_device) { CHK_IF_RETURN(cudaFreeAsync((void*)scalar, cuda_stream)); }
  if (!config.is_b_on_device) { CHK_IF_RETURN(cudaFreeAsync((void*)vec, cuda_stream)); }

  // wait for stream to empty is not async
  if (!config.is_async) return CHK_STICKY(cudaStreamSynchronize(cuda_stream));

  return CHK_LAST();
}

/*============================== add ==============================*/
template <typename E>
__global__ void add_kernel(const E* element_vec1, const E* element_vec2, uint64_t size, E* result)
{
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) { result[tid] = element_vec1[tid] + element_vec2[tid]; }
}

template <typename E>
eIcicleError
add_cuda(const Device& device, const E* vec_a, const E* vec_b, uint64_t size, const VecOpsConfig& config, E* result)
{
  cudaError_t err = vec_op<E, E, add_kernel>(vec_a, vec_b, size, size, config, result, size);
  return translateCudaError(err);
}

/*============================== accumulate ==============================*/
template <typename E>
eIcicleError accumulate_cuda(const Device& device, E* vec_a, const E* vec_b, uint64_t size, const VecOpsConfig& config)
{
  cudaError_t err = vec_op<E, E, add_kernel>(vec_a, vec_b, size, size, config, vec_a, size);
  return translateCudaError(err);
}

template <typename E>
__global__ void add_scalar_kernel(
  const E* scalar, const E* element_vec, uint64_t vec_size, uint64_t nof_vecs, bool columns_batch, E* result)
{
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < vec_size * nof_vecs) {
    E scalar_val = scalar[columns_batch ? tid % nof_vecs : tid / vec_size];
    result[tid] = element_vec[tid] + scalar_val;
  }
}

template <typename E>
eIcicleError add_scalar_cuda(
  const Device& device, const E* scalar_a, const E* vec_b, uint64_t size, const VecOpsConfig& config, E* result)
{
  cudaError_t err = vec_scalar_op<E, add_scalar_kernel>(scalar_a, vec_b, config.batch_size, size, config, result, size);
  return translateCudaError(err);
}

/*============================== sub ==============================*/
template <typename E>
__global__ void sub_kernel(const E* element_vec1, const E* element_vec2, uint64_t size, E* result)
{
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) { result[tid] = element_vec1[tid] - element_vec2[tid]; }
}

template <typename E>
eIcicleError
sub_cuda(const Device& device, const E* vec_a, const E* vec_b, uint64_t size, const VecOpsConfig& config, E* result)
{
  cudaError_t err = vec_op<E, E, sub_kernel>(vec_a, vec_b, size, size, config, result, size);
  return translateCudaError(err);
}

template <typename E>
__global__ void sub_scalar_kernel(
  const E* scalar, const E* element_vec, uint64_t vec_size, uint64_t nof_vecs, bool columns_batch, E* result)
{
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < vec_size * nof_vecs) {
    E scalar_val = scalar[columns_batch ? tid % nof_vecs : tid / vec_size];
    result[tid] = scalar_val - element_vec[tid];
  }
}

template <typename E>
eIcicleError sub_scalar_cuda(
  const Device& device, const E* scalar_a, const E* vec_b, uint64_t size, const VecOpsConfig& config, E* result)
{
  cudaError_t err = vec_scalar_op<E, sub_scalar_kernel>(scalar_a, vec_b, config.batch_size, size, config, result, size);
  return translateCudaError(err);
}

/*============================== mul ==============================*/
template <typename E, typename F>
__global__ void mul_kernel(const E* vec_a, const F* vec_b, uint64_t size, E* result)
{
  uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < size) { result[tid] = vec_a[tid] * vec_b[tid]; }
}

template <typename E, typename F>
eIcicleError
mul_cuda(const Device& device, const E* vec_a, const F* vec_b, uint64_t size, const VecOpsConfig& config, E* result)
{
  cudaError_t err = vec_op<E, F, mul_kernel>(vec_a, vec_b, size, size, config, result, size);
  return translateCudaError(err);
}

template <typename E>
__global__ void mul_scalar_kernel(
  const E* scalar, const E* element_vec, uint64_t vec_size, uint64_t nof_vecs, bool columns_batch, E* result)
{
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < vec_size * nof_vecs) {
    E scalar_val = scalar[columns_batch ? tid % nof_vecs : tid / vec_size];
    result[tid] = element_vec[tid] * scalar_val;
  }
}

template <typename E>
eIcicleError mul_scalar_cuda(
  const Device& device, const E* scalar_a, const E* vec_b, uint64_t size, const VecOpsConfig& config, E* result)
{
  cudaError_t err = vec_scalar_op<E, mul_scalar_kernel>(scalar_a, vec_b, config.batch_size, size, config, result, size);
  return translateCudaError(err);
}

/*============================== div ==============================*/
template <typename E>
__global__ void div_element_wise_kernel(const E* element_vec1, const E* element_vec2, uint64_t size, E* result)
{
  // TODO:implement better based on https://eprint.iacr.org/2008/199
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) { result[tid] = element_vec1[tid] * E::inverse(element_vec2[tid]); }
}

template <typename E>
eIcicleError
div_cuda(const Device& device, const E* vec_a, const E* vec_b, uint64_t size, const VecOpsConfig& config, E* result)
{
  cudaError_t err = vec_op<E, E, div_element_wise_kernel>(vec_a, vec_b, size, size, config, result, size);
  return translateCudaError(err);
}

/************************************ REGISTRATION ************************************/

REGISTER_VECTOR_ADD_BACKEND("CUDA", add_cuda<scalar_t>);
REGISTER_VECTOR_ACCUMULATE_BACKEND("CUDA", accumulate_cuda<scalar_t>);
REGISTER_VECTOR_SUB_BACKEND("CUDA", sub_cuda<scalar_t>);
REGISTER_VECTOR_MUL_BACKEND("CUDA", (mul_cuda<scalar_t, scalar_t>));
REGISTER_VECTOR_DIV_BACKEND("CUDA", div_cuda<scalar_t>);
REGISTER_SCALAR_MUL_VEC_BACKEND("CUDA", (mul_scalar_cuda<scalar_t>));
REGISTER_SCALAR_ADD_VEC_BACKEND("CUDA", (add_scalar_cuda<scalar_t>));
REGISTER_SCALAR_SUB_VEC_BACKEND("CUDA", (sub_scalar_cuda<scalar_t>));