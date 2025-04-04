#pragma once

#include <functional>

#include "errors.h"
#include "runtime.h"

#include "icicle/fields/field.h"
#include "icicle/utils/utils.h"
#include "icicle/config_extension.h"
namespace icicle {

  /*************************** Frontend APIs ***************************/
  /**
   * @brief Configuration for vector operations.
   * @note APIs with a single input, ignore input b.
   */
  struct VecOpsConfig {
    icicleStreamHandle stream = nullptr; /** Stream for asynchronous execution. */
    bool is_a_on_device = false;         /** True if `a` is on the device, false if it is not. Default value: false. */
    bool is_b_on_device =
      false; /** True if `b` is on the device, false if it is not. Default value: false. OPTIONAL. */
    bool is_result_on_device = false; /** If true, the output is preserved on the device, otherwise on the host. Default
                                    value: false. */
    bool is_async = false;            /** Whether to run the vector operations asynchronously.
                                    If set to `true`, the function will be non-blocking and synchronization
                                    must be explicitly managed using `cudaStreamSynchronize` or `cudaDeviceSynchronize`.
                                    If set to `false`, the function will block the current CPU thread. */
    int batch_size = 1;               /** Number of vectors (or operations) to process in a batch.
                                        Each vector operation will be performed independently on each batch element.
                                        Default value: 1. */
    bool columns_batch = false; /** True if the batched vectors are stored as columns in a 2D array (i.e., the vectors
                             are strided in memory as columns of a matrix). If false, the batched vectors are stored
                             contiguously in memory (e.g., as rows or in a flat array). Default value: false. */
    ConfigExtension* ext = nullptr; /** Backend-specific extension. */
  };

  /**
   * @brief Returns the default value of VecOpsConfig.
   *
   * @return Default value of VecOpsConfig.
   */
  static VecOpsConfig default_vec_ops_config() { return VecOpsConfig{}; }

  // Element-wise vector operations

  /**
   * @brief Adds two vectors element-wise.
   *
   * @tparam T Type of the elements in the vectors.
   * @param vec_a Pointer to the first input vector(s).
   *              - If `config.batch_size > 1`, this should be a concatenated array of vectors.
   *              - The layout depends on `config.columns_batch`:
   *                - If `false`, vectors are stored contiguously in memory.
   *                - If `true`, vectors are stored as columns in a 2D array.
   * @param vec_b Pointer to the second input vector(s).
   *              - The storage layout should match that of `vec_a`.
   * @param size Number of elements in each vector.
   * @param config Configuration for the operation.
   * @param output Pointer to the output vector(s) where the results will be stored.
   *               The output array should have the same storage layout as the input vectors.
   * @return eIcicleError Error code indicating success or failure.
   */
  template <typename T>
  eIcicleError vector_add(const T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output);

  /**
   * @brief Accumulates the elements of two vectors element-wise and stores the result in the first vector.
   *
   * @tparam T Type of the elements in the vectors.
   * @param vec_a Pointer to the first Input/output vector(s). The result will be written back to this vector.
   *              - If `config.batch_size > 1`, this should be a concatenated array of vectors.
   *              - The layout depends on `config.columns_batch`:
   *                - If `false`, vectors are stored contiguously in memory.
   *                - If `true`, vectors are stored as columns in a 2D array.
   * @param vec_b Pointer to the second input vector(s).
   *              - The storage layout should match that of `vec_a`.
   * @param size Number of elements in each vector.
   * @param config Configuration for the operation.
   * @return eIcicleError Error code indicating success or failure.
   */
  template <typename T>
  eIcicleError
  vector_accumulate(T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config); // use vector_add (inplace)

  /**
   * @brief Subtracts vector `b` from vector `a` element-wise.
   *
   * @tparam T Type of the elements in the vectors.
   * @param vec_a Pointer to the first input vector(s).
   *              - If `config.batch_size > 1`, this should be a concatenated array of vectors.
   *              - The layout depends on `config.columns_batch`:
   *                - If `false`, vectors are stored contiguously in memory.
   *                - If `true`, vectors are stored as columns in a 2D array.
   * @param vec_b Pointer to the second input vector(s).
   *              - The storage layout should match that of `vec_a`.
   * @param size Number of elements in each vector.
   * @param config Configuration for the operation.
   * @param output Pointer to the output vector(s) where the results will be stored.
   *               The output array should have the same storage layout as the input vectors.
   * @return eIcicleError Error code indicating success or failure.
   */
  template <typename T>
  eIcicleError vector_sub(const T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output);

  /**
   * @brief Multiplies two vectors element-wise.
   *
   * @tparam T Type of the elements in the vectors.
   * @param vec_a Pointer to the first input vector(s).
   *              - If `config.batch_size > 1`, this should be a concatenated array of vectors.
   *              - The layout depends on `config.columns_batch`:
   *                - If `false`, vectors are stored contiguously in memory.
   *                - If `true`, vectors are stored as columns in a 2D array.
   * @param vec_b Pointer to the second input vector(s).
   *              - The storage layout should match that of `vec_a`.
   * @param size Number of elements in each vector.
   * @param config Configuration for the operation.
   * @param output Pointer to the output vector(s) where the results will be stored.
   *               The output array should have the same storage layout as the input vectors.
   * @return eIcicleError Error code indicating success or failure.
   */
  template <typename T>
  eIcicleError vector_mul(const T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output);

  /**
   * @brief Multiplies two vectors element-wise.
   *
   * @tparam T Type of the elements in the vectors.
   * @param vec_a Pointer to the first input vector(s).
   *              - If `config.batch_size > 1`, this should be a concatenated array of vectors.
   *              - The layout depends on `config.columns_batch`:
   *                - If `false`, vectors are stored contiguously in memory.
   *                - If `true`, vectors are stored as columns in a 2D array.
   * @param vec_b Pointer to the second input vector(s).
   *              - The storage layout should match that of `vec_a`.
   * @param size Number of elements in each vector.
   * @param config Configuration for the operation.
   * @param output Pointer to the output vector(s) where the results will be stored.
   *               The output array should have the same storage layout as the input vectors.
   * @return eIcicleError Error code indicating success or failure.
   */
  template <typename T, typename U>
  eIcicleError vector_mul(const T* vec_a, const U* vec_b, uint64_t size, const VecOpsConfig& config, T* output);

  /**
   * @brief Divides vector `a` by vector `b` element-wise.
   *
   * @tparam T Type of the elements in the vectors.
   * @param vec_a Pointer to the first input vector(s).
   *              - If `config.batch_size > 1`, this should be a concatenated array of vectors.
   *              - The layout depends on `config.columns_batch`:
   *                - If `false`, vectors are stored contiguously in memory.
   *                - If `true`, vectors are stored as columns in a 2D array.
   * @param vec_b Pointer to the second input vector(s).
   *              - The storage layout should match that of `vec_a`.
   * @param size Number of elements in each vector.
   * @param config Configuration for the operation.
   * @param output Pointer to the output vector(s) where the results will be stored.
   *               The output array should have the same storage layout as the input vectors.
   * @return eIcicleError Error code indicating success or failure.
   */
  template <typename T>
  eIcicleError vector_div(const T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output);

  /**
   * @brief Converts elements to and from Montgomery form.
   *
   * @tparam T Type of the elements.
   * @param input Pointer to the input vector(s).
   *              - If `config.batch_size > 1`, this should be a concatenated array of vectors.
   *              - The layout depends on `config.columns_batch`:
   *                - If `false`, vectors are stored contiguously in memory.
   *                - If `true`, vectors are stored as columns in a 2D array.
   * @param size Number of elements in each vector.
   * @param is_to_montgomery True to convert into Montgomery form, false to convert out of Montgomery form.
   * @param config Configuration for the operation.
   * @param output Pointer to the output vector(s) where the results will be stored.
   *               The output array should have the same storage layout as the input vectors.
   * @return eIcicleError Error code indicating success or failure.
   */
  template <typename T>
  eIcicleError
  convert_montgomery(const T* input, uint64_t size, bool is_to_montgomery, const VecOpsConfig& config, T* output);

  // Reduction operations

  /**
   * @brief Computes the sum of all elements in each vector in a batch.
   *
   * @tparam T Type of the elements in the vector.
   * @param vec_a Pointer to the input vector(s).
   *              - If `config.batch_size > 1`, this should be a concatenated array of vectors.
   *              - The layout depends on `config.columns_batch`:
   *                - If `false`, vectors are stored contiguously.
   *                - If `true`, vectors are stored as columns in a 2D array.
   * @param size Number of elements in each vector.
   * @param config Configuration for the operation.
   * @param output Pointer to the output array where the results will be stored.
   * @return eIcicleError Error code indicating success or failure.
   */

  template <typename T>
  eIcicleError vector_sum(const T* vec_a, uint64_t size, const VecOpsConfig& config, T* output);

  /**
   * @brief Computes the product of all elements in each vector in the batch.
   *
   * @tparam T Type of the elements in the vectors.
   * @param vec_a Pointer to the input vector(s).
   *              - If `config.batch_size > 1`, this should be a concatenated array of vectors.
   *              - The layout depends on `config.columns_batch`:
   *                - If `false`, vectors are stored contiguously.
   *                - If `true`, vectors are stored as columns in a 2D array.
   * @param size Number of elements in each vector.
   * @param config Configuration for the operation.
   * @param output Pointer to the output array where the results will be stored.
   * @return eIcicleError Error code indicating success or failure.
   */

  template <typename T>
  eIcicleError vector_product(const T* vec_a, uint64_t size, const VecOpsConfig& config, T* output);

  // Scalar-Vector operations

  /**
   * @brief Adds a scalar to each element of a vector.
   *
   * @tparam T Type of the elements in the vector and the scalar.
   * @param scalar_a Pointer to the input scalar(s).
   * @param vec_b Pointer to the input vector(s).
   *              - If `config.batch_size > 1`, this should be a concatenated array of vectors.
   *              - The layout depends on `config.columns_batch`:
   *                - If `false`, vectors are stored contiguously.
   *                - If `true`, vectors are stored as columns in a 2D array.
   * @param size Number of elements in a vector.
   * @param config Configuration for the operation.
   * @param output Pointer to the output vector(s) where the results will be stored.
   * @return eIcicleError Error code indicating success or failure.
   * @note To subtract a scalar from each element of a vector - use scalar_add_vec with negative scalar.
   */
  template <typename T>
  eIcicleError scalar_add_vec(const T* scalar_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output);

  /**
   * @brief Subtracts each element of a vector from a scalar, elementwise (res[i]=scalar-vec[i]).
   *
   * @tparam T Type of the elements in the vector and the scalar.
   * @param scalar_a Pointer to Input scalar(s).
   * @param vec_b Pointer to the input vector(s).
   *              - If `config.batch_size > 1`, this should be a concatenated array of vectors.
   *              - The layout depends on `config.columns_batch`:
   *                - If `false`, vectors are stored contiguously.
   *                - If `true`, vectors are stored as columns in a 2D array.
   * @param size Number of elements in a vector.
   * @param config Configuration for the operation.
   * @param output Pointer to the output vector(s) where the results will be stored.
   * @return eIcicleError Error code indicating success or failure.
   * @note To subtract a scalar from each element of a vector - use scalar_add_vec with negative scalar.
   */
  template <typename T>
  eIcicleError scalar_sub_vec(const T* scalar_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output);

  /**
   * @brief Multiplies each element of a vector by a scalar.
   *
   * @tparam T Type of the elements in the vector and the scalar.
   * @param scalar_a Pointer to Input scalar(s).
   * @param vec_b Pointer to the input vector(s).
   *              - If `config.batch_size > 1`, this should be a concatenated array of vectors.
   *              - The layout depends on `config.columns_batch`:
   *                - If `false`, vectors are stored contiguously.
   *                - If `true`, vectors are stored as columns in a 2D array.
   * @param size Number of elements in a vector.
   * @param config Configuration for the operation.
   * @param output Pointer to the output vector(s) where the results will be stored.
   * @return eIcicleError Error code indicating success or failure.
   */
  template <typename T>
  eIcicleError scalar_mul_vec(const T* scalar_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output);

} // namespace icicle