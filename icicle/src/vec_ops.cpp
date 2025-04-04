#include "icicle/backend/vec_ops_backend.h"
#include "icicle/dispatcher.h"

namespace icicle {

  /*********************************** REDUCE PRODUCT ************************/
  ICICLE_DISPATCHER_INST(VectorProductDispatcher, vector_product, VectorReduceOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, vector_product)(
    const scalar_t* vec_a, uint64_t size, const VecOpsConfig* config, scalar_t* output)
  {
    return VectorProductDispatcher::execute(vec_a, size, *config, output);
  }

  template <>
  eIcicleError vector_product(const scalar_t* vec_a, uint64_t size, const VecOpsConfig& config, scalar_t* output)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, vector_product)(vec_a, size, &config, output);
  }

  /*********************************** REDUCE SUM ****************************/
  ICICLE_DISPATCHER_INST(VectorSumDispatcher, vector_sum, VectorReduceOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, vector_sum)(
    const scalar_t* vec_a, uint64_t size, const VecOpsConfig* config, scalar_t* output)
  {
    return VectorSumDispatcher::execute(vec_a, size, *config, output);
  }

  template <>
  eIcicleError vector_sum(const scalar_t* vec_a, uint64_t size, const VecOpsConfig& config, scalar_t* output)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, vector_sum)(vec_a, size, &config, output);
  }

  /*********************************** ADD ***********************************/
  ICICLE_DISPATCHER_INST(VectorAddDispatcher, vector_add, scalarVectorOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, vector_add)(
    const scalar_t* vec_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig* config, scalar_t* output)
  {
    return VectorAddDispatcher::execute(vec_a, vec_b, size, *config, output);
  }

  template <>
  eIcicleError
  vector_add(const scalar_t* vec_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig& config, scalar_t* output)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, vector_add)(vec_a, vec_b, size, &config, output);
  }

  /*********************************** ACCUMULATE ***********************************/
  ICICLE_DISPATCHER_INST(VectorAccumulateDispatcher, vector_accumulate, vectorVectorOpImplInplaceA);

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, vector_accumulate)(
    scalar_t* vec_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig* config)
  {
    return VectorAccumulateDispatcher::execute(vec_a, vec_b, size, *config);
  }

  template <>
  eIcicleError vector_accumulate(scalar_t* vec_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig& config)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, vector_accumulate)(vec_a, vec_b, size, &config);
  }

  /*********************************** SUB ***********************************/
  ICICLE_DISPATCHER_INST(VectorSubDispatcher, vector_sub, scalarVectorOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, vector_sub)(
    const scalar_t* vec_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig* config, scalar_t* output)
  {
    return VectorSubDispatcher::execute(vec_a, vec_b, size, *config, output);
  }

  template <>
  eIcicleError
  vector_sub(const scalar_t* vec_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig& config, scalar_t* output)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, vector_sub)(vec_a, vec_b, size, &config, output);
  }

  /*********************************** MUL ***********************************/
  ICICLE_DISPATCHER_INST(VectorMulDispatcher, vector_mul, scalarVectorOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, vector_mul)(
    const scalar_t* vec_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig* config, scalar_t* output)
  {
    return VectorMulDispatcher::execute(vec_a, vec_b, size, *config, output);
  }

  template <>
  eIcicleError
  vector_mul(const scalar_t* vec_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig& config, scalar_t* output)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, vector_mul)(vec_a, vec_b, size, &config, output);
  }

  /*********************************** DIV ***********************************/
  ICICLE_DISPATCHER_INST(VectorDivDispatcher, vector_div, scalarVectorOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, vector_div)(
    const scalar_t* vec_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig* config, scalar_t* output)
  {
    return VectorDivDispatcher::execute(vec_a, vec_b, size, *config, output);
  }

  template <>
  eIcicleError
  vector_div(const scalar_t* vec_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig& config, scalar_t* output)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, vector_div)(vec_a, vec_b, size, &config, output);
  }

  /*********************************** (Scalar + Vector) ELEMENT WISE ***********************************/
  ICICLE_DISPATCHER_INST(ScalarAddDispatcher, scalar_add_vec, scalarVectorOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, scalar_add_vec)(
    const scalar_t* scalar_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig* config, scalar_t* output)
  {
    return ScalarAddDispatcher::execute(scalar_a, vec_b, size, *config, output);
  }

  template <>
  eIcicleError scalar_add_vec(
    const scalar_t* scalar_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig& config, scalar_t* output)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, scalar_add_vec)(scalar_a, vec_b, size, &config, output);
  }

  /*********************************** (Scalar - Vector) ELEMENT WISE ***********************************/
  ICICLE_DISPATCHER_INST(ScalarSubDispatcher, scalar_sub_vec, scalarVectorOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, scalar_sub_vec)(
    const scalar_t* scalar_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig* config, scalar_t* output)
  {
    return ScalarSubDispatcher::execute(scalar_a, vec_b, size, *config, output);
  }

  template <>
  eIcicleError scalar_sub_vec(
    const scalar_t* scalar_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig& config, scalar_t* output)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, scalar_sub_vec)(scalar_a, vec_b, size, &config, output);
  }

  /*********************************** MUL BY SCALAR ***********************************/
  ICICLE_DISPATCHER_INST(ScalarMulDispatcher, scalar_mul_vec, scalarVectorOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, scalar_mul_vec)(
    const scalar_t* scalar_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig* config, scalar_t* output)
  {
    return ScalarMulDispatcher::execute(scalar_a, vec_b, size, *config, output);
  }

  template <>
  eIcicleError scalar_mul_vec(
    const scalar_t* scalar_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig& config, scalar_t* output)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, scalar_mul_vec)(scalar_a, vec_b, size, &config, output);
  }

  /*********************************** CONVERT MONTGOMERY ***********************************/

  ICICLE_DISPATCHER_INST(ScalarConvertMontgomeryDispatcher, scalar_convert_montgomery, scalarConvertMontgomeryImpl)

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, scalar_convert_montgomery)(
    const scalar_t* input, uint64_t size, bool is_to_montgomery, const VecOpsConfig* config, scalar_t* output)
  {
    return ScalarConvertMontgomeryDispatcher::execute(input, size, is_to_montgomery, *config, output);
  }

  template <>
  eIcicleError convert_montgomery(
    const scalar_t* input, uint64_t size, bool is_to_montgomery, const VecOpsConfig& config, scalar_t* output)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, scalar_convert_montgomery)(input, size, is_to_montgomery, &config, output);
  }
} // namespace icicle