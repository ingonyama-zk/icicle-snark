#include <cuda.h>
#include <stdexcept>

#include "icicle/errors.h"
#include "icicle/backend/vec_ops_backend.h"
#include "gpu-utils/error_handler.h"
#include "error_translation.h"
#include "cuda_mont.cuh"

namespace icicle {

#include "icicle/fields/field_config.h"
  using namespace field_config;

  template <typename F>
  eIcicleError convert_montgomery_cuda(
    const Device& device, const F* input, uint64_t n, bool is_into, const VecOpsConfig& config, F* output)
  {
    n *= config.batch_size;
    auto err = is_into ? montgomery::ConvertMontgomery<F, true>(input, n, config, output)
                       : montgomery::ConvertMontgomery<F, false>(input, n, config, output);
    return translateCudaError(err);
  }

  REGISTER_CONVERT_MONTGOMERY_BACKEND("CUDA", convert_montgomery_cuda<scalar_t>);

} // namespace icicle
