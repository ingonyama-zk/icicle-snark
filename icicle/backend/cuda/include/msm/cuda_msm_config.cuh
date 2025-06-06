#pragma once

#include <stdint.h>
#include "icicle/config_extension.h"
#include "icicle/backend/msm_config.h"

using namespace CudaBackendConfig;

static inline bool is_big_triangle(const icicle::ConfigExtension* ext)
{
  return ext && ext->has(CUDA_MSM_IS_BIG_TRIANGLE) ? ext->get<bool>(CUDA_MSM_IS_BIG_TRIANGLE) : false;
}

static inline int get_large_bucket_factor(const icicle::ConfigExtension* ext)
{
  return ext && ext->has(CUDA_MSM_LARGE_BUCKET_FACTOR) ? ext->get<int>(CUDA_MSM_LARGE_BUCKET_FACTOR)
                                                       : CUDA_MSM_LARGE_BUCKET_FACTOR_DEFAULT_VAL;
}

static inline int get_nof_chunks(const icicle::ConfigExtension* ext)
{
  return ext && ext->has(CUDA_MSM_NOF_CHUNKS) ? ext->get<int>(CUDA_MSM_NOF_CHUNKS) : CUDA_MSM_NOF_CHUNKS_DEFAULT_VAL;
}
