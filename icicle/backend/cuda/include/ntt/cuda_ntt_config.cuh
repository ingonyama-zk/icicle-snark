#pragma once

#include <stdint.h>
#include "icicle/backend/ntt_config.h"

using namespace CudaBackendConfig;

static inline NttAlgorithm get_ntt_alg_from_config(const icicle::ConfigExtension* ext)
{
  // for some reason this does not compile without this small function. WHY ??
  if (ext && ext->has(CUDA_NTT_ALGORITHM)) { return NttAlgorithm(ext->get<int>(CUDA_NTT_ALGORITHM)); }
  return NttAlgorithm::Auto;
}