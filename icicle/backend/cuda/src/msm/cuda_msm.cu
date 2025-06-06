#include "cuda_msm.cuh"

/************************************** BACKEND REGISTRATION **************************************/

using namespace msm;
using namespace icicle;

REGISTER_MSM_BACKEND("CUDA", (msm_cuda_wrapper<scalar_t, affine_t, projective_t>));
REGISTER_MSM_PRE_COMPUTE_BASES_BACKEND("CUDA", (msm_precompute_bases_cuda_wrapper<scalar_t, affine_t, projective_t>));
