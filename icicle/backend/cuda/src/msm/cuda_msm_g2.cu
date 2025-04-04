#include "cuda_msm.cuh"

/************************************** BACKEND REGISTRATION **************************************/

using namespace msm;

// Note: splitting from cuda_msm.cu to compile it in parallel to g1 msm
REGISTER_MSM_G2_BACKEND("CUDA", (msm_cuda_wrapper<scalar_t, g2_affine_t, g2_projective_t>));
REGISTER_MSM_G2_PRE_COMPUTE_BASES_BACKEND(
  "CUDA", (msm_precompute_bases_cuda_wrapper<scalar_t, g2_affine_t, g2_projective_t>));