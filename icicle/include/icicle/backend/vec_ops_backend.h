#pragma once

#include "icicle/vec_ops.h"
#include "icicle/fields/field_config.h"
using namespace field_config;

namespace icicle {
  /*************************** Backend registration ***************************/

  using vectorVectorOpImplInplaceA = std::function<eIcicleError(
    const Device& device, scalar_t* vec_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig& config)>;

  using scalarConvertMontgomeryImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* input,
    uint64_t size,
    bool is_to_montgomery,
    const VecOpsConfig& config,
    scalar_t* output)>;

  using VectorReduceOpImpl = std::function<eIcicleError(
    const Device& device, const scalar_t* vec_a, uint64_t size, const VecOpsConfig& config, scalar_t* output)>;

  using scalarVectorOpImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* scalar_a,
    const scalar_t* vec_b,
    uint64_t size,
    const VecOpsConfig& config,
    scalar_t* output)>;

  using scalarBitReverseOpImpl = std::function<eIcicleError(
    const Device& device, const scalar_t* input, uint64_t size, const VecOpsConfig& config, scalar_t* output)>;

  using scalarSliceOpImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* input,
    uint64_t offset,
    uint64_t stride,
    uint64_t size_in,
    uint64_t size_out,
    const VecOpsConfig& config,
    scalar_t* output)>;

  using scalarHighNonZeroIdxOpImpl = std::function<eIcicleError(
    const Device& device, const scalar_t* input, uint64_t size, const VecOpsConfig& config, int64_t* out_idx)>;

  using scalarPolyEvalImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* coeffs,
    uint64_t coeffs_size,
    const scalar_t* domain,
    uint64_t domain_size,
    const VecOpsConfig& config,
    scalar_t* evals /*OUT*/)>;

  using scalarPolyDivImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* numerator,
    uint64_t numerator_size,
    const scalar_t* denominator,
    uint64_t denominator_size,
    const VecOpsConfig& config,
    scalar_t* q_out /*OUT*/,
    uint64_t q_size,
    scalar_t* r_out /*OUT*/,
    uint64_t r_size)>;

  void register_vector_add(const std::string& deviceType, scalarVectorOpImpl impl);

#define REGISTER_VECTOR_ADD_BACKEND(DEVICE_TYPE, FUNC)                                                                 \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_vec_add) = []() -> bool {                                                                  \
      register_vector_add(DEVICE_TYPE, FUNC);                                                                          \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_vector_accumulate(const std::string& deviceType, vectorVectorOpImplInplaceA impl);

#define REGISTER_VECTOR_ACCUMULATE_BACKEND(DEVICE_TYPE, FUNC)                                                          \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_vec_accumulate) = []() -> bool {                                                           \
      register_vector_accumulate(DEVICE_TYPE, FUNC);                                                                   \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_vector_sub(const std::string& deviceType, scalarVectorOpImpl impl);
#define REGISTER_VECTOR_SUB_BACKEND(DEVICE_TYPE, FUNC)                                                                 \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_vec_sub) = []() -> bool {                                                                  \
      register_vector_sub(DEVICE_TYPE, FUNC);                                                                          \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_vector_mul(const std::string& deviceType, scalarVectorOpImpl impl);

#define REGISTER_VECTOR_MUL_BACKEND(DEVICE_TYPE, FUNC)                                                                 \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_vec_mul) = []() -> bool {                                                                  \
      register_vector_mul(DEVICE_TYPE, FUNC);                                                                          \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_vector_div(const std::string& deviceType, scalarVectorOpImpl impl);

#define REGISTER_VECTOR_DIV_BACKEND(DEVICE_TYPE, FUNC)                                                                 \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_vec_div) = []() -> bool {                                                                  \
      register_vector_div(DEVICE_TYPE, FUNC);                                                                          \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_scalar_convert_montgomery(const std::string& deviceType, scalarConvertMontgomeryImpl);

#define REGISTER_CONVERT_MONTGOMERY_BACKEND(DEVICE_TYPE, FUNC)                                                         \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_scalar_convert_mont) = []() -> bool {                                                      \
      register_scalar_convert_montgomery(DEVICE_TYPE, FUNC);                                                           \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_vector_sum(const std::string& deviceType, VectorReduceOpImpl impl);

#define REGISTER_VECTOR_SUM_BACKEND(DEVICE_TYPE, FUNC)                                                                 \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_vec_sum) = []() -> bool {                                                                  \
      register_vector_sum(DEVICE_TYPE, FUNC);                                                                          \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_vector_product(const std::string& deviceType, VectorReduceOpImpl impl);

#define REGISTER_VECTOR_PRODUCT_BACKEND(DEVICE_TYPE, FUNC)                                                             \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_vec_product) = []() -> bool {                                                              \
      register_vector_product(DEVICE_TYPE, FUNC);                                                                      \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_scalar_mul_vec(const std::string& deviceType, scalarVectorOpImpl impl);

#define REGISTER_SCALAR_MUL_VEC_BACKEND(DEVICE_TYPE, FUNC)                                                             \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_scalar_mul_vec) = []() -> bool {                                                           \
      register_scalar_mul_vec(DEVICE_TYPE, FUNC);                                                                      \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_scalar_add_vec(const std::string& deviceType, scalarVectorOpImpl impl);

#define REGISTER_SCALAR_ADD_VEC_BACKEND(DEVICE_TYPE, FUNC)                                                             \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_scalar_add_vec) = []() -> bool {                                                           \
      register_scalar_add_vec(DEVICE_TYPE, FUNC);                                                                      \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_scalar_sub_vec(const std::string& deviceType, scalarVectorOpImpl impl);

#define REGISTER_SCALAR_SUB_VEC_BACKEND(DEVICE_TYPE, FUNC)                                                             \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_scalar_sub_vec) = []() -> bool {                                                           \
      register_scalar_sub_vec(DEVICE_TYPE, FUNC);                                                                      \
      return true;                                                                                                     \
    }();                                                                                                               \
  }
} // namespace icicle