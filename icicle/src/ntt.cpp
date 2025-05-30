#include "icicle/backend/ntt_backend.h"
#include "icicle/dispatcher.h"

using namespace field_config;
namespace icicle {

  /*************************** NTT ***************************/
  ICICLE_DISPATCHER_INST(NttDispatcher, ntt, NttImpl);

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, ntt)(
    const scalar_t* input, int size, NTTDir dir, const NTTConfig<scalar_t>* config, scalar_t* output)
  {
    return NttDispatcher::execute(input, size, dir, *config, output);
  }

  template <>
  eIcicleError ntt(const scalar_t* input, int size, NTTDir dir, const NTTConfig<scalar_t>& config, scalar_t* output)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, ntt)(input, size, dir, &config, output);
  }

  /*************************** INIT DOMAIN ***************************/
  ICICLE_DISPATCHER_INST(NttInitDomainDispatcher, ntt_init_domain, NttInitDomainImpl);

  extern "C" eIcicleError
  CONCAT_EXPAND(ICICLE_FFI_PREFIX, ntt_init_domain)(const scalar_t* primitive_root, const NTTInitDomainConfig* config)
  {
    return NttInitDomainDispatcher::execute(*primitive_root, *config);
  }

  template <>
  eIcicleError ntt_init_domain(const scalar_t& primitive_root, const NTTInitDomainConfig& config)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, ntt_init_domain)(&primitive_root, &config);
  }

  /*************************** RELEASE DOMAIN ***************************/
  ICICLE_DISPATCHER_INST(NttReleaseDomainDispatcher, ntt_release_domain, NttReleaseDomainImpl);

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, ntt_release_domain)()
  {
    // Note: passing zero is a workaround for the function required per field but need to differentiate by type when
    // calling
    return NttReleaseDomainDispatcher::execute(scalar_t::zero());
  }

  template <>
  eIcicleError ntt_release_domain<scalar_t>()
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, ntt_release_domain)();
  }

  /*************************** GET ROOT OF UNITY ***************************/
  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, get_root_of_unity)(uint64_t max_size, scalar_t* rou)
  {
    const auto log_max_size = static_cast<uint32_t>(std::ceil(std::log2(max_size)));
    if (scalar_t::get_omegas_count() < log_max_size) {
      ICICLE_LOG_ERROR << "no root-of-unity of order " << log_max_size << " in field " << typeid(scalar_t).name();
      return eIcicleError::INVALID_ARGUMENT;
    }
    *rou = scalar_t::omega(log_max_size);
    return eIcicleError::SUCCESS;
  }

  template <>
  eIcicleError get_root_of_unity(uint64_t max_size, scalar_t* rou)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, get_root_of_unity)(max_size, rou);
  }

  /*************************** GET ROOT OF UNITY FROM DOMAIN ***************************/
  ICICLE_DISPATCHER_INST(NttRouFromDomainDispatcher, ntt_get_rou_from_domain, NttGetRouFromDomainImpl);

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, get_root_of_unity_from_domain)(uint64_t logn, scalar_t* rou)
  {
    return NttRouFromDomainDispatcher::execute(logn, rou);
  }

  template <>
  eIcicleError get_root_of_unity_from_domain<scalar_t>(uint64_t logn, scalar_t* rou)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, get_root_of_unity_from_domain)(logn, rou);
  }
} // namespace icicle