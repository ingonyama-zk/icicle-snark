#include "icicle/utils/utils.h"
#include "icicle/fields/field_config.h"

using namespace field_config;

extern "C" void CONCAT_EXPAND(ICICLE_FFI_PREFIX, generate_scalars)(scalar_t* scalars, int size)
{
  scalar_t::rand_host_many(scalars, size);
}

extern "C" void CONCAT_EXPAND(ICICLE_FFI_PREFIX, sub)(scalar_t* scalar1, scalar_t* scalar2, scalar_t* result)
{
  *result = *scalar1 - *scalar2;
}

extern "C" void CONCAT_EXPAND(ICICLE_FFI_PREFIX, add)(scalar_t* scalar1, scalar_t* scalar2, scalar_t* result)
{
  *result = *scalar1 + *scalar2;
}

extern "C" void CONCAT_EXPAND(ICICLE_FFI_PREFIX, mul)(scalar_t* scalar1, scalar_t* scalar2, scalar_t* result)
{
  *result = *scalar1 * *scalar2;
}

extern "C" void CONCAT_EXPAND(ICICLE_FFI_PREFIX, inv)(scalar_t* scalar1, scalar_t* result)
{
  *result = scalar_t::inverse(*scalar1);
}

extern "C" void CONCAT_EXPAND(ICICLE_FFI_PREFIX, pow)(scalar_t* base, int exp, scalar_t* result)
{
  *result = scalar_t::pow(*base, exp);
}

extern "C" void CONCAT_EXPAND(ICICLE_FFI_PREFIX, from_u32)(uint32_t val, scalar_t* result)
{
  *result = scalar_t::from(val);
}
