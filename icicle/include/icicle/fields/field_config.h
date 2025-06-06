#pragma once

#include "icicle/fields/id.h"
#include "icicle/fields/field.h"

/**
 * @namespace field_config
 * Namespace with type definitions for finite fields. Here, concrete types are created in accordance
 * with the `-DFIELD` env variable passed during build.
 */
#if FIELD_ID == BN254
  #include "icicle/fields/snark_fields/bn254_scalar.h"
namespace field_config = bn254;
#elif FIELD_ID == BLS12_381
  #include "icicle/fields/snark_fields/bls12_381_scalar.h"
using bls12_381::fp_config;
namespace field_config = bls12_381;
#elif FIELD_ID == BLS12_377
  #include "icicle/fields/snark_fields/bls12_377_scalar.h"
namespace field_config = bls12_377;
#elif FIELD_ID == BW6_761
  #include "icicle/fields/snark_fields/bw6_761_scalar.h"
namespace field_config = bw6_761;
#endif
