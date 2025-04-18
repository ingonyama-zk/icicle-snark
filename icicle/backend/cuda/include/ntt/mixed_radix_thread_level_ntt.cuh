#pragma once

#include <stdio.h>
#include <stdint.h>
#include "icicle/utils/modifiers.h"

struct stage_metadata {
  uint32_t th_stride;
  uint32_t ntt_block_size;
  uint32_t batch_id;
  uint32_t ntt_block_id;
  uint32_t ntt_inp_id;
};

#define STAGE_SIZES_DATA                                                                                               \
  {                                                                                                                    \
    {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {4, 0, 0, 0, 0}, {5, 0, 0, 0, 0},              \
    {6, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {4, 4, 0, 0, 0}, {5, 4, 0, 0, 0}, {5, 5, 0, 0, 0}, {6, 5, 0, 0, 0},              \
    {6, 6, 0, 0, 0}, {4, 5, 4, 0, 0}, {4, 6, 4, 0, 0}, {5, 5, 5, 0, 0}, {6, 4, 6, 0, 0}, {6, 5, 6, 0, 0},              \
    {6, 6, 6, 0, 0}, {6, 5, 4, 4, 0}, {5, 5, 5, 5, 0}, {6, 5, 5, 5, 0}, {6, 5, 5, 6, 0}, {6, 6, 6, 5, 0},              \
    {6, 6, 6, 6, 0}, {5, 5, 5, 5, 5}, {6, 5, 4, 5, 6}, {6, 5, 5, 5, 6}, {6, 5, 6, 5, 6}, {6, 6, 5, 6, 6},              \
    {6, 6, 6, 6, 6},                                                                                                   \
  }
uint32_t constexpr STAGE_SIZES_HOST[31][5] = STAGE_SIZES_DATA;
__device__ constexpr uint32_t STAGE_SIZES_DEVICE[31][5] = STAGE_SIZES_DATA;

// construction for fast-twiddles
uint32_t constexpr STAGE_PREV_SIZES[31] = {0,  0,  0,  0,  0,  0,  0,  0,  4,  5,  5,  6,  6,  9,  9, 10,
                                           11, 11, 12, 15, 15, 16, 16, 18, 18, 20, 21, 21, 22, 23, 24};

#define STAGE_SIZES_DATA_FAST_TW                                                                                       \
  {                                                                                                                    \
    {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {4, 0, 0, 0, 0}, {5, 0, 0, 0, 0},              \
    {6, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {4, 4, 0, 0, 0}, {5, 4, 0, 0, 0}, {5, 5, 0, 0, 0}, {6, 5, 0, 0, 0},              \
    {6, 6, 0, 0, 0}, {5, 4, 4, 0, 0}, {5, 4, 5, 0, 0}, {5, 5, 5, 0, 0}, {6, 5, 5, 0, 0}, {6, 5, 6, 0, 0},              \
    {6, 6, 6, 0, 0}, {5, 5, 5, 4, 0}, {5, 5, 5, 5, 0}, {6, 5, 5, 5, 0}, {6, 5, 5, 6, 0}, {6, 6, 6, 5, 0},              \
    {6, 6, 6, 6, 0}, {5, 5, 5, 5, 5}, {6, 5, 5, 5, 5}, {6, 5, 5, 5, 6}, {6, 5, 5, 6, 6}, {6, 6, 6, 5, 6},              \
    {6, 6, 6, 6, 6},                                                                                                   \
  }
uint32_t constexpr STAGE_SIZES_HOST_FT[31][5] = STAGE_SIZES_DATA_FAST_TW;
__device__ uint32_t constexpr STAGE_SIZES_DEVICE_FT[31][5] = STAGE_SIZES_DATA_FAST_TW;

template <typename E, typename S>
class NTTEngine
{
public:
  E X[8];
  S WB[3];
  S WI[7];
  S WE[8];

  DEVICE_INLINE void loadBasicTwiddles(S* basic_twiddles)
  {
    UNROLL
    for (int i = 0; i < 3; i++) {
      WB[i] = basic_twiddles[i];
    }
  }

  DEVICE_INLINE void loadBasicTwiddlesGeneric(S* basic_twiddles, bool inv)
  {
    UNROLL
    for (int i = 0; i < 3; i++) {
      WB[i] = basic_twiddles[inv ? i + 3 : i];
    }
  }

  DEVICE_INLINE void loadInternalTwiddles64(S* data, bool stride)
  {
    UNROLL
    for (int i = 0; i < 7; i++) {
      WI[i] = data[((stride ? (threadIdx.x >> 3) : (threadIdx.x)) & 0x7) * (i + 1)];
    }
  }

  DEVICE_INLINE void loadInternalTwiddles32(S* data, bool stride)
  {
    UNROLL
    for (int i = 0; i < 7; i++) {
      WI[i] = data[2 * ((stride ? (threadIdx.x >> 4) : (threadIdx.x)) & 0x3) * (i + 1)];
    }
  }

  DEVICE_INLINE void loadInternalTwiddles16(S* data, bool stride)
  {
    UNROLL
    for (int i = 0; i < 7; i++) {
      WI[i] = data[4 * ((stride ? (threadIdx.x >> 5) : (threadIdx.x)) & 0x1) * (i + 1)];
    }
  }

  DEVICE_INLINE void loadInternalTwiddlesGeneric64(S* data, bool stride, bool inv)
  {
    UNROLL
    for (int i = 0; i < 7; i++) {
      uint32_t exp = ((stride ? (threadIdx.x >> 3) : (threadIdx.x)) & 0x7) * (i + 1);
      WI[i] = data[(inv && exp) ? 64 - exp : exp]; // if exp = 0 we also take exp and not 64-exp
    }
  }

  DEVICE_INLINE void loadInternalTwiddlesGeneric32(S* data, bool stride, bool inv)
  {
    UNROLL
    for (int i = 0; i < 7; i++) {
      uint32_t exp = 2 * ((stride ? (threadIdx.x >> 4) : (threadIdx.x)) & 0x3) * (i + 1);
      WI[i] = data[(inv && exp) ? 64 - exp : exp];
    }
  }

  DEVICE_INLINE void loadInternalTwiddlesGeneric16(S* data, bool stride, bool inv)
  {
    UNROLL
    for (int i = 0; i < 7; i++) {
      uint32_t exp = 4 * ((stride ? (threadIdx.x >> 5) : (threadIdx.x)) & 0x1) * (i + 1);
      WI[i] = data[(inv && exp) ? 64 - exp : exp];
    }
  }

  DEVICE_INLINE void loadExternalTwiddles64(S* data, uint32_t tw_order, uint32_t tw_log_order, stage_metadata s_meta)
  {
    data += tw_order * s_meta.ntt_inp_id + (s_meta.ntt_block_id & (tw_order - 1));

    UNROLL
    for (uint32_t i = 0; i < 8; i++) {
      WE[i] = data[8 * i * tw_order + (1 << tw_log_order + 6) - 1];
    }
  }

  DEVICE_INLINE void loadExternalTwiddles32(S* data, uint32_t tw_order, uint32_t tw_log_order, stage_metadata s_meta)
  {
    data += tw_order * s_meta.ntt_inp_id * 2 + (s_meta.ntt_block_id & (tw_order - 1));

    UNROLL
    for (uint32_t j = 0; j < 2; j++) {
      UNROLL
      for (uint32_t i = 0; i < 4; i++) {
        WE[4 * j + i] = data[(8 * i + j) * tw_order + (1 << tw_log_order + 5) - 1];
      }
    }
  }

  DEVICE_INLINE void loadExternalTwiddles16(S* data, uint32_t tw_order, uint32_t tw_log_order, stage_metadata s_meta)
  {
    data += tw_order * s_meta.ntt_inp_id * 4 + (s_meta.ntt_block_id & (tw_order - 1));

    UNROLL
    for (uint32_t j = 0; j < 4; j++) {
      UNROLL
      for (uint32_t i = 0; i < 2; i++) {
        WE[2 * j + i] = data[(8 * i + j) * tw_order + (1 << tw_log_order + 4) - 1];
      }
    }
  }

  DEVICE_INLINE void loadExternalTwiddlesGeneric64(
    S* data, uint32_t tw_order, uint32_t tw_log_order, stage_metadata s_meta, uint32_t tw_log_size, bool inv)
  {
    UNROLL
    for (uint32_t i = 0; i < 8; i++) {
      uint32_t exp = (s_meta.ntt_inp_id + 8 * i) * (s_meta.ntt_block_id & (tw_order - 1))
                     << (tw_log_size - tw_log_order - 6);
      WE[i] = data[(inv && exp) ? ((1 << tw_log_size) - exp) : exp];
    }
  }

  DEVICE_INLINE void loadExternalTwiddlesGeneric32(
    S* data, uint32_t tw_order, uint32_t tw_log_order, stage_metadata s_meta, uint32_t tw_log_size, bool inv)
  {
    UNROLL
    for (uint32_t j = 0; j < 2; j++) {
      UNROLL
      for (uint32_t i = 0; i < 4; i++) {
        uint32_t exp = (s_meta.ntt_inp_id * 2 + 8 * i + j) * (s_meta.ntt_block_id & (tw_order - 1))
                       << (tw_log_size - tw_log_order - 5);
        WE[4 * j + i] = data[(inv && exp) ? ((1 << tw_log_size) - exp) : exp];
      }
    }
  }

  DEVICE_INLINE void loadExternalTwiddlesGeneric16(
    S* data, uint32_t tw_order, uint32_t tw_log_order, stage_metadata s_meta, uint32_t tw_log_size, bool inv)
  {
    UNROLL
    for (uint32_t j = 0; j < 4; j++) {
      UNROLL
      for (uint32_t i = 0; i < 2; i++) {
        uint32_t exp = (s_meta.ntt_inp_id * 4 + 8 * i + j) * (s_meta.ntt_block_id & (tw_order - 1))
                       << (tw_log_size - tw_log_order - 4);
        WE[2 * j + i] = data[(inv && exp) ? ((1 << tw_log_size) - exp) : exp];
      }
    }
  }

  DEVICE_INLINE void
  loadGlobalData(const E* data, uint32_t data_stride, uint32_t log_data_stride, bool strided, stage_metadata s_meta)
  {
    const uint64_t data_stride_u64 = data_stride;
    if (strided) {
      data += (s_meta.ntt_block_id & (data_stride - 1)) + data_stride_u64 * s_meta.ntt_inp_id +
              (s_meta.ntt_block_id >> log_data_stride) * data_stride_u64 * s_meta.ntt_block_size;
    } else {
      data += (uint64_t)s_meta.ntt_block_id * s_meta.ntt_block_size + s_meta.ntt_inp_id;
    }

    UNROLL
    for (uint32_t i = 0; i < 8; i++) {
      X[i] = data[s_meta.th_stride * i * data_stride_u64];
    }
  }

  DEVICE_INLINE void loadGlobalDataColumnBatch(
    const E* data, uint32_t data_stride, uint32_t log_data_stride, stage_metadata s_meta, uint32_t batch_size)
  {
    const uint64_t data_stride_u64 = data_stride;
    data += ((s_meta.ntt_block_id & (data_stride - 1)) + data_stride_u64 * s_meta.ntt_inp_id +
             (s_meta.ntt_block_id >> log_data_stride) * data_stride_u64 * s_meta.ntt_block_size) *
              batch_size +
            s_meta.batch_id;

    UNROLL
    for (uint32_t i = 0; i < 8; i++) {
      X[i] = data[s_meta.th_stride * i * data_stride_u64 * batch_size];
    }
  }

  DEVICE_INLINE void
  storeGlobalData(E* data, uint32_t data_stride, uint32_t log_data_stride, bool strided, stage_metadata s_meta)
  {
    const uint64_t data_stride_u64 = data_stride;
    if (strided) {
      data += (s_meta.ntt_block_id & (data_stride - 1)) + data_stride_u64 * s_meta.ntt_inp_id +
              (s_meta.ntt_block_id >> log_data_stride) * data_stride_u64 * s_meta.ntt_block_size;
    } else {
      data += (uint64_t)s_meta.ntt_block_id * s_meta.ntt_block_size + s_meta.ntt_inp_id;
    }

    UNROLL
    for (uint32_t i = 0; i < 8; i++) {
      data[s_meta.th_stride * i * data_stride_u64] = X[i];
    }
  }

  DEVICE_INLINE void storeGlobalDataColumnBatch(
    E* data, uint32_t data_stride, uint32_t log_data_stride, stage_metadata s_meta, uint32_t batch_size)
  {
    const uint64_t data_stride_u64 = data_stride;
    data += ((s_meta.ntt_block_id & (data_stride - 1)) + data_stride_u64 * s_meta.ntt_inp_id +
             (s_meta.ntt_block_id >> log_data_stride) * data_stride_u64 * s_meta.ntt_block_size) *
              batch_size +
            s_meta.batch_id;

    UNROLL
    for (uint32_t i = 0; i < 8; i++) {
      data[s_meta.th_stride * i * data_stride_u64 * batch_size] = X[i];
    }
  }

  DEVICE_INLINE void
  loadGlobalData32(const E* data, uint32_t data_stride, uint32_t log_data_stride, bool strided, stage_metadata s_meta)
  {
    const uint64_t data_stride_u64 = data_stride;
    if (strided) {
      data += (s_meta.ntt_block_id & (data_stride - 1)) + data_stride_u64 * s_meta.ntt_inp_id * 2 +
              (s_meta.ntt_block_id >> log_data_stride) * data_stride_u64 * s_meta.ntt_block_size;
    } else {
      data += (uint64_t)s_meta.ntt_block_id * s_meta.ntt_block_size + s_meta.ntt_inp_id * 2;
    }

    UNROLL
    for (uint32_t j = 0; j < 2; j++) {
      UNROLL
      for (uint32_t i = 0; i < 4; i++) {
        X[4 * j + i] = data[(8 * i + j) * data_stride_u64];
      }
    }
  }

  DEVICE_INLINE void loadGlobalData32ColumnBatch(
    const E* data, uint32_t data_stride, uint32_t log_data_stride, stage_metadata s_meta, uint32_t batch_size)
  {
    const uint64_t data_stride_u64 = data_stride;
    data += ((s_meta.ntt_block_id & (data_stride - 1)) + data_stride_u64 * s_meta.ntt_inp_id * 2 +
             (s_meta.ntt_block_id >> log_data_stride) * data_stride_u64 * s_meta.ntt_block_size) *
              batch_size +
            s_meta.batch_id;

    UNROLL
    for (uint32_t j = 0; j < 2; j++) {
      UNROLL
      for (uint32_t i = 0; i < 4; i++) {
        X[4 * j + i] = data[(8 * i + j) * data_stride_u64 * batch_size];
      }
    }
  }

  DEVICE_INLINE void
  storeGlobalData32(E* data, uint32_t data_stride, uint32_t log_data_stride, bool strided, stage_metadata s_meta)
  {
    const uint64_t data_stride_u64 = data_stride;
    if (strided) {
      data += (s_meta.ntt_block_id & (data_stride - 1)) + data_stride_u64 * s_meta.ntt_inp_id * 2 +
              (s_meta.ntt_block_id >> log_data_stride) * data_stride_u64 * s_meta.ntt_block_size;
    } else {
      data += (uint64_t)s_meta.ntt_block_id * s_meta.ntt_block_size + s_meta.ntt_inp_id * 2;
    }

    UNROLL
    for (uint32_t j = 0; j < 2; j++) {
      UNROLL
      for (uint32_t i = 0; i < 4; i++) {
        data[(8 * i + j) * data_stride_u64] = X[4 * j + i];
      }
    }
  }

  DEVICE_INLINE void storeGlobalData32ColumnBatch(
    E* data, uint32_t data_stride, uint32_t log_data_stride, stage_metadata s_meta, uint32_t batch_size)
  {
    const uint64_t data_stride_u64 = data_stride;
    data += ((s_meta.ntt_block_id & (data_stride - 1)) + data_stride_u64 * s_meta.ntt_inp_id * 2 +
             (s_meta.ntt_block_id >> log_data_stride) * data_stride_u64 * s_meta.ntt_block_size) *
              batch_size +
            s_meta.batch_id;

    UNROLL
    for (uint32_t j = 0; j < 2; j++) {
      UNROLL
      for (uint32_t i = 0; i < 4; i++) {
        data[(8 * i + j) * data_stride_u64 * batch_size] = X[4 * j + i];
      }
    }
  }

  DEVICE_INLINE void
  loadGlobalData16(const E* data, uint32_t data_stride, uint32_t log_data_stride, bool strided, stage_metadata s_meta)
  {
    const uint64_t data_stride_u64 = data_stride;
    if (strided) {
      data += (s_meta.ntt_block_id & (data_stride - 1)) + data_stride_u64 * s_meta.ntt_inp_id * 4 +
              (s_meta.ntt_block_id >> log_data_stride) * data_stride_u64 * s_meta.ntt_block_size;
    } else {
      data += (uint64_t)s_meta.ntt_block_id * s_meta.ntt_block_size + s_meta.ntt_inp_id * 4;
    }

    UNROLL
    for (uint32_t j = 0; j < 4; j++) {
      UNROLL
      for (uint32_t i = 0; i < 2; i++) {
        X[2 * j + i] = data[(8 * i + j) * data_stride_u64];
      }
    }
  }

  DEVICE_INLINE void loadGlobalData16ColumnBatch(
    const E* data, uint32_t data_stride, uint32_t log_data_stride, stage_metadata s_meta, uint32_t batch_size)
  {
    const uint64_t data_stride_u64 = data_stride;
    data += ((s_meta.ntt_block_id & (data_stride - 1)) + data_stride_u64 * s_meta.ntt_inp_id * 4 +
             (s_meta.ntt_block_id >> log_data_stride) * data_stride_u64 * s_meta.ntt_block_size) *
              batch_size +
            s_meta.batch_id;

    UNROLL
    for (uint32_t j = 0; j < 4; j++) {
      UNROLL
      for (uint32_t i = 0; i < 2; i++) {
        X[2 * j + i] = data[(8 * i + j) * data_stride_u64 * batch_size];
      }
    }
  }

  DEVICE_INLINE void
  storeGlobalData16(E* data, uint32_t data_stride, uint32_t log_data_stride, bool strided, stage_metadata s_meta)
  {
    const uint64_t data_stride_u64 = data_stride;
    if (strided) {
      data += (s_meta.ntt_block_id & (data_stride - 1)) + data_stride_u64 * s_meta.ntt_inp_id * 4 +
              (s_meta.ntt_block_id >> log_data_stride) * data_stride_u64 * s_meta.ntt_block_size;
    } else {
      data += (uint64_t)s_meta.ntt_block_id * s_meta.ntt_block_size + s_meta.ntt_inp_id * 4;
    }

    UNROLL
    for (uint32_t j = 0; j < 4; j++) {
      UNROLL
      for (uint32_t i = 0; i < 2; i++) {
        data[(8 * i + j) * data_stride_u64] = X[2 * j + i];
      }
    }
  }

  DEVICE_INLINE void storeGlobalData16ColumnBatch(
    E* data, uint32_t data_stride, uint32_t log_data_stride, stage_metadata s_meta, uint32_t batch_size)
  {
    const uint64_t data_stride_u64 = data_stride;
    data += ((s_meta.ntt_block_id & (data_stride - 1)) + data_stride_u64 * s_meta.ntt_inp_id * 4 +
             (s_meta.ntt_block_id >> log_data_stride) * data_stride_u64 * s_meta.ntt_block_size) *
              batch_size +
            s_meta.batch_id;

    UNROLL
    for (uint32_t j = 0; j < 4; j++) {
      UNROLL
      for (uint32_t i = 0; i < 2; i++) {
        data[(8 * i + j) * data_stride_u64 * batch_size] = X[2 * j + i];
      }
    }
  }

  DEVICE_INLINE void ntt4_2()
  {
    UNROLL
    for (int i = 0; i < 2; i++) {
      ntt4(X[4 * i], X[4 * i + 1], X[4 * i + 2], X[4 * i + 3]);
    }
  }

  DEVICE_INLINE void ntt2_4()
  {
    UNROLL
    for (int i = 0; i < 4; i++) {
      ntt2(X[2 * i], X[2 * i + 1]);
    }
  }

  DEVICE_INLINE void ntt2(E& X0, E& X1)
  {
    E T;

    T = X0 + X1;
    X1 = X0 - X1;
    X0 = T;
  }

  DEVICE_INLINE void ntt4(E& X0, E& X1, E& X2, E& X3)
  {
    E T;

    T = X0 + X2;
    X2 = X0 - X2;
    X0 = X1 + X3;
    X1 = X1 - X3; // T has X0, X0 has X1, X2 has X2, X1 has X3

    X1 = X1 * WB[0];

    X3 = X2 - X1;
    X1 = X2 + X1;
    X2 = T - X0;
    X0 = T + X0;
  }

  // rbo version
  DEVICE_INLINE void ntt4rbo(E& X0, E& X1, E& X2, E& X3)
  {
    E T;

    T = X0 - X1;
    X0 = X0 + X1;
    X1 = X2 + X3;
    X3 = X2 - X3; // T has X0, X0 has X1, X2 has X2, X1 has X3

    X3 = X3 * WB[0];

    X2 = X0 - X1;
    X0 = X0 + X1;
    X1 = T + X3;
    X3 = T - X3;
  }

  DEVICE_INLINE void ntt8(E& X0, E& X1, E& X2, E& X3, E& X4, E& X5, E& X6, E& X7)
  {
    E T;

    // out of 56,623,104 possible mappings, we have:
    T = X3 - X7;
    X7 = X3 + X7;
    X3 = X1 - X5;
    X5 = X1 + X5;
    X1 = X2 + X6;
    X2 = X2 - X6;
    X6 = X0 + X4;
    X0 = X0 - X4;

    T = T * WB[1];
    X2 = X2 * WB[1];

    X4 = X6 + X1;
    X6 = X6 - X1;
    X1 = X3 + T;
    X3 = X3 - T;
    T = X5 + X7;
    X5 = X5 - X7;
    X7 = X0 + X2;
    X0 = X0 - X2;

    X1 = X1 * WB[0];
    X5 = X5 * WB[1];
    X3 = X3 * WB[2];

    X2 = X6 + X5;
    X6 = X6 - X5;
    X5 = X7 - X1;
    X1 = X7 + X1;
    X7 = X0 - X3;
    X3 = X0 + X3;
    X0 = X4 + T;
    X4 = X4 - T;
  }

  DEVICE_INLINE void ntt8win()
  {
    E T;

    T = X[3] - X[7];
    X[7] = X[3] + X[7];
    X[3] = X[1] - X[5];
    X[5] = X[1] + X[5];
    X[1] = X[2] + X[6];
    X[2] = X[2] - X[6];
    X[6] = X[0] + X[4];
    X[0] = X[0] - X[4];

    X[2] = X[2] * WB[0];

    X[4] = X[6] + X[1];
    X[6] = X[6] - X[1];
    X[1] = X[3] + T;
    X[3] = X[3] - T;
    T = X[5] + X[7];
    X[5] = X[5] - X[7];
    X[7] = X[0] + X[2];
    X[0] = X[0] - X[2];

    X[1] = X[1] * WB[1];
    X[5] = X[5] * WB[0];
    X[3] = X[3] * WB[2];

    X[2] = X[6] + X[5];
    X[6] = X[6] - X[5];

    X[5] = X[1] + X[3];
    X[3] = X[1] - X[3];

    X[1] = X[7] + X[5];
    X[5] = X[7] - X[5];
    X[7] = X[0] - X[3];
    X[3] = X[0] + X[3];
    X[0] = X[4] + T;
    X[4] = X[4] - T;
  }

  DEVICE_INLINE void SharedData64Columns8(E* shmem, bool store, bool high_bits, bool stride)
  {
    uint32_t ntt_id = stride ? threadIdx.x & 0x7 : threadIdx.x >> 3;
    uint32_t column_id = stride ? threadIdx.x >> 3 : threadIdx.x & 0x7;

    UNROLL
    for (uint32_t i = 0; i < 8; i++) {
      if (store) {
        shmem[ntt_id * 64 + i * 8 + column_id] = X[i];
      } else {
        X[i] = shmem[ntt_id * 64 + i * 8 + column_id];
      }
    }
  }

  DEVICE_INLINE void SharedData64Rows8(E* shmem, bool store, bool high_bits, bool stride)
  {
    uint32_t ntt_id = stride ? threadIdx.x & 0x7 : threadIdx.x >> 3;
    uint32_t row_id = stride ? threadIdx.x >> 3 : threadIdx.x & 0x7;

    UNROLL
    for (uint32_t i = 0; i < 8; i++) {
      if (store) {
        shmem[ntt_id * 64 + row_id * 8 + i] = X[i];
      } else {
        X[i] = shmem[ntt_id * 64 + row_id * 8 + i];
      }
    }
  }

  DEVICE_INLINE void SharedData32Columns8(E* shmem, bool store, bool high_bits, bool stride)
  {
    uint32_t ntt_id = stride ? threadIdx.x & 0xf : threadIdx.x >> 2;
    uint32_t column_id = stride ? threadIdx.x >> 4 : threadIdx.x & 0x3;

    UNROLL
    for (uint32_t i = 0; i < 8; i++) {
      if (store) {
        shmem[ntt_id * 32 + i * 4 + column_id] = X[i];
      } else {
        X[i] = shmem[ntt_id * 32 + i * 4 + column_id];
      }
    }
  }

  DEVICE_INLINE void SharedData32Rows8(E* shmem, bool store, bool high_bits, bool stride)
  {
    uint32_t ntt_id = stride ? threadIdx.x & 0xf : threadIdx.x >> 2;
    uint32_t row_id = stride ? threadIdx.x >> 4 : threadIdx.x & 0x3;

    UNROLL
    for (uint32_t i = 0; i < 8; i++) {
      if (store) {
        shmem[ntt_id * 32 + row_id * 8 + i] = X[i];
      } else {
        X[i] = shmem[ntt_id * 32 + row_id * 8 + i];
      }
    }
  }

  DEVICE_INLINE void SharedData32Columns4_2(E* shmem, bool store, bool high_bits, bool stride)
  {
    uint32_t ntt_id = stride ? threadIdx.x & 0xf : threadIdx.x >> 2;
    uint32_t column_id = (stride ? threadIdx.x >> 4 : threadIdx.x & 0x3) * 2;

    UNROLL
    for (uint32_t j = 0; j < 2; j++) {
      UNROLL
      for (uint32_t i = 0; i < 4; i++) {
        if (store) {
          shmem[ntt_id * 32 + i * 8 + column_id + j] = X[4 * j + i];
        } else {
          X[4 * j + i] = shmem[ntt_id * 32 + i * 8 + column_id + j];
        }
      }
    }
  }

  DEVICE_INLINE void SharedData32Rows4_2(E* shmem, bool store, bool high_bits, bool stride)
  {
    uint32_t ntt_id = stride ? threadIdx.x & 0xf : threadIdx.x >> 2;
    uint32_t row_id = (stride ? threadIdx.x >> 4 : threadIdx.x & 0x3) * 2;

    UNROLL
    for (uint32_t j = 0; j < 2; j++) {
      UNROLL
      for (uint32_t i = 0; i < 4; i++) {
        if (store) {
          shmem[ntt_id * 32 + row_id * 4 + 4 * j + i] = X[4 * j + i];
        } else {
          X[4 * j + i] = shmem[ntt_id * 32 + row_id * 4 + 4 * j + i];
        }
      }
    }
  }

  DEVICE_INLINE void SharedData16Columns8(E* shmem, bool store, bool high_bits, bool stride)
  {
    uint32_t ntt_id = stride ? threadIdx.x & 0x1f : threadIdx.x >> 1;
    uint32_t column_id = stride ? threadIdx.x >> 5 : threadIdx.x & 0x1;

    UNROLL
    for (uint32_t i = 0; i < 8; i++) {
      if (store) {
        shmem[ntt_id * 16 + i * 2 + column_id] = X[i];
      } else {
        X[i] = shmem[ntt_id * 16 + i * 2 + column_id];
      }
    }
  }

  DEVICE_INLINE void SharedData16Rows8(E* shmem, bool store, bool high_bits, bool stride)
  {
    uint32_t ntt_id = stride ? threadIdx.x & 0x1f : threadIdx.x >> 1;
    uint32_t row_id = stride ? threadIdx.x >> 5 : threadIdx.x & 0x1;

    UNROLL
    for (uint32_t i = 0; i < 8; i++) {
      if (store) {
        shmem[ntt_id * 16 + row_id * 8 + i] = X[i];
      } else {
        X[i] = shmem[ntt_id * 16 + row_id * 8 + i];
      }
    }
  }

  DEVICE_INLINE void SharedData16Columns2_4(E* shmem, bool store, bool high_bits, bool stride)
  {
    uint32_t ntt_id = stride ? threadIdx.x & 0x1f : threadIdx.x >> 1;
    uint32_t column_id = (stride ? threadIdx.x >> 5 : threadIdx.x & 0x1) * 4;

    UNROLL
    for (uint32_t j = 0; j < 4; j++) {
      UNROLL
      for (uint32_t i = 0; i < 2; i++) {
        if (store) {
          shmem[ntt_id * 16 + i * 8 + column_id + j] = X[2 * j + i];
        } else {
          X[2 * j + i] = shmem[ntt_id * 16 + i * 8 + column_id + j];
        }
      }
    }
  }

  DEVICE_INLINE void SharedData16Rows2_4(E* shmem, bool store, bool high_bits, bool stride)
  {
    uint32_t ntt_id = stride ? threadIdx.x & 0x1f : threadIdx.x >> 1;
    uint32_t row_id = (stride ? threadIdx.x >> 5 : threadIdx.x & 0x1) * 4;

    UNROLL
    for (uint32_t j = 0; j < 4; j++) {
      UNROLL
      for (uint32_t i = 0; i < 2; i++) {
        if (store) {
          shmem[ntt_id * 16 + row_id * 2 + 2 * j + i] = X[2 * j + i];
        } else {
          X[2 * j + i] = shmem[ntt_id * 16 + row_id * 2 + 2 * j + i];
        }
      }
    }
  }

  DEVICE_INLINE void twiddlesInternal()
  {
    UNROLL
    for (int i = 1; i < 8; i++) {
      X[i] = X[i] * WI[i - 1];
    }
  }

  DEVICE_INLINE void twiddlesExternal()
  {
    UNROLL
    for (int i = 0; i < 8; i++) {
      X[i] = X[i] * WE[i];
    }
  }
};
