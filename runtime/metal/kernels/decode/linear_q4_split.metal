#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/q4_mpp_tiles.h"

// Split-K decode projections for one lane (rows == 8). A threadgroup holds
// Parts partitions of Simdgroups simdgroups each and launches with
// Parts * Simdgroups * 32 threads; the partitions stream equal K ranges of one
// 8 x TileN tile into threadgroup partials (q4_mpp_tile_split), then the whole
// threadgroup reduces the partials and applies the epilogue. A projection
// with fewer 256-wide tiles than the GPU has cores cannot fill the cores with
// the sequential kernels; splitting K multiplies the simdgroups per tile
// instead of narrowing the tile further. No device scratch, extra dispatch or
// weight-layout change. Requires input_size % 1024 == 0 (four 256-input
// ranges) and, as dispatched by ops::Q4Linear, one threadgroup per tile.
//
// The buffer contracts equal the decode kernels in linear_q4.metal: Affine
// (input, weights, scales, biases, output, params), Residual (..., residual,
// output, params) and GateUp (gate stream, output, up stream, params). The
// destination's type Out is bf16, or fp32 for a plain projection's logits
// (ops::Projection::destination), which keeps the sum unrounded.
// splash-m5: Rows generalises the one-lane kernel to 16/24/32 verify rows (two
// to four lanes); the Rows == 8 kernels below are unchanged.
template <ushort TileN, ushort Simdgroups, bool Residual, bool GateUp = false, class Out,
          ushort Rows = 8, ushort Parts = 4, ushort Depth = 2, ushort Diag = 0>
inline void q4_split(device bfloat *input, device uchar *weights,
                     device bfloat *scales, device bfloat *biases,
                     device bfloat *residual, device Out *output,
                     device uchar *upWeights, device bfloat *upScales,
                     device bfloat *upBiases, constant Q4Params &p, uint group,
                     uint lane, uint simd, threadgroup float *sums,
                     threadgroup float *partials) {
  uint partition = simd / Simdgroups;
  for (uint tile = group; tile < p.output_size / TileN;
       tile += p.persistent_groups) {
    q4_mpp_tile_split<TileN, GateUp, 256, true, Simdgroups, Parts, Rows, Depth, false, Diag>(
        input, weights, scales, biases, partials, upWeights, upScales,
        upBiases, p.input_size, sums + partition * 8 * Rows, tile * TileN, lane,
        simd % Simdgroups, partition);
    // Same epilogue as q4_mpp_tile: one bf16 rounding of the projection,
    // then the residual add or the SiLU gate, then the output rounding.
    for (uint i = simd * 32 + lane; i < Rows * TileN;
         i += Parts * Simdgroups * 32) {
      float value = 0;
      for (uint part = 0; part < Parts; ++part)
        value += partials[part * Rows * TileN + i];
      uint index = (i / TileN) * p.output_size + tile * TileN + i % TileN;
      if constexpr (!is_same_v<Out, float>) value = float(bfloat(value));
      if constexpr (GateUp) {
        float up = 0;
        for (uint part = 0; part < Parts; ++part)
          up += partials[(Parts + part) * Rows * TileN + i];
        value = value / (1.0f + fast::exp2(-1.44269504089f * value)) *
                float(bfloat(up));
      }
      if constexpr (Residual)
        value += float(residual[index]);
      output[index] = Out(value);
    }
    // The next tile's partials overwrite this reduction's inputs.
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// A plain projection reads no residual (its input stands in) and writes a
// destination of type Out: each kernel into bf16 and into fp32 (Name_f32: the
// logits, ops::Projection::destination).
#define Q4_SPLIT_OUTPUT(Name, TileN, Simdgroups, Out)                          \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   device Out *output [[buffer(4)]],                           \
                   constant Q4Params &params [[buffer(5)]],                    \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint lane [[thread_index_in_simdgroup]],                    \
                   uint simd [[simdgroup_index_in_threadgroup]]) {             \
    threadgroup float sums[4 * 64], partials[4 * 8 * TileN];                   \
    q4_split<TileN, Simdgroups, false>(input, weights, scales, biases, input,  \
                                       output, weights, scales, biases,        \
                                       params, group, lane, simd, sums,        \
                                       partials);                              \
  }
#define Q4_SPLIT_AFFINE(Name, TileN, Simdgroups)                               \
  Q4_SPLIT_OUTPUT(Name, TileN, Simdgroups, bfloat)                             \
  Q4_SPLIT_OUTPUT(Name##_f32, TileN, Simdgroups, float)

#define Q4_SPLIT_RESIDUAL(Name, TileN, Simdgroups)                             \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   device bfloat *residual [[buffer(4)]],                      \
                   device bfloat *output [[buffer(5)]],                        \
                   constant Q4Params &params [[buffer(6)]],                    \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint lane [[thread_index_in_simdgroup]],                    \
                   uint simd [[simdgroup_index_in_threadgroup]]) {             \
    threadgroup float sums[4 * 64], partials[4 * 8 * TileN];                   \
    q4_split<TileN, Simdgroups, true>(input, weights, scales, biases,          \
                                      residual, output, weights, scales,       \
                                      biases, params, group, lane, simd, sums, \
                                      partials);                               \
  }

// Threads per threadgroup = 4 partitions x Simdgroups x 32. Only N32 and
// N64 are instantiated: these are the tiles selected by the split policy.
// A wider tile would require separate register-pressure and timing evidence.
Q4_SPLIT_AFFINE(decode_linear_q4_n32_split4, 32, 1)        // 128 threads
Q4_SPLIT_AFFINE(decode_linear_q4_n64_split4, 64, 2)        // 256 threads
Q4_SPLIT_RESIDUAL(decode_linear_q4_n32_split4_residual, 32, 1)
Q4_SPLIT_RESIDUAL(decode_linear_q4_n64_split4_residual, 64, 2)
#undef Q4_SPLIT_AFFINE
#undef Q4_SPLIT_OUTPUT
#undef Q4_SPLIT_RESIDUAL

// 128 threads: four single-simdgroup partitions, two weight streams.
kernel void decode_linear_q4_n32_split4_gate_up(
    device bfloat *input [[buffer(0)]], device uchar *weights_0 [[buffer(1)]],
    device bfloat *scales_0 [[buffer(2)]],
    device bfloat *biases_0 [[buffer(3)]], device bfloat *output [[buffer(4)]],
    device uchar *weights_1 [[buffer(5)]], device bfloat *scales_1 [[buffer(6)]],
    device bfloat *biases_1 [[buffer(7)]], constant Q4Params &params [[buffer(8)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup float sums[4 * 64], partials[2 * 4 * 8 * 32];
  q4_split<32, 1, false, true>(input, weights_0, scales_0, biases_0, output,
                               output, weights_1, scales_1, biases_1, params,
                               group, lane, simd, sums, partials);
}

// splash-m5: two to four lanes (16/24/32 rows) with the N32 split tile, 128
// threads. A 5120-wide projection then runs 160 threadgroups of four K
// partitions instead of 40 sequential N128 tiles, which is what the 17408-deep
// down projection needs to fill 40 cores. Numerics as the one-lane form: each
// partition sums its K range in group order; the four sums are added in
// partition order.
#define Q4_SPLIT_ROWS_OUT(Name, Rows, Sg, Out)                                            \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   device Out *output [[buffer(4)]],                           \
                   constant Q4Params &params [[buffer(5)]],                    \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint lane [[thread_index_in_simdgroup]],                    \
                   uint simd [[simdgroup_index_in_threadgroup]]) {             \
    threadgroup float sums[4 * 8 * Rows], partials[4 * Rows * 32];             \
    q4_split<32, Sg, false, false, Out, Rows>(                                 \
        input, weights, scales, biases, input, output, weights, scales,        \
        biases, params, group, lane, simd, sums, partials);                    \
  }
// Each plain kernel into bf16 and into fp32 (Name_f32: the logits destination).
#define Q4_SPLIT_ROWS(Name, Rows, Sg)                                          \
  Q4_SPLIT_ROWS_OUT(Name, Rows, Sg, bfloat)                                    \
  Q4_SPLIT_ROWS_OUT(Name##_f32, Rows, Sg, float)

#define Q4_SPLIT_RESIDUAL_ROWS(Name, Rows, Sg)                                   \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   device bfloat *residual [[buffer(4)]],                      \
                   device bfloat *output [[buffer(5)]],                        \
                   constant Q4Params &params [[buffer(6)]],                    \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint lane [[thread_index_in_simdgroup]],                    \
                   uint simd [[simdgroup_index_in_threadgroup]]) {             \
    threadgroup float sums[4 * 8 * Rows], partials[4 * Rows * 32];             \
    q4_split<32, Sg, true, false, bfloat, Rows>(                               \
        input, weights, scales, biases, residual, output, weights, scales,     \
        biases, params, group, lane, simd, sums, partials);                    \
  }
Q4_SPLIT_ROWS(decode_linear_q4_n32_split4_m16, 16, 1)
Q4_SPLIT_ROWS(decode_linear_q4_n32_split4_m24, 24, 1)
Q4_SPLIT_ROWS(decode_linear_q4_n32_split4_m32, 32, 1)
Q4_SPLIT_RESIDUAL_ROWS(decode_linear_q4_n32_split4_residual_m16, 16, 1)
Q4_SPLIT_RESIDUAL_ROWS(decode_linear_q4_n32_split4_residual_m24, 24, 1)
Q4_SPLIT_RESIDUAL_ROWS(decode_linear_q4_n32_split4_residual_m32, 32, 1)
Q4_SPLIT_ROWS(decode_linear_q4_n32_split4_m16_sg8, 16, 2)
Q4_SPLIT_ROWS(decode_linear_q4_n32_split4_m24_sg8, 24, 2)
Q4_SPLIT_ROWS(decode_linear_q4_n32_split4_m32_sg8, 32, 2)
Q4_SPLIT_RESIDUAL_ROWS(decode_linear_q4_n32_split4_residual_m16_sg8, 16, 2)
Q4_SPLIT_RESIDUAL_ROWS(decode_linear_q4_n32_split4_residual_m24_sg8, 24, 2)
Q4_SPLIT_RESIDUAL_ROWS(decode_linear_q4_n32_split4_residual_m32_sg8, 32, 2)
#undef Q4_SPLIT_ROWS
#undef Q4_SPLIT_ROWS_OUT
#undef Q4_SPLIT_RESIDUAL_ROWS

// splash-m5 experiments (not planned by Linear; dispatched raw by kernel-bench):
// one-lane residual split kernels with Parts K partitions of one simdgroup
// (Parts * 32 threads) and TileN columns. K must split into whole four-group
// blocks per partition: K % (256 * Parts) == 0.
#define M5X_SPLIT_RESIDUAL(Name, TileN, Parts, Depth)                                \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   device bfloat *residual [[buffer(4)]],                      \
                   device bfloat *output [[buffer(5)]],                        \
                   constant Q4Params &params [[buffer(6)]],                    \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint lane [[thread_index_in_simdgroup]],                    \
                   uint simd [[simdgroup_index_in_threadgroup]]) {             \
    threadgroup float sums[Parts * 64], partials[Parts * 8 * TileN];           \
    q4_split<TileN, 1, true, false, bfloat, 8, Parts, Depth>(                  \
        input, weights, scales, biases, residual, output, weights, scales,     \
        biases, params, group, lane, simd, sums, partials);                    \
  }
M5X_SPLIT_RESIDUAL(m5x_n32_p2_residual, 32, 2, 2)
M5X_SPLIT_RESIDUAL(m5x_n32_p6_residual, 32, 6, 2)
M5X_SPLIT_RESIDUAL(m5x_n32_p8_residual, 32, 8, 2)
M5X_SPLIT_RESIDUAL(m5x_n32_p12_residual, 32, 12, 2)
M5X_SPLIT_RESIDUAL(m5x_n32_p17_residual, 32, 17, 2)
M5X_SPLIT_RESIDUAL(m5x_n32_p24_residual, 32, 24, 2)
M5X_SPLIT_RESIDUAL(m5x_n16_p4_residual, 16, 4, 2)
M5X_SPLIT_RESIDUAL(m5x_n16_p8_residual, 16, 8, 2)
M5X_SPLIT_RESIDUAL(m5x_n16_p17_residual, 16, 17, 2)
M5X_SPLIT_RESIDUAL(m5x_n32_p4_d4_residual, 32, 4, 4)
M5X_SPLIT_RESIDUAL(m5x_n32_p8_d4_residual, 32, 8, 4)
M5X_SPLIT_RESIDUAL(m5x_n32_p17_d4_residual, 32, 17, 4)
M5X_SPLIT_RESIDUAL(m5x_n32_p24_d4_residual, 32, 24, 4)
// H6: N64 tiles with one simdgroup per K partition (half the input re-reads of
// N32, 80 threadgroups of 128 threads), with and without depth-4 pipelining.
M5X_SPLIT_RESIDUAL(m5x_n64s1_p4_residual, 64, 4, 2)
M5X_SPLIT_RESIDUAL(m5x_n64s1_p4_d4_residual, 64, 4, 4)
M5X_SPLIT_RESIDUAL(m5x_n64s1_p8_residual, 64, 8, 2)
// Planned by Linear (widerSplit): one-lane residual N32 with 8, 17 or 24 K
// partitions of one simdgroup each (256, 544 and 768 threads).
M5X_SPLIT_RESIDUAL(decode_linear_q4_n32_split8_residual, 32, 8, 2)
M5X_SPLIT_RESIDUAL(decode_linear_q4_n32_split17_residual, 32, 17, 2)
M5X_SPLIT_RESIDUAL(decode_linear_q4_n32_split24_residual, 32, 24, 2)
#undef M5X_SPLIT_RESIDUAL

// H9 experiment: one-lane residual N32 split with every quant group's input sums
// computed once per threadgroup (all K, [group][row]) before the partitions run,
// so the partition loops carry no input-sum work or barriers. MaxGroups bounds K
// (272 groups = 17408, the largest Qwen3.8-27B input).
template <ushort Parts, ushort MaxGroups = 272>
inline void m5x_split_sums_ready(device bfloat *input, device uchar *weights,
                                 device bfloat *scales, device bfloat *biases,
                                 device bfloat *residual, device bfloat *output,
                                 constant Q4Params &p, uint group, uint lane,
                                 uint simd, threadgroup float *sums,
                                 threadgroup float *partials) {
  constexpr uint TileN = 32, Rows = 8;
  const uint groups = p.input_size / 64;
  for (uint task = simd; task < groups * Rows; task += Parts) {
    uint g = task / Rows, row = task % Rows;
    uint origin = row * p.input_size + g * 64 + lane;
    float sum = simd_sum(float(input[origin]) + float(input[origin + 32]));
    if (lane == 0) sums[g * Rows + row] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint partition = simd;
  for (uint tile = group; tile < p.output_size / TileN; tile += p.persistent_groups) {
    q4_mpp_tile_split<TileN, false, 256, true, 1, Parts, Rows, 2, true>(
        input, weights, scales, biases, partials, weights, scales, biases,
        p.input_size, sums, tile * TileN, lane, 0, partition);
    for (uint i = simd * 32 + lane; i < Rows * TileN; i += Parts * 32) {
      float value = 0;
      for (uint part = 0; part < Parts; ++part) value += partials[part * Rows * TileN + i];
      uint index = (i / TileN) * p.output_size + tile * TileN + i % TileN;
      value = float(bfloat(value)) + float(residual[index]);
      output[index] = bfloat(value);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
#define M5X_SUMS_READY(Name, Parts)                                            \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   device bfloat *residual [[buffer(4)]],                      \
                   device bfloat *output [[buffer(5)]],                        \
                   constant Q4Params &params [[buffer(6)]],                    \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint lane [[thread_index_in_simdgroup]],                    \
                   uint simd [[simdgroup_index_in_threadgroup]]) {             \
    threadgroup float sums[272 * 8], partials[Parts * 8 * 32];                 \
    m5x_split_sums_ready<Parts>(input, weights, scales, biases, residual,      \
                                output, params, group, lane, simd, sums,       \
                                partials);                                     \
  }
M5X_SUMS_READY(m5x_n32_p4_ps_residual, 4)
M5X_SUMS_READY(m5x_n32_p8_ps_residual, 8)
M5X_SUMS_READY(m5x_n32_p17_ps_residual, 17)
#undef M5X_SUMS_READY

// Diagnostics (timing only; outputs are wrong by design): Diag 1 drops the
// scale/bias epilogue, 2 skips the matmul (no weight reads), 3 both.
#define M5X_DIAG(Name, Parts, Diag)                                            \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   device bfloat *residual [[buffer(4)]],                      \
                   device bfloat *output [[buffer(5)]],                        \
                   constant Q4Params &params [[buffer(6)]],                    \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint lane [[thread_index_in_simdgroup]],                    \
                   uint simd [[simdgroup_index_in_threadgroup]]) {             \
    threadgroup float sums[Parts * 64], partials[Parts * 8 * 32];              \
    q4_split<32, 1, true, false, bfloat, 8, Parts, 2, Diag>(                   \
        input, weights, scales, biases, residual, output, weights, scales,     \
        biases, params, group, lane, simd, sums, partials);                    \
  }
M5X_DIAG(m5x_diag1_p4, 4, 1)
M5X_DIAG(m5x_diag2_p4, 4, 2)
M5X_DIAG(m5x_diag3_p4, 4, 3)
M5X_DIAG(m5x_diag1_p17, 17, 1)
M5X_DIAG(m5x_diag2_p17, 17, 2)
M5X_DIAG(m5x_diag3_p17, 17, 3)
M5X_DIAG(m5x_diag4_p4, 4, 4)
M5X_DIAG(m5x_diag4_p17, 17, 4)
#undef M5X_DIAG
// D5: launch floor: the same grid and buffers, one residual copy per output.
kernel void m5x_diag5(device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],
                      device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]],
                      device bfloat *residual [[buffer(4)]], device bfloat *output [[buffer(5)]],
                      constant Q4Params &params [[buffer(6)]],
                      uint group [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]]) {
  (void)input; (void)weights; (void)scales; (void)biases;
  for (uint i = tid; i < 8 * 32; i += 128) {
    uint index = (i / 32) * params.output_size + group * 32 + i % 32;
    output[index] = residual[index];
  }
}

// H10: one-lane row sums computed once per projection ([group][row], the same
// simd_sum as q4_store_input_sums, so bit-identical), then split kernels that copy
// them into threadgroup memory instead of every threadgroup recomputing them.
// Grid: input_size / 64 threadgroups of 256 threads (one simdgroup per row).
kernel void m5x_row_sums8(device const bfloat *input [[buffer(0)]],
                          device float *sums [[buffer(1)]],
                          constant Q4Params &p [[buffer(2)]],
                          uint g [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint row [[simdgroup_index_in_threadgroup]]) {
  uint origin = row * p.input_size + g * 64 + lane;
  float sum = simd_sum(float(input[origin]) + float(input[origin + 32]));
  if (lane == 0) sums[g * 8 + row] = sum;
}
kernel void decode_linear_q4_row_sums8(device const bfloat *input [[buffer(0)]],
                          device float *sums [[buffer(1)]],
                          constant Q4Params &p [[buffer(2)]],
                          uint g [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint row [[simdgroup_index_in_threadgroup]]) {
  uint origin = row * p.input_size + g * 64 + lane;
  float sum = simd_sum(float(input[origin]) + float(input[origin + 32]));
  if (lane == 0) sums[g * 8 + row] = sum;
}
template <ushort Parts, bool Residual = true, class Out = bfloat, ushort Early = 0>
inline void m5x_split_device_sums(device bfloat *input, device uchar *weights,
                                  device bfloat *scales, device bfloat *biases,
                                  device bfloat *residual, device Out *output,
                                  device const float *row_sums, constant Q4Params &p,
                                  uint group, uint lane, uint simd,
                                  threadgroup float *sums, threadgroup float *partials) {
  constexpr uint TileN = 32, Rows = 8;
  // splash-m5 (kernel lab 2, DeviceSums): the epilogue reads the row sums from device memory
  // instead of a per-threadgroup copy (8.7 KB of threadgroup memory, a barrier before any weight
  // load); the same floats, so bit-identical. `sums` is unused.
  (void)sums;
  for (uint tile = group; tile < p.output_size / TileN; tile += p.persistent_groups) {
    q4_mpp_tile_split<TileN, false, 256, true, 1, Parts, Rows, 2, true, 0, true, Early>(
        input, weights, scales, biases, partials, weights, scales, biases,
        p.input_size, sums, tile * TileN, lane, 0, simd, row_sums);
    for (uint i = simd * 32 + lane; i < Rows * TileN; i += Parts * 32) {
      float value = 0;
      for (uint part = 0; part < Parts; ++part) value += partials[part * Rows * TileN + i];
      uint index = (i / TileN) * p.output_size + tile * TileN + i % TileN;
      if constexpr (!is_same_v<Out, float>) value = float(bfloat(value));
      if constexpr (Residual) value += float(residual[index]);
      output[index] = Out(value);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
#define M5X_DEVICE_SUMS(Name, Parts)                                           \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   device bfloat *residual [[buffer(4)]],                      \
                   device bfloat *output [[buffer(5)]],                        \
                   device const float *row_sums [[buffer(6)]],                 \
                   constant Q4Params &params [[buffer(7)]],                    \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint lane [[thread_index_in_simdgroup]],                    \
                   uint simd [[simdgroup_index_in_threadgroup]]) {             \
    threadgroup float sums[1], partials[Parts * 8 * 32];                 \
    m5x_split_device_sums<Parts>(input, weights, scales, biases, residual,     \
                                 output, row_sums, params, group, lane, simd,  \
                                 sums, partials);                              \
  }
M5X_DEVICE_SUMS(m5x_n32_p4_devsums, 4)
M5X_DEVICE_SUMS(m5x_n32_p8_devsums, 8)
M5X_DEVICE_SUMS(m5x_n32_p17_devsums, 17)
// Planned by Linear as LinearTile::SplitSums32 (splits 1 = four partitions).
M5X_DEVICE_SUMS(decode_linear_q4_n32_split4_sums_residual, 4)
M5X_DEVICE_SUMS(decode_linear_q4_n32_split8_sums_residual, 8)
M5X_DEVICE_SUMS(decode_linear_q4_n32_split17_sums_residual, 17)
#undef M5X_DEVICE_SUMS
// Plain one-lane form (GDN-in, attention qkv): buffers input, weights, scales,
// biases, output, row_sums, params; bf16 and the fp32 logits destination.
#define M5_SPLIT_SUMS_PLAIN_OUT(Name, Parts, Out)                              \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   device Out *output [[buffer(4)]],                           \
                   device const float *row_sums [[buffer(5)]],                 \
                   constant Q4Params &params [[buffer(6)]],                    \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint lane [[thread_index_in_simdgroup]],                    \
                   uint simd [[simdgroup_index_in_threadgroup]]) {             \
    threadgroup float sums[1], partials[Parts * 8 * 32];                 \
    m5x_split_device_sums<Parts, false, Out>(input, weights, scales, biases,   \
                                             input, output, row_sums, params,  \
                                             group, lane, simd, sums,          \
                                             partials);                        \
  }
M5_SPLIT_SUMS_PLAIN_OUT(decode_linear_q4_n32_split4_sums, 4, bfloat)
M5_SPLIT_SUMS_PLAIN_OUT(decode_linear_q4_n32_split4_sums_f32, 4, float)
#undef M5_SPLIT_SUMS_PLAIN_OUT

// H11 experiments: plain one-lane projections reading once-per-projection row sums.
// Buffers: input, weights, scales, biases, output, row_sums, params.
template <ushort TileN, ushort Sg>
inline void m5x_tile_device_sums(device bfloat *input, device uchar *weights,
                                 device bfloat *scales, device bfloat *biases,
                                 device bfloat *output, device const float *row_sums,
                                 constant Q4Params &p, uint group, uint lane, uint simd,
                                 threadgroup float *sums) {
  const uint count = p.input_size / 64 * 8;
  for (uint i = simd * 32 + lane; i < count; i += Sg * 32) sums[i] = row_sums[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint tile = group; tile < p.output_size / TileN; tile += p.persistent_groups)
    q4_mpp_tile<TileN, false, false, 256, false, Sg, true>(
        input, weights, scales, biases, output, weights, scales, biases, input,
        p.output_size, p.input_size, sums, tile * TileN, lane, simd);
}
kernel void m5x_n128_plain_devsums(device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],
                                   device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]],
                                   device bfloat *output [[buffer(4)]], device const float *row_sums [[buffer(5)]],
                                   constant Q4Params &params [[buffer(6)]],
                                   uint group [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup float sums[272 * 8];
  m5x_tile_device_sums<128, 8>(input, weights, scales, biases, output, row_sums, params, group, lane, simd, sums);
}
// Split form, plain, TileN columns, Sg simdgroups per partition, 4 partitions.
template <ushort TileN, ushort Sg>
inline void m5x_split_plain_device_sums(device bfloat *input, device uchar *weights,
                                        device bfloat *scales, device bfloat *biases,
                                        device bfloat *output, device const float *row_sums,
                                        constant Q4Params &p, uint group, uint lane, uint simd,
                                        threadgroup float *sums, threadgroup float *partials) {
  constexpr uint Parts = 4, Rows = 8;
  const uint count = p.input_size / 64 * Rows;
  for (uint i = simd * 32 + lane; i < count; i += Parts * Sg * 32) sums[i] = row_sums[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint partition = simd / Sg;
  for (uint tile = group; tile < p.output_size / TileN; tile += p.persistent_groups) {
    q4_mpp_tile_split<TileN, false, 256, true, Sg, Parts, Rows, 2, true>(
        input, weights, scales, biases, partials, weights, scales, biases,
        p.input_size, sums, tile * TileN, lane, simd % Sg, partition);
    for (uint i = simd * 32 + lane; i < Rows * TileN; i += Parts * Sg * 32) {
      float value = 0;
      for (uint part = 0; part < Parts; ++part) value += partials[part * Rows * TileN + i];
      uint index = (i / TileN) * p.output_size + tile * TileN + i % TileN;
      output[index] = bfloat(value);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
kernel void m5x_n64_split4_plain_devsums(device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],
                                         device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]],
                                         device bfloat *output [[buffer(4)]], device const float *row_sums [[buffer(5)]],
                                         constant Q4Params &params [[buffer(6)]],
                                         uint group [[threadgroup_position_in_grid]],
                                         uint lane [[thread_index_in_simdgroup]],
                                         uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup float sums[272 * 8], partials[4 * 8 * 64];
  m5x_split_plain_device_sums<64, 2>(input, weights, scales, biases, output, row_sums, params, group, lane, simd,
                                     sums, partials);
}

// H12: two-to-four-lane residual N32 split kernels reading once-per-projection
// row sums from device memory ([group][row], Rows = 16/24/32), written by
// decode_linear_q4_row_sums (grid K/64 threadgroups of Rows * 32 threads).
kernel void decode_linear_q4_row_sums(device const bfloat *input [[buffer(0)]],
                                      device float *sums [[buffer(1)]],
                                      constant Q4Params &p [[buffer(2)]],
                                      uint g [[threadgroup_position_in_grid]],
                                      uint lane [[thread_index_in_simdgroup]],
                                      uint row [[simdgroup_index_in_threadgroup]],
                                      uint rows [[simdgroups_per_threadgroup]]) {
  uint origin = row * p.input_size + g * 64 + lane;
  float sum = simd_sum(float(input[origin]) + float(input[origin + 32]));
  if (lane == 0) sums[g * rows + row] = sum;
}
// Residual adds the auxiliary buffer; SiluGate multiplies by SiLU of it (the
// up pass of a two-pass gate/up, as q4_mpp_tile_batched's MultiplySiluGate).
template <ushort Rows, ushort Sg, bool Residual = true, bool SiluGate = false, class Out = bfloat,
          ushort TileN = 32, ushort Parts = 4, ushort Diag = 0>
inline void m5_split_rows_device_sums(device bfloat *input, device uchar *weights,
                                      device bfloat *scales, device bfloat *biases,
                                      device bfloat *residual, device Out *output,
                                      device const float *row_sums, constant Q4Params &p,
                                      uint group, uint lane, uint simd,
                                      threadgroup float *partials) {
  uint partition = simd / Sg;
  for (uint tile = group; tile < p.output_size / TileN; tile += p.persistent_groups) {
    q4_mpp_tile_split<TileN, false, 256, true, Sg, Parts, Rows, 2, true, Diag, true>(
        input, weights, scales, biases, partials, weights, scales, biases,
        p.input_size, partials, tile * TileN, lane, simd % Sg, partition, row_sums);
    for (uint i = simd * 32 + lane; i < Rows * TileN; i += Parts * Sg * 32) {
      float value = 0;
      for (uint part = 0; part < Parts; ++part) value += partials[part * Rows * TileN + i];
      uint index = (i / TileN) * p.output_size + tile * TileN + i % TileN;
      if constexpr (!is_same_v<Out, float>) value = float(bfloat(value));
      if constexpr (SiluGate) {
        float gate = float(residual[index]);
        value = gate / (1.0f + fast::exp2(-1.44269504089f * gate)) * value;
      }
      if constexpr (Residual) value += float(residual[index]);
      output[index] = Out(value);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
#define M5_SPLIT_ROWS_SUMS(Name, Rows, Sg)                                     \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   device bfloat *residual [[buffer(4)]],                      \
                   device bfloat *output [[buffer(5)]],                        \
                   device const float *row_sums [[buffer(6)]],                 \
                   constant Q4Params &params [[buffer(7)]],                    \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint lane [[thread_index_in_simdgroup]],                    \
                   uint simd [[simdgroup_index_in_threadgroup]]) {             \
    threadgroup float partials[4 * Rows * 32];                                 \
    m5_split_rows_device_sums<Rows, Sg>(input, weights, scales, biases,        \
                                        residual, output, row_sums, params,    \
                                        group, lane, simd, partials);          \
  }
M5_SPLIT_ROWS_SUMS(decode_linear_q4_n32_split4_sums_residual_m16, 16, 1)
M5_SPLIT_ROWS_SUMS(decode_linear_q4_n32_split4_sums_residual_m24, 24, 1)
M5_SPLIT_ROWS_SUMS(decode_linear_q4_n32_split4_sums_residual_m32, 32, 1)
M5_SPLIT_ROWS_SUMS(decode_linear_q4_n32_split4_sums_residual_m16_sg8, 16, 2)
M5_SPLIT_ROWS_SUMS(decode_linear_q4_n32_split4_sums_residual_m24_sg8, 24, 2)
M5_SPLIT_ROWS_SUMS(decode_linear_q4_n32_split4_sums_residual_m32_sg8, 32, 2)
#undef M5_SPLIT_ROWS_SUMS
// H13: plain (no residual) forms. Buffers: input, weights, scales, biases,
// output, row_sums, params.
#define M5_SPLIT_ROWS_SUMS_PLAIN_OUT(Name, Rows, Sg, Out)                               \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   device Out *output [[buffer(4)]],                           \
                   device const float *row_sums [[buffer(5)]],                 \
                   constant Q4Params &params [[buffer(6)]],                    \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint lane [[thread_index_in_simdgroup]],                    \
                   uint simd [[simdgroup_index_in_threadgroup]]) {             \
    threadgroup float partials[4 * Rows * 32];                                 \
    m5_split_rows_device_sums<Rows, Sg, false, false, Out>(input, weights,   \
                                               scales, biases, input, output,  \
                                               row_sums, params, group, lane,  \
                                               simd, partials);                \
  }
#define M5_SPLIT_ROWS_SUMS_PLAIN(Name, Rows, Sg)                               \
  M5_SPLIT_ROWS_SUMS_PLAIN_OUT(Name, Rows, Sg, bfloat)                         \
  M5_SPLIT_ROWS_SUMS_PLAIN_OUT(Name##_f32, Rows, Sg, float)
M5_SPLIT_ROWS_SUMS_PLAIN(decode_linear_q4_n32_split4_sums_m16, 16, 1)
M5_SPLIT_ROWS_SUMS_PLAIN(decode_linear_q4_n32_split4_sums_m24, 24, 1)
M5_SPLIT_ROWS_SUMS_PLAIN(decode_linear_q4_n32_split4_sums_m32, 32, 1)
M5_SPLIT_ROWS_SUMS_PLAIN(decode_linear_q4_n32_split4_sums_m16_sg8, 16, 2)
M5_SPLIT_ROWS_SUMS_PLAIN(decode_linear_q4_n32_split4_sums_m24_sg8, 24, 2)
M5_SPLIT_ROWS_SUMS_PLAIN(decode_linear_q4_n32_split4_sums_m32_sg8, 32, 2)
#undef M5_SPLIT_ROWS_SUMS_PLAIN
#undef M5_SPLIT_ROWS_SUMS_PLAIN_OUT
// H14: the up pass of a two-pass gate/up. Buffers: input, weights, scales,
// biases, gate, output, row_sums, params.
#define M5_SPLIT_ROWS_SUMS_UP_SILU(Name, Rows, Sg)                             \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   device bfloat *gate [[buffer(4)]],                          \
                   device bfloat *output [[buffer(5)]],                        \
                   device const float *row_sums [[buffer(6)]],                 \
                   constant Q4Params &params [[buffer(7)]],                    \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint lane [[thread_index_in_simdgroup]],                    \
                   uint simd [[simdgroup_index_in_threadgroup]]) {             \
    threadgroup float partials[4 * Rows * 32];                                 \
    m5_split_rows_device_sums<Rows, Sg, false, true>(                          \
        input, weights, scales, biases, gate, output, row_sums, params,        \
        group, lane, simd, partials);                                          \
  }
M5_SPLIT_ROWS_SUMS_UP_SILU(decode_linear_q4_n32_split4_sums_up_silu_m16, 16, 1)
M5_SPLIT_ROWS_SUMS_UP_SILU(decode_linear_q4_n32_split4_sums_up_silu_m24, 24, 1)
M5_SPLIT_ROWS_SUMS_UP_SILU(decode_linear_q4_n32_split4_sums_up_silu_m32, 32, 1)
M5_SPLIT_ROWS_SUMS_UP_SILU(decode_linear_q4_n32_split4_sums_up_silu_m16_sg8, 16, 2)
M5_SPLIT_ROWS_SUMS_UP_SILU(decode_linear_q4_n32_split4_sums_up_silu_m24_sg8, 24, 2)
M5_SPLIT_ROWS_SUMS_UP_SILU(decode_linear_q4_n32_split4_sums_up_silu_m32_sg8, 32, 2)
#undef M5_SPLIT_ROWS_SUMS_UP_SILU

// H15: one-lane gate/up (both streams in one threadgroup, N32, four partitions of
// one simdgroup) reading once-per-projection row sums copied into threadgroup
// memory. Buffers: input, gate weights/scales/biases, output, up
// weights/scales/biases, row_sums, params. Epilogue as q4_split's GateUp.
kernel void decode_linear_q4_n32_split4_sums_gate_up(
    device bfloat *input [[buffer(0)]], device uchar *weights_0 [[buffer(1)]],
    device bfloat *scales_0 [[buffer(2)]], device bfloat *biases_0 [[buffer(3)]],
    device bfloat *output [[buffer(4)]], device uchar *weights_1 [[buffer(5)]],
    device bfloat *scales_1 [[buffer(6)]], device bfloat *biases_1 [[buffer(7)]],
    device const float *row_sums [[buffer(8)]], constant Q4Params &p [[buffer(9)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  constexpr uint TileN = 32, Rows = 8, Parts = 4;
  threadgroup float sums[272 * 8], partials[2 * Parts * Rows * TileN];
  const uint count = p.input_size / 64 * Rows;
  for (uint i = simd * 32 + lane; i < count; i += Parts * 32) sums[i] = row_sums[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint tile = group; tile < p.output_size / TileN; tile += p.persistent_groups) {
    q4_mpp_tile_split<TileN, true, 256, true, 1, Parts, Rows, 2, true>(
        input, weights_0, scales_0, biases_0, partials, weights_1, scales_1, biases_1,
        p.input_size, sums, tile * TileN, lane, 0, simd);
    for (uint i = simd * 32 + lane; i < Rows * TileN; i += Parts * 32) {
      float value = 0, up = 0;
      for (uint part = 0; part < Parts; ++part) {
        value += partials[part * Rows * TileN + i];
        up += partials[(Parts + part) * Rows * TileN + i];
      }
      uint index = (i / TileN) * p.output_size + tile * TileN + i % TileN;
      value = float(bfloat(value));
      value = value / (1.0f + fast::exp2(-1.44269504089f * value)) * float(bfloat(up));
      output[index] = bfloat(value);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// H17 experiments: wider multi-lane plain tiles (less input re-reading per
// projection), sums read from device. Buffers as the plain sums kernels.
#define M5X_ML_PLAIN(Name, Rows, Sg, TileN, Parts)                             \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   device bfloat *output [[buffer(4)]],                        \
                   device const float *row_sums [[buffer(5)]],                 \
                   constant Q4Params &params [[buffer(6)]],                    \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint lane [[thread_index_in_simdgroup]],                    \
                   uint simd [[simdgroup_index_in_threadgroup]]) {             \
    threadgroup float partials[Parts * Rows * TileN];                          \
    m5_split_rows_device_sums<Rows, Sg, false, false, bfloat, TileN, Parts>(   \
        input, weights, scales, biases, input, output, row_sums, params,       \
        group, lane, simd, partials);                                          \
  }
M5X_ML_PLAIN(m5x_ml_sums_n64_p2_sg2_m32, 32, 2, 64, 2)
M5X_ML_PLAIN(m5x_ml_sums_n64_p2_sg4_m32, 32, 4, 64, 2)
M5X_ML_PLAIN(m5x_ml_sums_n64_p4_sg2_m24, 24, 2, 64, 4)
M5X_ML_PLAIN(m5x_ml_sums_n64_p2_sg2_m24, 24, 2, 64, 2)
M5X_ML_PLAIN(m5x_ml_sums_n64_p4_sg1_m16, 16, 1, 64, 4)
M5X_ML_PLAIN(m5x_ml_sums_n64_p4_sg2_m16, 16, 2, 64, 4)
#undef M5X_ML_PLAIN
// Diagnostics (timing only): Diag 1 no scale/bias epilogue, 2 no matmul/weights.
#define M5X_ML_DIAG(Name, Rows, Sg, Diag)                                      \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   device bfloat *output [[buffer(4)]],                        \
                   device const float *row_sums [[buffer(5)]],                 \
                   constant Q4Params &params [[buffer(6)]],                    \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint lane [[thread_index_in_simdgroup]],                    \
                   uint simd [[simdgroup_index_in_threadgroup]]) {             \
    threadgroup float partials[4 * Rows * 32];                                 \
    m5_split_rows_device_sums<Rows, Sg, false, false, bfloat, 32, 4, Diag>(    \
        input, weights, scales, biases, input, output, row_sums, params,       \
        group, lane, simd, partials);                                          \
  }
M5X_ML_DIAG(m5x_ml_sums_diag1_sg2_m32, 32, 2, 1)
M5X_ML_DIAG(m5x_ml_sums_diag2_sg2_m32, 32, 2, 2)
M5X_ML_DIAG(m5x_ml_sums_diag1_sg1_m24, 24, 1, 1)
M5X_ML_DIAG(m5x_ml_sums_diag2_sg1_m24, 24, 1, 2)
M5X_ML_DIAG(m5x_ml_sums_diag1_sg1_m8, 8, 1, 1)
#undef M5X_ML_DIAG

// Epilogue redesign, idea 1 (experiment): one-lane residual N32 split with the
// bias term taken out of the per-group loop. The loop accumulates only
// partial * scale; afterwards each partition adds sum x bias over its groups as
// matmul2d calls (float row sums [row][group] x bf16 biases read in place,
// fp32 accumulation). TransposeB selects the biases' layout interpretation.
template <ushort Parts, bool TransposeB>
inline void m5x_bias_matmul_split(device bfloat *input, device uchar *weights,
                                  device bfloat *scales, device bfloat *biases,
                                  device bfloat *residual, device bfloat *output,
                                  device const float *row_sums, constant Q4Params &p,
                                  uint group, uint lane, uint simd,
                                  threadgroup float *sums, threadgroup float *partials,
                                  threadgroup float *bias_tiles) {
  constexpr uint TileN = 32, Rows = 8, StorageN = 256;
  const uint G = p.input_size / 64;
  // Row sums arrive [group][row]; stage them [row][group] (K contiguous per row).
  for (uint i = simd * 32 + lane; i < G * Rows; i += Parts * 32) {
    uint g = i / Rows, row = i % Rows;
    sums[row * G + g] = row_sums[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint Gp = G / Parts, first_group = simd * Gp;
  for (uint tile = group; tile < p.output_size / TileN; tile += p.persistent_groups) {
    const uint output_origin = tile * TileN;
    const uint storage_tile = output_origin / StorageN, tile_offset = output_origin % StorageN;
    // 1. Main loop without the bias term (Diag 5 = scale-only epilogue).
    q4_mpp_tile_split<TileN, false, 256, true, 1, Parts, Rows, 2, true, 5>(
        input, weights, scales, biases, partials, weights, scales, biases,
        p.input_size, sums, output_origin, lane, 0, simd);
    // 2. Bias term: sums[rows x Gp] x biases[Gp x 32], one matmul with dynamic K.
    //    Tensor extents are (x, y) = (column, row). A = sums, x = group (stride 1),
    //    y = row (stride G). B = biases: NN form x = column (stride 1), y = group
    //    (stride StorageN); NT form (TransposeB) x = group, y = column.
    auto a = tensor(sums + first_group, dextents<int, 2>{int(Gp), int(Rows)}, array<int, 2>{1, int(G)});
    constexpr auto descriptor = matmul2d_descriptor(Rows, TileN, static_cast<int>(dynamic_extent), false,
                                                    TransposeB, false,
                                                    matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<1>> bias_op;
    device bfloat *bias_base = biases + (ulong(storage_tile) * G + first_group) * StorageN + tile_offset;
    auto b = TransposeB
        ? tensor(bias_base, dextents<int, 2>{int(Gp), int(TileN)}, array<int, 2>{int(StorageN), 1})
        : tensor(bias_base, dextents<int, 2>{int(TileN), int(Gp)}, array<int, 2>{1, int(StorageN)});
    auto bias_acc = bias_op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
    for (ushort i = 0; i < bias_acc.get_capacity(); ++i) bias_acc[i] = 0.0f;
    bias_op.run(a, b, bias_acc);
    // 3. Add the bias tile into this partition's partials ([row][col]).
    for (ushort i = 0; i < bias_acc.get_capacity(); ++i) {
      if (!bias_acc.is_valid_element(i)) continue;
      auto index = bias_acc.get_multidimensional_index(i);
      partials[simd * Rows * TileN + index[1] * TileN + index[0]] += bias_acc[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = simd * 32 + lane; i < Rows * TileN; i += Parts * 32) {
      float value = 0;
      for (uint part = 0; part < Parts; ++part) value += partials[part * Rows * TileN + i];
      uint index = (i / TileN) * p.output_size + tile * TileN + i % TileN;
      value = float(bfloat(value)) + float(residual[index]);
      output[index] = bfloat(value);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  (void)bias_tiles;
}
#define M5X_BIAS_MATMUL(Name, Parts, TransposeB)                               \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   device bfloat *residual [[buffer(4)]],                      \
                   device bfloat *output [[buffer(5)]],                        \
                   device const float *row_sums [[buffer(6)]],                 \
                   constant Q4Params &params [[buffer(7)]],                    \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint lane [[thread_index_in_simdgroup]],                    \
                   uint simd [[simdgroup_index_in_threadgroup]]) {             \
    threadgroup float sums[272 * 8], partials[Parts * 8 * 32], bias_tiles[1];  \
    m5x_bias_matmul_split<Parts, TransposeB>(input, weights, scales, biases,   \
        residual, output, row_sums, params, group, lane, simd, sums, partials, \
        bias_tiles);                                                           \
  }
M5X_BIAS_MATMUL(m5x_biasmm_t_sums_residual, 4, true)
M5X_BIAS_MATMUL(m5x_biasmm_n_sums_residual, 4, false)
#undef M5X_BIAS_MATMUL

// H18 experiment: the one-lane sums kernels with early scale/bias loads.
#define M5X_EARLY_RES(Name, Parts)                                             \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   device bfloat *residual [[buffer(4)]],                      \
                   device bfloat *output [[buffer(5)]],                        \
                   device const float *row_sums [[buffer(6)]],                 \
                   constant Q4Params &params [[buffer(7)]],                    \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint lane [[thread_index_in_simdgroup]],                    \
                   uint simd [[simdgroup_index_in_threadgroup]]) {             \
    threadgroup float sums[1], partials[Parts * 8 * 32];                 \
    m5x_split_device_sums<Parts, true, bfloat, true>(input, weights, scales,   \
        biases, residual, output, row_sums, params, group, lane, simd, sums,   \
        partials);                                                             \
  }
M5X_EARLY_RES(m5x_early_sums_residual, 4)
#undef M5X_EARLY_RES
#define M5X_EARLY_PLAIN(Name, Parts)                                           \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   device bfloat *output [[buffer(4)]],                        \
                   device const float *row_sums [[buffer(5)]],                 \
                   constant Q4Params &params [[buffer(6)]],                    \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint lane [[thread_index_in_simdgroup]],                    \
                   uint simd [[simdgroup_index_in_threadgroup]]) {             \
    threadgroup float sums[1], partials[Parts * 8 * 32];                 \
    m5x_split_device_sums<Parts, false, bfloat, true>(input, weights, scales,  \
        biases, input, output, row_sums, params, group, lane, simd, sums,      \
        partials);                                                             \
  }
M5X_EARLY_PLAIN(m5x_early_sums, 4)
#undef M5X_EARLY_PLAIN
#define M5X_SHUF(Name, Residual, Out4)                                         \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   device bfloat *Out4 [[buffer(4)]],                          \
                   device bfloat *output [[buffer(5)]],                        \
                   device const float *row_sums [[buffer(6)]],                 \
                   constant Q4Params &params [[buffer(7)]],                    \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint lane [[thread_index_in_simdgroup]],                    \
                   uint simd [[simdgroup_index_in_threadgroup]]) {             \
    threadgroup float sums[1], partials[4 * 8 * 32];                     \
    m5x_split_device_sums<4, Residual, bfloat, 2>(input, weights, scales,      \
        biases, Out4, output, row_sums, params, group, lane, simd, sums,       \
        partials);                                                             \
  }
M5X_SHUF(m5x_shuf_sums_residual, true, residual)
#undef M5X_SHUF
kernel void m5x_shuf_sums(device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],
                          device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]],
                          device bfloat *output [[buffer(4)]], device const float *row_sums [[buffer(5)]],
                          constant Q4Params &params [[buffer(6)]],
                          uint group [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup float sums[1], partials[4 * 8 * 32];
  m5x_split_device_sums<4, false, bfloat, 2>(input, weights, scales, biases, input, output, row_sums,
                                             params, group, lane, simd, sums, partials);
}
