// splash-m5 kernel bench: correctness and speed of Q4 Linear kernel choices on
// Swift-1.5's (Qwen3.8-27B) real decode shapes.
//
//   kernel-bench METALLIB [--shapes a,b] [--rows 8,16] [--pairs N] [--gb G]
//                [--candidates] [--config TILE:GROUPS:SIMDGROUPS[:SPLITS]]
//
// For every shape x rows it runs the device's default plan and, with
// --candidates, every candidate Splash would let tune-kernels try; --config
// adds an explicit configuration (a new kernel's tile once it exists).
//
// Correctness: each plan's bf16 output is checked against an fp64 reference
// with the rigorous per-element bound of dev/tests/engine/q4_sgmatrix_metal_test
// (sampled columns, every row), and run twice for bitwise determinism.
// Speed: each sample is one command of C projections over C distinct weight
// copies (C chosen so the copies exceed --gb, so weights are not cache
// resident), divided by C. Default and candidate samples alternate; the report
// gives median per-projection time, weight bandwidth and the median paired gain.

#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/Linear.hpp"
#include "tuning/LinearNumerics.hpp"
#include "tests/engine/GgufFormatReference.hpp"
#include "metal/abi/Linear.h"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace splash;
using namespace splash::ops;
using namespace splash::ops::tuning;

namespace {

struct Shape { std::string name; uint32_t n, k; LinearEpilogue epilogue; };
const std::vector<Shape> kShapes = {
    {"down", 5120, 17408, LinearEpilogue::Residual},     // FFN down (64 layers)
    {"gdn_out", 5120, 6144, LinearEpilogue::Residual},   // GDN out / attention out (64)
    {"gdn_in", 16640, 5120, LinearEpilogue::None},       // GDN in-projection (48)
    {"attn_qkv", 14336, 5120, LinearEpilogue::None},     // attention q/k/v (16)
    {"gate_up", 17408, 5120, LinearEpilogue::GateUp},    // FFN gate+up (64)
    {"ffn_gate", 17408, 5120, LinearEpilogue::None},    // FFN gate alone (B3/B4 split gate/up)
    {"draft_dyn", 1280, 5120, LinearEpilogue::None},    // DFlash draft attention/MLP dynamic (10 per step)
    {"draft_down", 5120, 17408, LinearEpilogue::None},  // DFlash draft down (plain; 5 per step)
    {"draft_out", 5120, 4096, LinearEpilogue::None},    // DFlash draft attention output (5 per step)
    {"plain_6144", 6144, 5120, LinearEpilogue::None},   // one-lane plain 6144 x 5120 (10 per step)
    {"plain_256", 256, 5120, LinearEpilogue::None},     // one-lane plain 256 x 5120 (1 per step)
    {"plain_k25600", 5120, 25600, LinearEpilogue::None}, // one-lane plain 5120 x 25600 (1 per step)
    // Qwen3.6-35B-A3B dense projections (tuning/m5max-40c-qwen36-35b.choices)
    {"q36_12544", 12544, 2048, LinearEpilogue::None},
    {"q36_9216", 9216, 2048, LinearEpilogue::None},
    {"q36_6144", 6144, 2048, LinearEpilogue::None},
    {"q36_gate_up", 6144, 2048, LinearEpilogue::GateUp},
    {"q36_512", 512, 2048, LinearEpilogue::None},
    {"q36_256", 256, 2048, LinearEpilogue::None},
    {"q36_k16384", 2048, 16384, LinearEpilogue::None},
    {"q36_k6144", 2048, 6144, LinearEpilogue::None},
    {"q36_k4096", 2048, 4096, LinearEpilogue::None},
    {"q36_k4096_res", 2048, 4096, LinearEpilogue::Residual},
    {"q36_head", 248320, 2048, LinearEpilogue::None},
    {"diag_res_k5120", 16640, 5120, LinearEpilogue::Residual},  // diagnostic only
    {"diag_plain_k17408", 5120, 17408, LinearEpilogue::None},   // diagnostic only
};

std::string tileName(LinearTile t) {
  switch (t) {
  case LinearTile::N128: return "n128";
  case LinearTile::N256: return "n256";
  case LinearTile::Paired128: return "paired128";
  case LinearTile::Split32: return "split32";
  case LinearTile::Split64: return "split64";
  case LinearTile::Paired256: return "paired256";
  case LinearTile::Simdgroup: return "simdgroup";
  case LinearTile::SplitSums32: return "splitsums32";
  case LinearTile::Deep256: return "deep256";
  case LinearTile::GgufStaged: return "ggufstaged";
  case LinearTile::GgufRegister: return "ggufregister";
  default: return "other";
  }
}
LinearTile tileFrom(const std::string &s) {
  for (auto t : {LinearTile::N128, LinearTile::N256, LinearTile::Paired128, LinearTile::Split32,
                 LinearTile::Split64, LinearTile::Paired256, LinearTile::Simdgroup, LinearTile::SplitSums32, LinearTile::Deep256,
                 LinearTile::GgufStaged, LinearTile::GgufRegister})
    if (tileName(t) == s) return t;
  throw std::invalid_argument("unknown tile " + s);
}
std::string describe(const LinearConfig &c) {
  std::ostringstream o;
  o << tileName(c.tile) << ":" << c.groups << ":" << int(c.simdgroups);
  if (c.splits != 1) o << ":" << c.splits;
  return o.str();
}

uint32_t hash(uint32_t v) { v ^= v >> 16; v *= 0x7feb352d; v ^= v >> 15; return v * 0x846ca68b; }

Projection makeWeights(metal::MetalBackend &backend, LinearMatrix shape, uint32_t seed) {
  const uint64_t params = uint64_t(shape.outputSize) * shape.inputSize / 64;
  Projection p(shape.outputSize, shape.inputSize,
               AffineWeights{backend.allocateBuffer(params * 32), backend.allocateBuffer(params * 2),
                             backend.allocateBuffer(params * 2)});
  auto *q = static_cast<uint8_t *>(p.affine().weights.contents());
  auto *s = static_cast<uint16_t *>(p.affine().scales.contents());
  auto *b = static_cast<uint16_t *>(p.affine().biases.contents());
  for (uint64_t i = 0; i < params * 32; ++i) q[i] = uint8_t(hash(uint32_t(i) + seed));
  for (uint64_t i = 0; i < params; ++i) {
    s[i] = floatToBf16((int(hash(uint32_t(i) + seed + 7) % 17) - 8) / 2048.0f);
    b[i] = floatToBf16(-float(hash(uint32_t(i) + seed + 11) % 16) * bf16ToFloat(s[i]));
  }
  return p;
}

// GGUF (block) weights built with Splash's own fixture (dev/tests/engine/
// GgufFormatReference.hpp): random native GGUF rows, repacked into the prepared
// planes the kernels read. The native rows stay for the fp64 reference.
struct GgufWeights { Projection projection; std::vector<uint8_t> native; };
std::mt19937 ggufRng{20260926};
GgufWeights makeGguf(metal::MetalBackend &backend, gguf_reference::Fmt f, uint32_t n, uint32_t k) {
  using namespace gguf_reference;
  std::vector<uint8_t> native = makeNative(f, n, k, ggufRng);
  const Packed planes = repack(f, native, n, k, nullptr);
  const auto upload = [&](const std::vector<uint8_t> &bytes) {
    metal::MetalBuffer buffer = backend.allocateBuffer(bytes.size());
    std::memcpy(buffer.contents(), bytes.data(), bytes.size());
    return buffer;
  };
  BlockWeights blocks;
  blocks.segments.push_back(QuantizedSegment::planes(
      f, n, k, upload(planes.w0), kQuantFormats[f].plane1_bytes ? upload(planes.w1) : metal::MetalBuffer{},
      upload(planes.meta)));
  return {Projection(n, k, std::move(blocks)), std::move(native)};
}

// fp64 reference and bound, as in dev/tests/engine/q4_sgmatrix_metal_test.mm.
struct Exact { double value, quantMagnitude, magnitude; };
Exact exact(const Projection &p, const uint16_t *input, uint32_t row, uint32_t col) {
  const auto *q = static_cast<const uint8_t *>(p.affine().weights.contents());
  const auto *sc = static_cast<const uint16_t *>(p.affine().scales.contents());
  const auto *bi = static_cast<const uint16_t *>(p.affine().biases.contents());
  double value = 0, magnitude = 0, quantMagnitude = 0;
  const uint32_t groups = p.inputSize / 64;
  for (uint32_t g = 0; g < groups; ++g) {
    const uint64_t at = (uint64_t(col / 256) * groups + g) * 256 + col % 256;
    double dot = 0, sum = 0, absolute = 0;
    for (uint32_t k = 0; k < 64; ++k) {
      const double x = bf16ToFloat(input[uint64_t(row) * p.inputSize + g * 64 + k]);
      const uint32_t nibble = (q[at * 32 + k / 2] >> (4 * (k % 2))) & 15;
      dot += x * nibble; sum += x; absolute += std::abs(x);
    }
    const double scale = bf16ToFloat(sc[at]), bias = bf16ToFloat(bi[at]);
    value += scale * dot + bias * sum;
    quantMagnitude += absolute * std::abs(scale);
    magnitude += absolute * (15 * std::abs(scale) + std::abs(bias));
  }
  return {value, quantMagnitude, magnitude};
}
struct Reference { double value, error; };
Reference reference(const Exact &e, uint32_t groups, uint32_t splits) {
  constexpr double u = 0x1p-24;
  const auto gamma = [&](double n) { return n * u / (1 - n * u); };
  const double error = gamma(72) * 143 * e.quantMagnitude + gamma(2 * groups + splits) * e.magnitude;
  return {double(bf16ToFloat(floatToBf16(float(e.value)))), error + ulpBf16(float(e.value))};
}
Reference withEpilogue(Reference ref, LinearEpilogue epilogue, double residual, Reference gate) {
  if (epilogue == LinearEpilogue::Residual) ref.value += residual;
  if (epilogue == LinearEpilogue::GateUp) {
    const double activation = gate.value / (1 + std::exp(-gate.value));
    ref.error = 1.1 * gate.error * (std::abs(ref.value) + ref.error) + std::abs(activation) * ref.error;
    ref.value *= activation;
  }
  return ref;
}
bool within(const Reference &ref, uint16_t actual) {
  const double expected = bf16ToFloat(floatToBf16(float(ref.value)));
  const double value = bf16ToFloat(actual);
  return std::isfinite(value) &&
         std::abs(value - expected) <= ref.error + ulpBf16(float(expected)) + ulpBf16(float(value));
}

struct Options {
  std::vector<std::string> shapes;
  std::vector<uint32_t> rows{8, 16, 24, 32};
  uint32_t pairs = 15;
  double gigabytes = 1.5;
  bool candidates = false;
  std::vector<LinearConfig> configs;
  struct Raw { std::string name; uint32_t tileN, parts, threads = 0; };
  std::vector<Raw> raws;
  std::string baseline;
  bool checkOnly = false;
  std::string gguf;  // GGUF format name (e.g. q8_0): block weights instead of affine Q4
};

std::vector<uint32_t> parseList(const std::string &s) {
  std::vector<uint32_t> out; std::stringstream ss(s); std::string item;
  while (std::getline(ss, item, ',')) out.push_back(uint32_t(std::stoul(item)));
  return out;
}
LinearConfig parseConfig(const std::string &s) {
  std::vector<std::string> f; std::stringstream ss(s); std::string item;
  while (std::getline(ss, item, ':')) f.push_back(item);
  if (f.size() < 3 || f.size() > 4) throw std::invalid_argument("--config TILE:GROUPS:SIMDGROUPS[:SPLITS]");
  LinearConfig c;
  c.tile = tileFrom(f[0]);
  c.groups = uint32_t(std::stoul(f[1]));
  c.simdgroups = static_cast<LinearSimdgroups>(std::stoul(f[2]));
  c.splits = f.size() == 4 ? uint32_t(std::stoul(f[3])) : 1;
  return c;
}

struct Buffers {
  metal::MetalBuffer input, output, residual, table, sums, partials, counters, gateScratch, rowSums;
};

LinearBuffers linearBuffers(const Buffers &b, const LinearPlan &plan, LinearEpilogue epilogue) {
  LinearBuffers lb;
  lb.input = b.input;
  lb.output = b.output;
  if (epilogue == LinearEpilogue::Residual) lb.residual = b.residual;
  if (plan.gateScratchBytes()) lb.gateScratch = b.gateScratch;
  if (plan.usesSimdgroup() || plan.configuration().tile == LinearTile::SplitSums32 ||
      plan.workload().weightLayout == WeightLayout::Block32)
    lb.scratch = LinearScratch{b.table, b.sums, b.partials, b.counters};
  return lb;
}

// One way to encode a projection: a planned Linear configuration or a raw
// experimental residual kernel (m5x_* in linear_q4_split.metal).
struct Variant {
  std::string label, pipeline;
  uint32_t splits = 4;  // K partitions, for the error bound
  std::function<void(metal::CommandGraph &, uint32_t copy)> encode;
};

// Correctness of one variant: sampled columns x every row, then bitwise repeat.
using ReferenceOf = std::function<Reference(bool gate, uint32_t row, uint32_t col, uint32_t splits)>;
std::string check(metal::MetalBackend &backend, const Variant &v, const Shape &shape, uint32_t rows,
                  const ReferenceOf &referenceOf, const Buffers &b) {
  metal::CommandGraph graph;
  v.encode(graph, 0);
  (void)backend.submitCommand(graph.dispatches());
  const auto *out = static_cast<const uint16_t *>(b.output.contents());
  const std::vector<uint16_t> first(out, out + uint64_t(rows) * shape.n);
  (void)backend.submitCommand(graph.dispatches());
  if (std::memcmp(first.data(), out, 2ULL * rows * shape.n) != 0) return "NONDETERMINISTIC";
  // SPLASH_M5_DUMP=DIR: the output, to compare builds bit for bit.
  if (const char *dump = std::getenv("SPLASH_M5_DUMP")) {
    const std::string path = std::string(dump) + "/" + shape.name + "_r" + std::to_string(rows) + "_" + v.label;
    if (FILE *f = std::fopen(path.c_str(), "wb")) { std::fwrite(first.data(), 2, first.size(), f); std::fclose(f); }
  }
  const auto *r = static_cast<const uint16_t *>(b.residual.contents());
  const uint32_t splits = std::max<uint32_t>(v.splits, 4);
  uint32_t checked = 0;
  for (uint32_t col = 0; col < shape.n; col += 37) {
    for (uint32_t row = 0; row < rows; ++row) {
      const uint64_t i = uint64_t(row) * shape.n + col;
      Reference gateRef{};
      if (shape.epilogue == LinearEpilogue::GateUp) gateRef = referenceOf(true, row, col, splits);
      const auto ref = withEpilogue(referenceOf(false, row, col, splits), shape.epilogue, bf16ToFloat(r[i]), gateRef);
      if (!within(ref, first[i])) {
        std::ostringstream o;
        o << "FAIL row " << row << " col " << col << " actual " << bf16ToFloat(first[i])
          << " reference " << ref.value << " bound " << ref.error;
        return o.str();
      }
      ++checked;
    }
  }
  return "ok (" + std::to_string(checked) + " outputs, deterministic)";
}

double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int run(const std::string &metallib, const Options &options) {
  int failures = 0;
  metal::MetalBackend backend(metallib);
  const auto &device = backend.capabilities();
  Linear linear(device);
  std::cout << "kernel-bench: " << device.deviceName << " (Apple GPU family " << device.appleGpuFamily
            << ", " << device.gpuCoreCount << " cores)\n";
  for (const Shape &shape : kShapes) {
    if (!options.shapes.empty() &&
        std::find(options.shapes.begin(), options.shapes.end(), shape.name) == options.shapes.end())
      continue;
    const LinearMatrix matrix{shape.n, shape.k};
    const bool gated = shape.epilogue == LinearEpilogue::GateUp;
    const bool gguf = !options.gguf.empty();
    const gguf_reference::Fmt format = gguf ? gguf_reference::fmtNamed(options.gguf) : gguf_reference::Fmt{};
    const double bytesPerProjection =
        (gguf ? double(gguf_reference::rowBytes(format, shape.k)) * shape.n
              : double(shape.n) * shape.k / 2 + double(shape.n) * (shape.k / 64) * 4) * (gated ? 2 : 1);
    const uint32_t copies = std::clamp<uint32_t>(
        uint32_t(std::ceil(options.gigabytes * 1e9 / bytesPerProjection)), 4, 96);
    std::vector<Projection> ws, gs;
    std::vector<uint8_t> nativeUp, nativeGate;  // copy 0's GGUF rows, for the reference
    for (uint32_t c = 0; c < copies; ++c) {
      if (gguf) {
        GgufWeights up = makeGguf(backend, format, shape.n, shape.k);
        if (c == 0) nativeUp = up.native;
        ws.push_back(std::move(up.projection));
        if (gated) {
          GgufWeights gate = makeGguf(backend, format, shape.n, shape.k);
          if (c == 0) nativeGate = gate.native;
          gs.push_back(std::move(gate.projection));
        }
      } else {
        ws.push_back(makeWeights(backend, matrix, 31 + c));
        if (gated) gs.push_back(makeWeights(backend, matrix, 177 + c));
      }
    }
    // Buffers hold the plans' storage rows: staged GGUF plans pad 24 rows to 32, as the engine sizes them.
    const uint32_t maxRows = (*std::max_element(options.rows.begin(), options.rows.end()) + 31) / 32 * 32;
    Buffers b;
    b.input = backend.allocateBuffer(2ULL * maxRows * shape.k);
    b.output = backend.allocateBuffer(2ULL * maxRows * shape.n);
    b.residual = backend.allocateBuffer(2ULL * maxRows * shape.n);
    b.rowSums = backend.allocateBuffer(4ULL * maxRows * (shape.k / 64));
    auto *x = static_cast<uint16_t *>(b.input.contents());
    auto *r = static_cast<uint16_t *>(b.residual.contents());
    for (uint64_t i = 0; i < uint64_t(maxRows) * shape.k; ++i)
      x[i] = floatToBf16(float(int(hash(uint32_t(i) + 37) % 257) - 128) / 32);
    for (uint64_t i = 0; i < uint64_t(maxRows) * shape.n; ++i)
      r[i] = floatToBf16(float(int(i % 31) - 15) / 8);
    std::cout << "\n== " << shape.name << " " << shape.n << "x" << shape.k << ", " << copies
              << " weight copies (" << std::fixed << std::setprecision(2)
              << copies * bytesPerProjection / 1e9 << " GB) ==\n";
    for (uint32_t rows : options.rows) {
      const LinearWorkload w{matrix, rows, LinearPhase::Decode, shape.epilogue,
                             gguf ? WeightLayout::Block32 : WeightLayout::Affine64};
      std::vector<LinearPlan> plans{gguf ? linear.plan(w, ws[0], gated ? &gs[0] : nullptr) : linear.plan(w)};
      if (options.candidates)
        for (const auto &c : linear.candidates(w))
          if (!(c.configuration() == plans[0].configuration())) plans.push_back(c);
      for (const auto &config : options.configs) {
        try { plans.push_back(Linear::plan(w, config)); }
        catch (const std::exception &e) {
          std::cout << "  rows " << rows << " " << describe(config) << ": not valid here (" << e.what() << ")\n";
        }
      }
      // Scratch for the largest simdgroup plan.
      LinearScratchSize scratch{};
      for (const auto &plan : plans) {
        const auto s = plan.scratchSize();
        scratch.input = std::max(scratch.input, s.input); scratch.sums = std::max(scratch.sums, s.sums);
        scratch.partials = std::max(scratch.partials, s.partials);
        scratch.counters = std::max(scratch.counters, s.counters);
      }
      b.table = backend.allocateBuffer(std::max<uint64_t>(scratch.input, 16));
      b.sums = backend.allocateBuffer(std::max<uint64_t>(scratch.sums, 16));
      b.partials = backend.allocateBuffer(std::max<uint64_t>(scratch.partials, 16));
      b.counters = backend.allocateBuffer(std::max<uint64_t>(scratch.counters, 16));
      std::memset(b.counters.contents(), 0, b.counters.sizeBytes());
      uint64_t gateBytes = 16;
      for (const auto &plan : plans) gateBytes = std::max(gateBytes, plan.gateScratchBytes());
      b.gateScratch = backend.allocateBuffer(gateBytes);

      std::vector<Variant> variants;
      for (size_t i = 0; i < plans.size(); ++i) {
        const LinearPlan plan = plans[i];
        variants.push_back({describe(plan.configuration()) + (i == 0 ? " (default)" : ""),
                            std::string(plan.pipeline()), plan.partialSums(),
                            [&, plan](metal::CommandGraph &g, uint32_t c) {
                              linear.add(g, linearBuffers(b, plan, shape.epilogue), ws[c], plan,
                                         gated ? &gs[c] : nullptr);
                            }});
      }
      for (const auto &raw : options.raws) {
        if (gguf) break;  // raw kernels bind affine weights
        // One-lane raws, or multi-lane raws named _m<rows>.
        if (rows != 8 && raw.name.find("_m" + std::to_string(rows)) == std::string::npos) continue;
        if (rows == 8 && raw.name.find("_m16") != std::string::npos) continue;
        if (rows == 8 && (raw.name.find("_m24") != std::string::npos || raw.name.find("_m32") != std::string::npos)) continue;
        if (shape.epilogue == LinearEpilogue::GateUp && raw.name.find("gate_up") == std::string::npos) continue;
        if (shape.epilogue == LinearEpilogue::None && raw.name.find("_devsums") == std::string::npos && raw.name.find("_sums") == std::string::npos) continue;
        // A sums kernel's buffer contract follows its epilogue: never bind a
        // residual kernel to a plain shape or the reverse (garbage parameters).
        if (raw.name.find("_sums") != std::string::npos &&
            (raw.name.find("residual") != std::string::npos) != (shape.epilogue == LinearEpilogue::Residual) &&
            shape.epilogue != LinearEpilogue::GateUp) {
          std::cout << "  rows " << rows << " " << raw.name << ": epilogue does not match shape, skipped\n";
          continue;
        }
        if (shape.k % (256 * raw.parts) || shape.n % raw.tileN) {
          std::cout << "  rows " << rows << " " << raw.name << ": K or N does not divide\n";
          continue;
        }
        const uint32_t groups = shape.n / raw.tileN, threads = raw.threads ? raw.threads : raw.parts * 32;
        // *_devsums kernels read row sums a preceding m5x_row_sums8 dispatch writes
        // (once per projection, timed with it).
        const bool deviceSums = raw.name.find("_devsums") != std::string::npos || raw.name.find("_sums") != std::string::npos;
        variants.push_back({raw.name + (deviceSums ? " (+sums)" : ""), raw.name, raw.parts,
                            [&, raw, groups, threads, deviceSums, rows](metal::CommandGraph &g, uint32_t c) {
                              const auto &a = ws[c].affine();
                              if (deviceSums) {
                                if (rows == 8)
                                  g.add("m5x_row_sums8", {b.input, b.rowSums}, Q4Params{shape.n, shape.k, 0},
                                        {shape.k / 64, 1, 1}, {256, 1, 1});
                                else
                                  g.add("decode_linear_q4_row_sums", {b.input, b.rowSums}, Q4Params{shape.n, shape.k, 0},
                                        {shape.k / 64, 1, 1}, {rows * 32, 1, 1});
                                if (shape.epilogue == LinearEpilogue::GateUp) {
                                  const auto &u = gs[c].affine();  // bench: gs = gate, ws = up
                                  g.add(raw.name, {b.input, u.weights, u.scales, u.biases, b.output, a.weights,
                                                   a.scales, a.biases, b.rowSums},
                                        Q4Params{shape.n, shape.k, groups}, {groups, 1, 1}, {threads, 1, 1});
                                } else if (shape.epilogue == LinearEpilogue::Residual)
                                  g.add(raw.name, {b.input, a.weights, a.scales, a.biases, b.residual, b.output,
                                                   b.rowSums},
                                        Q4Params{shape.n, shape.k, groups}, {groups, 1, 1}, {threads, 1, 1});
                                else
                                  g.add(raw.name, {b.input, a.weights, a.scales, a.biases, b.output, b.rowSums},
                                        Q4Params{shape.n, shape.k, groups}, {groups, 1, 1}, {threads, 1, 1});
                              } else {
                                g.add(raw.name, {b.input, a.weights, a.scales, a.biases, b.residual, b.output},
                                      Q4Params{shape.n, shape.k, groups}, {groups, 1, 1}, {threads, 1, 1});
                              }
                            }});
      }

      auto timeOnce = [&](const Variant &v) {
        metal::CommandGraph graph;
        for (uint32_t c = 0; c < copies; ++c) v.encode(graph, c);
        return backend.submitCommand(graph.dispatches()).gpuSeconds / copies;
      };
      // The reference for gains: the named --baseline variant, else the default.
      size_t baseIndex = 0;
      for (size_t i = 0; i < variants.size(); ++i)
        if (!options.baseline.empty() && variants[i].label.rfind(options.baseline, 0) == 0) baseIndex = i;
      const Variant &base = variants[baseIndex];
      if (!options.checkOnly)
        for (int warm = 0; warm < 2; ++warm) (void)timeOnce(base);
      for (size_t index = 0; index < variants.size(); ++index) {
        const Variant &v = variants[index];
        std::string verdict;
        const auto *x = static_cast<const uint16_t *>(b.input.contents());
        std::map<std::pair<bool, uint32_t>, std::vector<float>> rowCache;  // decoded GGUF rows
        const ReferenceOf referenceOf = [&](bool isGate, uint32_t row, uint32_t col, uint32_t splits) {
          if (!gguf)
            return reference(exact(isGate ? gs[0] : ws[0], x, row, col), shape.k / 64, splits);
          using namespace gguf_reference;
          auto &wv = rowCache[{isGate, col}];
          if (wv.empty()) {
            wv.resize(shape.k);
            const auto &native = isGate ? nativeGate : nativeUp;
            rowValues(format, native.data() + uint64_t(col) * rowBytes(format, shape.k), shape.k, wv.data());
          }
          std::vector<float> xv(shape.k);
          for (uint32_t kk = 0; kk < shape.k; ++kk) xv[kk] = bf16ToFloat(x[uint64_t(row) * shape.k + kk]);
          const Dot d = dot(xv.data(), wv.data(), shape.k);
          // The staged tile's bound (it is the larger), then the bf16 rounding.
          return Reference{double(bf16ToFloat(floatToBf16(float(d.value)))),
                           projectionBound(d, true) + ulpBf16(float(d.value))};
        };
        try { verdict = check(backend, v, shape, rows, referenceOf, b); }
        catch (const std::exception &e) {
          std::cout << "  rows " << rows << "  " << v.label << ": " << e.what() << '\n';
          ++failures;
          continue;
        }
        if (verdict.rfind("ok", 0) != 0) ++failures;
        if (options.checkOnly) {
          std::cout << "  rows " << std::setw(2) << rows << "  " << std::left << std::setw(24) << v.label
                    << std::right << "  " << v.pipeline << "  " << verdict << '\n';
          continue;
        }
        std::vector<double> times, gains;
        if (index == baseIndex) {
          for (uint32_t i = 0; i < options.pairs; ++i) times.push_back(timeOnce(v));
        } else {
          for (uint32_t i = 0; i < options.pairs; ++i) {
            double tb, tc;
            if (i % 2) { tc = timeOnce(v); tb = timeOnce(base); }
            else { tb = timeOnce(base); tc = timeOnce(v); }
            times.push_back(tc); gains.push_back(tb / tc - 1);
          }
        }
        const double t = median(times);
        std::cout << "  rows " << std::setw(2) << rows << "  " << std::left << std::setw(24)
                  << (v.label + (index == baseIndex && baseIndex ? " (baseline)" : "")) << std::right
                  << std::setw(9) << std::setprecision(3) << t * 1e3 << " ms  " << std::setw(5)
                  << std::setprecision(0) << bytesPerProjection / t / 1e9 << " GB/s  ";
        if (index != baseIndex) std::cout << std::showpos << std::setprecision(1) << median(gains) * 100 << "%" << std::noshowpos << "  ";
        else std::cout << "         ";
        std::cout << v.pipeline << "  " << verdict << '\n';
      }
    }
  }
  std::cout << (failures ? "\nkernel-bench: " + std::to_string(failures) + " FAILED\n" : std::string("\nkernel-bench: all checks passed\n"));
  return failures ? 1 : 0;
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    try {
      if (argc < 2) throw std::invalid_argument(
          "usage: kernel-bench METALLIB [--shapes a,b] [--rows 8,16] [--pairs N] [--gb G] "
          "[--candidates] [--config TILE:GROUPS:SIMDGROUPS[:SPLITS]] [--raw KERNEL:TILEN:PARTS] [--baseline LABEL]");
      Options options;
      for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        const bool v = i + 1 < argc;
        if (a == "--shapes" && v) { std::stringstream ss(argv[++i]); std::string s; while (std::getline(ss, s, ',')) options.shapes.push_back(s); }
        else if (a == "--rows" && v) options.rows = parseList(argv[++i]);
        else if (a == "--pairs" && v) options.pairs = uint32_t(std::stoul(argv[++i]));
        else if (a == "--gb" && v) options.gigabytes = std::stod(argv[++i]);
        else if (a == "--candidates") options.candidates = true;
        else if (a == "--check") { options.checkOnly = true; options.candidates = true; }
        else if (a == "--config" && v) options.configs.push_back(parseConfig(argv[++i]));
        else if (a == "--raw" && v) {
          std::vector<std::string> f; std::stringstream ss(argv[++i]); std::string item;
          while (std::getline(ss, item, ':')) f.push_back(item);
          if (f.size() != 3 && f.size() != 4) throw std::invalid_argument("--raw KERNEL:TILEN:PARTS[:THREADS]");
          options.raws.push_back({f[0], uint32_t(std::stoul(f[1])), uint32_t(std::stoul(f[2])),
                                  f.size() == 4 ? uint32_t(std::stoul(f[3])) : 0u});
        }
        else if (a == "--baseline" && v) options.baseline = argv[++i];
        else if (a == "--gguf" && v) options.gguf = argv[++i];
        else throw std::invalid_argument("unknown option " + a);
      }
      return run(argv[1], options);
    } catch (const std::exception &e) {
      std::cerr << "FAIL: " << e.what() << '\n';
      return 1;
    }
  }
  return 0;
}
