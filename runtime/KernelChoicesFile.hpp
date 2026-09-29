#pragma once

// splash-m5: load measured Linear kernel choices from a text file named by
// SPLASH_KERNEL_CHOICES, so a device can be re-tuned without a rebuild.
//
// One choice per line, whitespace-separated; '#' starts a comment:
//
//   out  in  rows  phase   epilogue  tile     groups  simdgroups  [splits]
//   5120 17408 8   decode  residual  split32  160     4
//
// phase: prefill | decode.  epilogue: none | residual | gateup | upwithgate.
// tile: n128 | n256 | paired128 | split32 | split64 | paired256 | simdgroup | splitsums32 |
//       ggufstaged | ggufregister (these two key GGUF / block-quantized workloads).
// Only the Affine64 weight layout is covered (the layout of Splash packages
// and MLX 4-bit checkpoints). Any malformed line fails startup loudly.

#include "ops/ExecutionPlans.hpp"

#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace splash::m5 {

namespace detail {

template <typename T, size_t N>
T lookup(std::string_view word, const std::pair<std::string_view, T> (&table)[N],
         const std::string &where) {
  for (const auto &[name, value] : table)
    if (word == name) return value;
  throw std::invalid_argument(where + ": unknown value '" + std::string(word) + "'");
}

} // namespace detail

inline ops::OperatorChoices loadKernelChoices(const std::string &path) {
  using namespace ops;
  static const std::pair<std::string_view, LinearPhase> phases[] = {
      {"prefill", LinearPhase::Prefill}, {"decode", LinearPhase::Decode}};
  static const std::pair<std::string_view, LinearEpilogue> epilogues[] = {
      {"none", LinearEpilogue::None}, {"residual", LinearEpilogue::Residual},
      {"gateup", LinearEpilogue::GateUp}, {"upwithgate", LinearEpilogue::UpWithGate}};
  static const std::pair<std::string_view, LinearTile> tiles[] = {
      {"n128", LinearTile::N128}, {"n256", LinearTile::N256},
      {"paired128", LinearTile::Paired128}, {"split32", LinearTile::Split32},
      {"split64", LinearTile::Split64}, {"paired256", LinearTile::Paired256},
      {"simdgroup", LinearTile::Simdgroup}, {"splitsums32", LinearTile::SplitSums32}, {"deep256", LinearTile::Deep256},
      {"ggufstaged", LinearTile::GgufStaged}, {"ggufregister", LinearTile::GgufRegister}};

  std::ifstream file(path);
  if (!file) throw std::runtime_error("cannot read kernel choices file " + path);
  OperatorChoices choices;
  std::string line;
  for (unsigned number = 1; std::getline(file, line); ++number) {
    if (const auto hash = line.find('#'); hash != std::string::npos) line.erase(hash);
    std::istringstream fields(line);
    std::string first;
    if (!(fields >> first)) continue;
    const std::string where = path + ":" + std::to_string(number);
    std::string phase, epilogue, tile;
    uint32_t in = 0, rows = 0, groups = 0, simdgroups = 0, splits = 1;
    if (!(fields >> in >> rows >> phase >> epilogue >> tile >> groups >> simdgroups))
      throw std::invalid_argument(where + ": expected 8 or 9 fields");
    fields >> splits;
    std::string extra;
    if (fields >> extra) throw std::invalid_argument(where + ": too many fields");
    if (simdgroups != 2 && simdgroups != 4 && simdgroups != 8)
      throw std::invalid_argument(where + ": simdgroups must be 2, 4 or 8");
    LinearChoice choice;
    choice.workload.matrix = {static_cast<uint32_t>(std::stoul(first)), in};
    choice.workload.rows = rows;
    choice.workload.phase = detail::lookup(phase, phases, where);
    choice.workload.epilogue = detail::lookup(epilogue, epilogues, where);
    choice.configuration.tile = detail::lookup(tile, tiles, where);
    // GGUF tiles key block-quantized (GGUF) workloads.
    if (choice.configuration.tile == LinearTile::GgufStaged || choice.configuration.tile == LinearTile::GgufRegister)
      choice.workload.weightLayout = WeightLayout::Block32;
    choice.configuration.groups = groups;
    choice.configuration.simdgroups = static_cast<LinearSimdgroups>(simdgroups);
    choice.configuration.splits = splits;
    choices.linear.push_back(choice);
  }
  return choices;
}

// The choices named by SPLASH_KERNEL_CHOICES, or none when it is unset.
inline std::optional<ops::OperatorChoices> kernelChoicesFromEnvironment() {
  const char *path = std::getenv("SPLASH_KERNEL_CHOICES");
  if (!path || !*path) return std::nullopt;
  return loadKernelChoices(path);
}

} // namespace splash::m5
