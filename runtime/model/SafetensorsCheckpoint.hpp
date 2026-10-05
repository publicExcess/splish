#pragma once

#include "model/PreparedWeights.hpp"

#include <memory>
#include <string_view>

namespace splash::model {

// Checkpoint configuration and safetensors index. Opening parses only
// metadata; tensor data is hashed when a prepared file's identity first
// needs it and read in bounded slices, without loading the MLX runtime or
// allocating tensors.
class SafetensorsCheckpoint final {
public:
  explicit SafetensorsCheckpoint(const std::filesystem::path &directory,
                        const PreparationCheck &check = {});
  ~SafetensorsCheckpoint();
  [[nodiscard]] const SourceTensor *find(std::string_view name) const noexcept;
  [[nodiscard]] const SourceTensor &require(std::string_view name) const;
  // Whether a tensor's name starts with prefix.
  [[nodiscard]] bool hasPrefix(std::string_view prefix) const noexcept;
  void requireQuantization(std::string_view projection, uint32_t bits) const;
  void requireConfigNumber(std::string_view key, double expected) const;
  void requireConfigString(std::string_view key, std::string_view expected) const;
  void requireLayerTypes(uint32_t layers, uint32_t fullAttentionPeriod) const;
  void checkUnchanged() const;
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace splash::model
