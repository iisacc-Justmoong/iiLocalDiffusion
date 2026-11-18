#pragma once

#include "Export.hpp"
#include <filesystem>
#include <string_view>

namespace iiLocalDiffusion {

enum class NativeTextEncoderState { Unsupported, Missing, Embedded, Incompatible };

// Check the pinned engine's actual parameter names and dimensions. Neither
// operation loads model payloads, tokenizers, or performs inference.
IILD_EXPORT NativeTextEncoderState inspectNativeTextEncoder(const std::filesystem::path &model,
                                                           std::string_view slot);
IILD_EXPORT bool validateNativeTextEncoder(const std::filesystem::path &model, std::string_view slot,
                                          const std::filesystem::path &candidate);

} // namespace iiLocalDiffusion
