#pragma once

#include "Export.hpp"
#include <filesystem>
#include <string>

namespace iiLocalDiffusion {

// Metadata-only results: no tensor payloads are loaded or executed.
enum class NativeVaeState { Unsupported, Missing, Embedded, Incompatible };
struct NativeVaeInspection {
    NativeVaeState state;
    std::string modelFamily;
    std::string vaeFamily;
};

IILD_EXPORT NativeVaeInspection inspectNativeVae(const std::filesystem::path &model);
IILD_EXPORT bool validateNativeVae(const std::filesystem::path &model,
                                  const std::filesystem::path &candidate);

} // namespace iiLocalDiffusion
