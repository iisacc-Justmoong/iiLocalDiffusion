#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace iild::detail {
// Some converters use these diffusion-loader tags for models that are not
// Wan video (including image models); they establish a role, not a modality.
[[nodiscard]] inline bool isWanCompatibilityGgufArchitecture(std::string_view value) noexcept {
    return value == "wan" || value == "wan2.1" || value == "wan2.2";
}
// GGUF architecture identifiers with diffusion denoisers as their primary
// component. Shared by catalogue inspection and package role detection.
[[nodiscard]] inline bool isDiffusionGgufArchitecture(std::string_view value) noexcept {
    return value == "ltxv" || value == "flux" || value == "sd1" || value == "sd2" ||
           value == "sdxl" || value == "sd3" || value == "qwen_image" ||
           value == "z_image" || value == "anima" || isWanCompatibilityGgufArchitecture(value);
}

struct GgufCatalog {
    std::string metadataJson;
    std::vector<std::string> tensorNames;
    std::uint64_t metadataBytesRead = 0;
};

// Header-only discovery. An unknown quantization type is preserved for
// classification and is not asserted to be supported by an inference engine.
[[nodiscard]] GgufCatalog inspectGgufCatalog(const std::filesystem::path &path);
struct EmbeddedGgufMain {
    std::filesystem::path relativePath;
    GgufCatalog catalog;
};
// A package's selected main GGUF, read in place with an asset-bounded header
// parser. No payload hashes are computed and no files are materialized.
[[nodiscard]] std::optional<EmbeddedGgufMain> inspectEmbeddedMainGguf(const std::filesystem::path &path);
}
