#pragma once
#include "Export.hpp"
#include <cstddef>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace iiLocalDiffusion {

// A bounded header inspection identifies evidence, not successful loading or
// availability of the components required by a particular engine.
enum class ImageModelArchitecture { Unknown, MossImage, Anima };
enum class ImageModelRole { Unknown, Checkpoint, Lora, Vae, ControlNet };
enum class ImageWeightEncoding { Unknown, Floating, ComfyInt8, Other };

struct ImageModelInspection {
    ImageModelArchitecture architecture = ImageModelArchitecture::Unknown;
    ImageModelRole role = ImageModelRole::Unknown;
    ImageWeightEncoding encoding = ImageWeightEncoding::Unknown;
    bool hasTextEncoder = false;
    bool hasVae = false;
    std::size_t syntheticAdapterMappings = 0;
};

IILD_EXPORT ImageModelInspection inspectImageModel(const std::filesystem::path &path);

// Original manifest paths or shard-index paths, not extraction paths. Indexes
// are verified during materialization. Empty components are unestablished.
struct EmbeddedImageComponents {
    bool packaged = false;
    std::string transformer;
    std::string textEncoder;
    std::string vae;
};
IILD_EXPORT EmbeddedImageComponents inspectEmbeddedImageComponents(const std::filesystem::path &path);

// One original Safetensors source, indexed group, or GGUF asset. GGUF entries
// have no tensor mappings. This is bounded header evidence; materialization checks the
// selected source's index, descriptors and payload checksums separately.
struct EmbeddedImageGroup {
    std::string path;
    std::vector<std::pair<std::string, std::string>> tensorMappings;
    // True only when the package explicitly records a user-selected main
    // model for the entire source/group; automatic positions remain false.
    bool userSelectedMain = false;
};
struct EmbeddedImageCatalog {
    bool packaged = false;
    std::vector<EmbeddedImageGroup> groups;
};
IILD_EXPORT EmbeddedImageCatalog inspectEmbeddedImageCatalog(const std::filesystem::path &path);
struct EmbeddedImagePackageInspection {
    ImageModelInspection model;
    EmbeddedImageComponents components;
    EmbeddedImageCatalog catalog;
};
IILD_EXPORT EmbeddedImagePackageInspection inspectEmbeddedImagePackage(const std::filesystem::path &path);

}
