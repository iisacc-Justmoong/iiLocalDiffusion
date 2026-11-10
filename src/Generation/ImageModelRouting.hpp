#pragma once
#include "Export.hpp"
#include <cstddef>
#include <filesystem>
#include <string>

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

// Original manifest paths, not extraction paths. Empty components were not
// established from a single, unambiguous source in the package.
struct EmbeddedImageComponents {
    bool packaged = false;
    std::string transformer;
    std::string textEncoder;
    std::string vae;
};
IILD_EXPORT EmbeddedImageComponents inspectEmbeddedImageComponents(const std::filesystem::path &path);

}
