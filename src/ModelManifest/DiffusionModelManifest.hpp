#pragma once

#include "Flux/FluxModelManifest.hpp"
#include "StableDiffusion/StableDiffusionModelManifest.hpp"

#include <filesystem>
#include <variant>

namespace iild
{

using DiffusionModelManifest = std::variant<StableDiffusionModelManifest, FluxModelManifest>;

[[nodiscard]] IILD_EXPORT DiffusionModelManifest loadModelManifest(
    const std::filesystem::path &modelRoot);

} // namespace iild
