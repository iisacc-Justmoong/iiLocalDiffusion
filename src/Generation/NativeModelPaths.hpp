#pragma once

#include "ModelPackaging/ModelPackaging.hpp"
#include "ModelPackaging/ModelInspection.hpp"
#include <filesystem>

namespace iiLocalDiffusion::native_detail {

// Ordinary files retain the existing canonical-path rule. A linked index
// must keep its logical path so the native loader uses the same shard base as
// the SDK inspector. Callers must still validate and retain an index snapshot.
inline bool isCanonicalOrShardIndex(const std::filesystem::path &path) {
    if (!path.is_absolute() || path != path.lexically_normal() || !std::filesystem::is_regular_file(path)) return false;
    if (std::filesystem::canonical(path) == path) return true;
    return iild::detectModelContainer(path) == "safetensors-index";
}

}
