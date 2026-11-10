#pragma once

#include <filesystem>

namespace iiLocalDiffusion::native_detail {
// Inspect bounded metadata; throws only for definite invalid or unsupported
// native image inputs and leaves unknown architectures to the native engine.
void inspectNativeImageModel(const std::filesystem::path &path);
bool embeddedAnimaTransformerNeedsExtraction(const std::filesystem::path &path);
}
