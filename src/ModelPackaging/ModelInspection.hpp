#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace iild {

// Output modalities are independent of storage roles (Checkpoint, LoRA, VAE).
// Inspection reads bounded metadata only; it never loads or hashes weight data.
struct ModelInspection {
    std::string architecture;
    std::string family;
    std::string version;
    std::vector<std::string> detectedModalities;
    std::vector<std::string> modalities;
    bool recognized = false;
    bool userOverride = false;
    bool cacheHit = false;
    std::uint64_t metadataBytesRead = 0;
    std::string diagnostic;
};

[[nodiscard]] ModelInspection inspectModel(const std::filesystem::path &path);
[[nodiscard]] std::string modelInspectionJson(const ModelInspection &inspection);

// Persists the common Society metadata contract, preserving unrelated fields.
// Supported values: image, video, audio, language; multiple values are allowed.
// An empty list restores automatic inspection, including over embedded metadata.
// Never moves or rewrites the model. Throws on invalid input or a concurrent edit.
void setModelModalities(const std::filesystem::path &path,
    const std::vector<std::string> &modalities);

}
