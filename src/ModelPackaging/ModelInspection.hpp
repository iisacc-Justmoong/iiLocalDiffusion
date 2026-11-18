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

// A bounded prefix probe only; full container and inference validation remain
// the responsibility of the selected reader and backend. Returns
// "safetensors", "safetensors-index", "gguf", "pytorch", "pytorch-index", or "unknown". Index
// detection reads a bounded index; a renamed index also probes each referenced
// shard's container signature. PyTorch is a bounded signature/ZIP-directory
// candidate only; pickle and tensor contents are not executed or validated.
[[nodiscard]] std::string detectModelContainer(const std::filesystem::path &path) noexcept;
[[nodiscard]] ModelInspection inspectModel(const std::filesystem::path &path);
[[nodiscard]] std::string modelInspectionJson(const ModelInspection &inspection);

// Persists the common Society metadata contract, preserving unrelated fields.
// Supported values: image, video, audio, language; multiple values are allowed.
// An empty list restores automatic inspection, including over embedded metadata.
// Never moves or rewrites the model. Throws on invalid input or a concurrent edit.
void setModelModalities(const std::filesystem::path &path,
    const std::vector<std::string> &modalities);

}
