#pragma once

#include <cstdint>
#include <filesystem>

namespace iild::detail {
// Identify a possible PyTorch file without evaluating pickle or reading tensor
// payloads. The generation worker validates the complete checkpoint later.
[[nodiscard]] bool isPyTorchCandidate(const std::filesystem::path &path, std::uint64_t size);
}
