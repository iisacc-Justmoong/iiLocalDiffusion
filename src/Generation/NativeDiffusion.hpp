#pragma once
#include "Export.hpp"
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace iiLocalDiffusion {
struct NativeGenerationRequest {
    std::filesystem::path modelPath;
    std::string prompt;
    int width = 512;
    int height = 512;
    int steps = 20;
    std::int64_t seed = 0;
    int timeoutMilliseconds = 900000;
    // Empty preserves the source precision. Otherwise prepare/reuse a Q8_0
    // GGUF derivative here, preserving VAE F16 and the original checkpoint.
    std::filesystem::path q8CacheDirectory;
};
enum class NativeGenerationStage { Waiting, Loading, Encoding, Denoising, Decoding, Preparing };
struct NativeGenerationProgress {
    NativeGenerationStage stage;
    int step = 0;
    int total = 0;
};
using NativeProgressCallback = std::function<void(const NativeGenerationProgress &)>;
struct NativeGenerationResult {
    std::vector<std::uint8_t> rgb;
    int width = 0;
    int height = 0;
    bool cancelled = false;
    std::string error;
    bool modelCacheHit = false;
    std::uint64_t memoryBudgetBytes = 0;
    int threads = 0;
    double modelLoadMilliseconds = 0;
    double generationMilliseconds = 0;
    bool q8CacheUsed = false;
    bool diskCacheHit = false;
    std::uint64_t modelBytes = 0;
    double preparationMilliseconds = 0;
};
// In-process inference. Optionally cache a Q8 derivative on disk; retain one
// validated model context and its mappings in memory. No downloads/network or
// child processes. Q8 can change pixels even with the same seed.
// Call on a worker thread. Cancellation/deadlines are checked while waiting,
// between weight tensors and between compute segments. GPU calls must return.
IILD_EXPORT bool nativeDiffusionAvailable() noexcept;
// Nonblocking: release idle residency now, or after the active GPU call returns.
// Call on memory pressure, actual background entry, or application teardown.
IILD_EXPORT void releaseNativeDiffusionCache() noexcept;
IILD_EXPORT NativeGenerationResult generateNativeImage(const NativeGenerationRequest &request,
    const std::atomic_bool &cancelled, const std::function<void(int, int)> &progress = {});
IILD_EXPORT NativeGenerationResult generateNativeImageWithProgress(const NativeGenerationRequest &request,
    const std::atomic_bool &cancelled, const NativeProgressCallback &progress);
}
