#pragma once
#include "Export.hpp"
#include "NativeExecutionControl.hpp"
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace iiLocalDiffusion {
struct NativeGenerationRequest {
    // A checkpoint file or an iild-unified-model-v1 package directory.
    std::filesystem::path modelPath;
    std::string prompt;
    // Final output size. Base inference uses half of each axis before Hires fix.
    int width = 512;
    int height = 512;
    int steps = 20;
    std::int64_t seed = 0;
    int timeoutMilliseconds = 900000;
    // Empty preserves the source precision. Otherwise prepare/reuse a Q8_0
    // GGUF derivative here, preserving VAE F16 and the original checkpoint.
    std::filesystem::path q8CacheDirectory;
};
enum class NativeGenerationStage { Waiting, Loading, Encoding, Denoising, Decoding, Preparing, Computing };
struct NativeGenerationProgress {
    NativeGenerationStage stage;
    int step = 0;
    int total = 0;
};
// Computing reports cumulative completed CPU graph batches (total is unknown,
// zero). It supplements the current phase without replacing denoising steps.
using NativeProgressCallback = std::function<void(const NativeGenerationProgress &)>;
// Owned RGB projection of the actual denoised latent, without a VAE decode.
// Sequence spans the base and Hires passes; step/total identify the current pass.
struct NativeGenerationPreview {
    std::vector<std::uint8_t> rgb;
    int width = 0;
    int height = 0;
    int sequence = 0;
    int step = 0;
    int total = 0;
};
using NativePreviewCallback = std::function<void(const NativeGenerationPreview &)>;
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
struct NativeLoRA {
    std::filesystem::path path;
    float strength = 1.0f;
};
// Separate options preserve the ABI of existing request/result structures.
// An empty LoRA list selects the manifest fallback for the loaded model family.
// Custom LoRAs replace it; each adapter must match its actual base architecture.
// Custom negative text is combined with the bundled compatible learned tokens.
struct NativeGenerationOptions {
    std::string negativePrompt;
    std::vector<NativeLoRA> loras;
    std::filesystem::path resourceDirectory;
    bool defaultModifiers = true;
};
enum class NativeComputeBackend { Automatic, Cpu };
// Explicit CPU placement supports OS background tasks without GPU access.
// Existing request/options layouts and entry points retain their ABI.
IILD_EXPORT NativeGenerationResult generateNativeImageWithBackend(const NativeGenerationRequest &request,
    NativeComputeBackend backend, const std::atomic_bool &cancelled,
    const NativeProgressCallback &progress = {},
    const std::shared_ptr<NativeExecutionControl> &control = {});
IILD_EXPORT std::filesystem::path nativeGenerationResourceDirectory();
// In-process inference. Optionally cache a Q8 derivative on disk; retain one
// validated model context and its mappings in memory. No downloads/network or
// child processes. Q8 can change pixels even with the same seed.
// Call on a worker thread. Cancellation/deadlines are checked while waiting,
// between weight tensors and between compute segments. GPU calls must return.
// Output dimensions match the requested 8-pixel grid. Half-size base inference
// is followed by Lanczos upscaling and diffusion refinement (strength 0.35).
// The model aligns its base canvas; the final canvas is rounded up to 64 pixels
// and extra borders are center-cropped after refinement.
IILD_EXPORT bool nativeDiffusionAvailable() noexcept;
// Nonblocking: release idle residency now, or after the active GPU call returns.
// Call on memory pressure, actual background entry, or application teardown.
IILD_EXPORT void releaseNativeDiffusionCache() noexcept;
IILD_EXPORT NativeGenerationResult generateNativeImage(const NativeGenerationRequest &request,
    const std::atomic_bool &cancelled, const std::function<void(int, int)> &progress = {});
IILD_EXPORT NativeGenerationResult generateNativeImageWithProgress(const NativeGenerationRequest &request,
    const std::atomic_bool &cancelled, const NativeProgressCallback &progress);
// The execution control may pause/resume a live request without discarding its
// context or latent state. Paused time is excluded from the inference deadline.
// Existing request/result layouts and entry points remain ABI-compatible.
IILD_EXPORT NativeGenerationResult generateNativeImageWithExecutionControl(const NativeGenerationRequest &request,
    const std::atomic_bool &cancelled, const NativeProgressCallback &progress,
    const std::shared_ptr<NativeExecutionControl> &control);
IILD_EXPORT NativeGenerationResult generateNativeImageWithOptions(const NativeGenerationRequest &request,
    const NativeGenerationOptions &options, const std::atomic_bool &cancelled,
    const NativeProgressCallback &progress = {},
    const std::shared_ptr<NativeExecutionControl> &control = {});
// A storage owner can supply shared resources while retaining explicit CPU
// placement for an OS background task. No process-wide resource override.
IILD_EXPORT NativeGenerationResult generateNativeImageWithOptions(const NativeGenerationRequest &request,
    const NativeGenerationOptions &options, NativeComputeBackend backend, const std::atomic_bool &cancelled,
    const NativeProgressCallback &progress = {},
    const std::shared_ptr<NativeExecutionControl> &control = {});
// Validate and retain the same native context without encoding or sampling.
// Lazy weight placement still occurs at first use. No image is produced.
IILD_EXPORT NativeGenerationResult prepareNativeImageModel(const NativeGenerationRequest &request,
    const NativeGenerationOptions &options, const std::atomic_bool &cancelled,
    const NativeProgressCallback &progress = {});
// Opt-in preview entry point preserves all existing request/result layouts.
IILD_EXPORT NativeGenerationResult generateNativeImageWithPreview(const NativeGenerationRequest &request,
    const NativeGenerationOptions &options, NativeComputeBackend backend, const std::atomic_bool &cancelled,
    const NativeProgressCallback &progress, const NativePreviewCallback &preview,
    const std::shared_ptr<NativeExecutionControl> &control = {});
}
