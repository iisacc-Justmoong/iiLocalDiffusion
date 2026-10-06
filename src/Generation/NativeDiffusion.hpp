#pragma once
#include "Export.hpp"
#include "NativeExecutionControl.hpp"
#include "ImageParameters.hpp"
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <limits>
#include <string>
#include <vector>

namespace iiLocalDiffusion {
// Metadata-only defaults for a quick request. Advanced callers retain their
// explicitly submitted values. Bare Krea2 checkpoints use the Raw quality preset.
IILD_EXPORT ImageParameters nativeImageParameterDefaults(const std::filesystem::path &modelPath);
struct NativeGenerationRequest {
    // A checkpoint file, packaged .iildmodel file, or legacy unified directory.
    std::filesystem::path modelPath;
    std::string prompt;
    // Final output size. Legacy entry points use half size followed by Hires;
    // the component-aware entry point makes Hires an explicit option.
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
enum class NativePrediction { Automatic, Epsilon, VPrediction };
enum class NativeSampler { Automatic, Euler, Heun };
// Separate control object preserves all existing public structure layouts.
// customSigmas are native engine sigmas: steps + 1 values including terminal zero.
struct NativeSamplingControls {
    NativeSampler sampler = NativeSampler::Automatic;
    float flowShift = std::numeric_limits<float>::infinity();
    std::vector<float> customSigmas;
};
// Component-aware entry point; existing request/options and their ABI stay intact.
// Empty paths select embedded weights. All supplied paths must be canonical files.
struct NativeModelComponents {
    std::filesystem::path clipL, clipG, t5xxl, llm, vae;
    float guidanceScale = 1.0f;
    float distilledGuidance = 3.5f;
    bool hires = false;
    NativePrediction prediction = NativePrediction::Automatic;
};
// Advanced generation is opt-in; legacy request/options layouts and symbols stay intact.
// Names use the ImageParameters catalog. The engine resolves "auto" per model.
struct NativeReferenceImage {
    int width = 0, height = 0;
    std::vector<std::uint8_t> rgb;
};
struct NativeAdvancedControls {
    std::string sampler = "auto", scheduler = "auto";
    int clipSkip = 0; // Model default, not an unconditional first-layer skip.
    float eta = 0;
    bool seamlessTiling = false;
    bool hires = false;
    float denoiseStrength = 0.25f;
    std::string upscaler = "lanczos";
    std::vector<NativeReferenceImage> references;
    float imageStrength = 0.65f; // Initial-reference denoising amount, [0,1].
    bool promptWeighting = true;
    struct Embedding { std::string token; std::filesystem::path path; };
    std::vector<Embedding> embeddings;
    struct ControlNet {
        std::filesystem::path model;
        NativeReferenceImage image;
        std::string process = "Canny"; // Canny, DWPose, or unchanged RGB Tile hint.
        float weight = 1.0f;
        NativeReferenceImage mask; // Empty disables regional coverage; RGB luminance in [0,255].
        std::filesystem::path poseDetector, poseModel; // Inline ONNX models, required only for Pose.
    };
    // Ordered applied controls; each model/image/weight/mask remains independent.
    std::vector<ControlNet> controls;
    bool freeU = false; // Model-profile FreeU on supported SD UNet decoders.
    std::filesystem::path upscalerModel; // Required for active 4x-ultra Hires.
    bool detailer = false;
    std::filesystem::path detailerModel; // Converted YOLOv8 detection weights.
    bool refiner = false;
    float refinerSwitch = 0.8f; // Fraction of final-pass steps performed by Base.
    std::filesystem::path refinerModel; // SDXL Refiner checkpoint, including bigG.
    struct IPAdapter {
        std::filesystem::path model, vision;
        NativeReferenceImage image; // Original reference RGB, never a detector's processed hint.
        float weight = 1.0f;
        NativeReferenceImage mask;
    };
    std::vector<IPAdapter> ipAdapters; // Ordered independent resident image-conditioning slots.
};
IILD_EXPORT NativeGenerationResult generateNativeAdvancedImage(const NativeGenerationRequest &request,
    const NativeGenerationOptions &options, const NativeModelComponents &components,
    const NativeAdvancedControls &advanced, NativeComputeBackend backend,
    const std::atomic_bool &cancelled, const NativeProgressCallback &progress = {},
    const NativePreviewCallback &preview = {}, const std::shared_ptr<NativeExecutionControl> &control = {});
IILD_EXPORT NativeGenerationResult generateNativeImageWithComponents(const NativeGenerationRequest &request,
    const NativeGenerationOptions &options, const NativeModelComponents &components,
    NativeComputeBackend backend, const std::atomic_bool &cancelled, bool prepareOnly = false,
    const NativeProgressCallback &progress = {}, const NativePreviewCallback &preview = {},
    const std::shared_ptr<NativeExecutionControl> &control = {});
IILD_EXPORT NativeGenerationResult generateNativeImageWithSampling(const NativeGenerationRequest &request,
    const NativeGenerationOptions &options, const NativeModelComponents &components,
    const NativeSamplingControls &sampling, NativeComputeBackend backend,
    const std::atomic_bool &cancelled, bool prepareOnly = false,
    const NativeProgressCallback &progress = {}, const NativePreviewCallback &preview = {},
    const std::shared_ptr<NativeExecutionControl> &control = {});
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
// Explicit user release or application teardown only. Never call for idle,
// background entry or memory pressure; anonymous sources belong to the runtime.
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
// Society host generation stages every tensor source in anonymous process memory
// before inference. The OS may page these anonymous pages to its managed swap.
// This is separate from Q8 disk-cache generation and leaves existing APIs intact.
// Component-aware desktop path: preparation and generation share the same
// resident context, retaining exact family/sampler/sigma/preview contracts.
IILD_EXPORT NativeGenerationResult generateNativeImageWithResidentWeights(const NativeGenerationRequest &request,
    const NativeGenerationOptions &options, const NativeModelComponents &components,
    const NativeSamplingControls &sampling, NativeComputeBackend backend,
    const std::atomic_bool &cancelled, bool prepareOnly = false,
    const NativeProgressCallback &progress = {}, const NativePreviewCallback &preview = {},
    const std::shared_ptr<NativeExecutionControl> &control = {});
IILD_EXPORT NativeGenerationResult generateNativeImageWithResidentWeights(const NativeGenerationRequest &request,
    const NativeGenerationOptions &options, const std::atomic_bool &cancelled,
    const NativeProgressCallback &progress = {},
    const std::shared_ptr<NativeExecutionControl> &control = {});
IILD_EXPORT NativeGenerationResult generateNativeImageWithResidentWeights(const NativeGenerationRequest &request,
    const NativeGenerationOptions &options, NativeComputeBackend backend, const std::atomic_bool &cancelled,
    const NativeProgressCallback &progress, const NativePreviewCallback &preview,
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
// Queue barrier: return only after all source files occupy anonymous memory.
IILD_EXPORT NativeGenerationResult prepareNativeImageModelWithResidentWeights(const NativeGenerationRequest &request,
    const NativeGenerationOptions &options, const std::atomic_bool &cancelled,
    const NativeProgressCallback &progress = {});
// Opt-in preview entry point preserves all existing request/result layouts.
IILD_EXPORT NativeGenerationResult generateNativeImageWithPreview(const NativeGenerationRequest &request,
    const NativeGenerationOptions &options, NativeComputeBackend backend, const std::atomic_bool &cancelled,
    const NativeProgressCallback &progress, const NativePreviewCallback &preview,
    const std::shared_ptr<NativeExecutionControl> &control = {});
}
