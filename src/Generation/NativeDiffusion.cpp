#include "NativeDiffusion.hpp"
#include "NativeCachePolicy.hpp"
#include "NativeDiskCache.hpp"
#include "NativeVaePolicy.hpp"
#include "GenerationDefaults.hpp"
#include "UnifiedModel.hpp"
#include <cmath>
#include <algorithm>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <thread>
#if defined(__APPLE__)
#include <pthread.h>
#include <pthread/qos.h>
#endif
#if IILD_HAS_NATIVE_DIFFUSION
#include <stable-diffusion.h>
#endif

namespace iiLocalDiffusion {
#if IILD_HAS_NATIVE_DIFFUSION
namespace {
struct Elapsed {
    double &milliseconds;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    ~Elapsed() { milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(); }
};
struct EngineCache {
    std::timed_mutex mutex;
    std::atomic_uint64_t releaseEpoch{0};
    std::unique_ptr<sd_ctx_t, decltype(&free_sd_ctx)> context{nullptr, free_sd_ctx};
    std::string identity;
    std::string inspectedModelIdentity;
    sd_model_vae_info_t vaeInfo{};
    std::string missingVaeFamily;
    std::string validatedVaeIdentity;
    std::uint64_t budget = 0;
};
EngineCache &engineCache() { static EngineCache cache; return cache; }
thread_local bool ownsEngine = false;
}
#endif
bool nativeDiffusionAvailable() noexcept { return IILD_HAS_NATIVE_DIFFUSION; }
std::filesystem::path nativeGenerationResourceDirectory() { return native_detail::generationResourceDirectory(); }
void releaseNativeDiffusionCache() noexcept {
#if IILD_HAS_NATIVE_DIFFUSION
    auto &cache = engineCache();
    ++cache.releaseEpoch;
    // May be called from a progress callback; never recurse into its lock.
    if (ownsEngine) return;
    std::unique_lock lock(cache.mutex, std::try_to_lock);
    if (lock.owns_lock()) {
        if (cache.context && std::getenv("IILD_NATIVE_DIAGNOSTICS"))
            std::fputs("iiLocalDiffusion cache: explicit idle release\n", stderr);
        cache.context.reset();
    }
#endif
}

NativeGenerationResult generateNativeImage(const NativeGenerationRequest &request,
    const std::atomic_bool &cancelled, const std::function<void(int, int)> &progress)
{
    return generateNativeImageWithProgress(request, cancelled, [&progress](const NativeGenerationProgress &event) {
        if (progress && event.stage == NativeGenerationStage::Denoising) progress(event.step, event.total);
    });
}

NativeGenerationResult generateNativeImageWithProgress(const NativeGenerationRequest &request,
    const std::atomic_bool &cancelled, const NativeProgressCallback &progress)
{
    return generateNativeImageWithExecutionControl(request, cancelled, progress, {});
}

NativeGenerationResult generateNativeImageWithExecutionControl(const NativeGenerationRequest &request,
    const std::atomic_bool &cancelled, const NativeProgressCallback &progress,
    const std::shared_ptr<NativeExecutionControl> &control)
{
    return generateNativeImageWithOptions(request, {}, cancelled, progress, control);
}

static NativeGenerationResult nativeImage(const NativeGenerationRequest &request,
    const NativeGenerationOptions &options, const std::atomic_bool &cancelled,
    const NativeProgressCallback &progress, const std::shared_ptr<NativeExecutionControl> &control,
    bool prepareOnly, NativeComputeBackend backend = NativeComputeBackend::Automatic,
    const NativePreviewCallback &preview = {}, const NativeGenerationResult *initial = nullptr, float strength = 1.0f)
{
    NativeGenerationResult result;
    const auto started = std::chrono::steady_clock::now();
    const auto pausedAtStart = control ? control->pausedDuration() : NativeExecutionControl::Clock::duration::zero();
    const auto timedOut = [&] {
        const auto paused = control ? control->pausedDuration() - pausedAtStart : NativeExecutionControl::Clock::duration::zero();
        return std::chrono::steady_clock::now() - started - paused >= std::chrono::milliseconds(request.timeoutMilliseconds);
    };
    const auto stopped = [&] {
        return cancelled || (control && !control->waitUntilRunnable(cancelled)) || timedOut();
    };
    try {
        if (cancelled) { result.cancelled = true; return result; }
        if (!request.modelPath.is_absolute() || !std::filesystem::is_regular_file(request.modelPath)
            || std::filesystem::canonical(request.modelPath) != request.modelPath)
            throw std::runtime_error("Choose an available local model file.");
        if (request.prompt.empty() || request.prompt.size() > 128000 || request.prompt.find('\0') != std::string::npos
            || request.width < 64 || request.height < 64 || request.width > 2048 || request.height > 2048
            || request.width % 8 || request.height % 8 || request.steps < 1 || request.steps > 1000
            || request.timeoutMilliseconds < 1)
            throw std::runtime_error("Invalid native image generation parameters.");
        if (options.negativePrompt.size() > 128000 || options.negativePrompt.find('\0') != std::string::npos)
            throw std::runtime_error("Invalid native negative prompt.");
        for (const auto &lora : options.loras)
            if (!lora.path.is_absolute() || !std::filesystem::is_regular_file(lora.path)
                || std::filesystem::canonical(lora.path) != lora.path || !std::isfinite(lora.strength))
                throw std::runtime_error("Choose an available local LoRA and finite strength.");
#if IILD_HAS_NATIVE_DIFFUSION
        // Upstream callbacks are process-global. Serialize only this backend's
        // calls and clear callback state before any caller-owned data is freed.
        auto &cache = engineCache();
        std::unique_lock lock(cache.mutex, std::defer_lock);
        if (progress) progress({NativeGenerationStage::Waiting});
        while (!lock.try_lock_for(std::chrono::milliseconds(20))) {
            if (control) control->waitUntilRunnable(cancelled);
            if (cancelled) { result.cancelled = true; return result; }
            if (timedOut()) throw std::runtime_error("Native image generation exceeded its time limit.");
        }
        if (control) control->waitUntilRunnable(cancelled);
        if (cancelled) { result.cancelled = true; return result; }
        if (timedOut()) throw std::runtime_error("Native image generation exceeded its time limit.");
        struct EngineAccess {
            EngineAccess() { ownsEngine = true; }
            ~EngineAccess() { ownsEngine = false; }
        } access;
        struct Callbacks {
            const std::atomic_bool &cancelled;
            const NativeProgressCallback &progress;
            const std::function<bool()> stopped;
            const NativePreviewCallback &preview;
            int previewSequence = 0;
            int samplingTotal = 0;
            sd_ctx_t *context = nullptr;
            bool preparing = false;
            unsigned graphNodes = 0;
            int completedGraphBatches = 0;
            std::string error;
            std::mutex logMutex;
            bool stop() const { return stopped(); }
            std::string failure(const char *message) {
                const std::lock_guard lock(logMutex);
                return error.empty() ? message : std::string(message) + " " + error;
            }
            ~Callbacks() {
                sd_set_abort_callback(nullptr, nullptr);
                sd_set_progress_stage_callback(nullptr, nullptr);
                sd_set_backend_eval_callback(nullptr, nullptr);
                sd_set_preview_callback(nullptr, PREVIEW_NONE, 1, false, false, nullptr);
                sd_set_log_callback(nullptr, nullptr);
            }
        } callbacks{cancelled, progress, stopped, preview};
        if (preview && !prepareOnly) {
            sd_set_preview_callback([](int step, int count, sd_image_t *frames, bool noisy, void *opaque) {
                auto &state = *static_cast<Callbacks *>(opaque);
                if (state.stop() || noisy || count != 1 || !frames || step < 1 || step > state.samplingTotal) return;
                const auto &frame = frames[0];
                if (!frame.data || frame.channel != 3 || !frame.width || !frame.height
                    || frame.width > 512 || frame.height > 512) return;
                NativeGenerationPreview value;
                value.width = static_cast<int>(frame.width); value.height = static_cast<int>(frame.height);
                value.sequence = ++state.previewSequence;
                value.step = step; value.total = state.samplingTotal;
                value.rgb.assign(frame.data, frame.data + static_cast<std::size_t>(frame.width) * frame.height * 3);
                state.preview(value);
            }, PREVIEW_PROJ, 1, true, false, &callbacks);
        }
        struct CacheUse {
            EngineCache &cache;
            std::uint64_t epoch;
            bool successful = false;
            ~CacheUse() {
                if (!successful || cache.releaseEpoch != epoch) {
                    if (cache.context && std::getenv("IILD_NATIVE_DIAGNOSTICS"))
                        std::fprintf(stderr, "iiLocalDiffusion cache: %s\n",
                            !successful ? "discarded incomplete generation" : "deferred release requested during generation");
                    cache.context.reset();
                }
            }
        } cacheUse{cache, cache.releaseEpoch.load()};
#if defined(__APPLE__)
        struct WorkerPriority {
            qos_class_t previous = QOS_CLASS_UNSPECIFIED;
            int relative = 0;
            WorkerPriority() {
                pthread_get_qos_class_np(pthread_self(), &previous, &relative);
                pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
            }
            ~WorkerPriority() { pthread_set_qos_class_self_np(previous, relative); }
        } priority;
#endif
        sd_set_abort_callback([](void *opaque) { return static_cast<Callbacks *>(opaque)->stop(); }, &callbacks);
        if (backend == NativeComputeBackend::Cpu) {
            // A CPU denoising step can take longer than the OS stall budget.
            // Observe completed graph batches, never a timer-based heartbeat.
            // The upstream callback also gives pause/cancel a safe CPU boundary.
            sd_set_backend_eval_callback([](ggml_tensor *, bool ask, void *opaque) {
                auto &state = *static_cast<Callbacks *>(opaque);
                if (ask) return ++state.graphNodes % 16 == 0;
                ++state.completedGraphBatches;
                if (state.progress) state.progress({NativeGenerationStage::Computing, state.completedGraphBatches, 0});
                return !state.stop();
            }, &callbacks);
        }
        sd_set_log_callback([](sd_log_level_t level, const char *text, void *opaque) {
            auto &state = *static_cast<Callbacks *>(opaque);
            if (text && std::getenv("IILD_NATIVE_DIAGNOSTICS")) std::fputs(text, stderr);
            if (text && std::string_view(text).find("decoding ") != std::string_view::npos && state.progress)
                state.progress({NativeGenerationStage::Decoding});
            // Warnings (including SDXL's built-in VAE scale) are not failures.
            // Keep real backend errors as context, never replace our diagnosis.
            if (level >= SD_LOG_ERROR && text) {
                const std::lock_guard lock(state.logMutex);
                state.error += text;
                if (state.error.size() > 4000) state.error.erase(0, state.error.size() - 4000);
            }
        }, &callbacks);
        sd_set_progress_stage_callback([](sd_progress_stage_t stage, int step, int total, float, void *opaque) {
            auto &state = *static_cast<Callbacks *>(opaque);
            if (stage == SD_PROGRESS_SAMPLE) state.samplingTotal = total;
            if (state.stop() && state.context) sd_cancel_generation(state.context, SD_CANCEL_ALL);
            if (state.progress) state.progress({state.preparing ? NativeGenerationStage::Preparing
                : stage == SD_PROGRESS_SAMPLE ? NativeGenerationStage::Denoising
                : stage == SD_PROGRESS_DECODE ? NativeGenerationStage::Decoding : NativeGenerationStage::Loading, step, total});
        }, &callbacks);
        result.threads = std::max(1, sd_get_num_physical_cores());
#if defined(__APPLE__)
        // Upstream counts only performance cores. Include the efficiency cores
        // for independent tensor preparation, loading and CPU kernels.
        result.threads = std::max(result.threads, static_cast<int>(std::thread::hardware_concurrency()));
#endif
        const auto sourceIdentity = native_detail::modelIdentity(request.modelPath);
        const auto defaults = options.defaultModifiers
            ? native_detail::loadGenerationDefaults(options.resourceDirectory) : native_detail::GenerationDefaults{};
        std::vector<std::string> embeddingPaths;
        for (const auto &embedding : defaults.embeddings) embeddingPaths.push_back(embedding.path.string());
        std::vector<sd_embedding_t> embeddings;
        for (std::size_t i = 0; i < defaults.embeddings.size(); ++i)
            embeddings.push_back({defaults.embeddings[i].token.c_str(), embeddingPaths[i].c_str()});
        auto effectiveModel = request.modelPath;
        if (!request.q8CacheDirectory.empty()) {
            Elapsed timing{result.preparationMilliseconds};
            callbacks.preparing = true;
            if (progress) progress({NativeGenerationStage::Preparing});
            const auto prepared = native_detail::prepareQ8Cache(request.modelPath, request.q8CacheDirectory,
                sourceIdentity, [&](const auto &output) {
                    // Conversion needs CPU working memory; evict previous GPU
                    // weights before starting its bounded streaming workers.
                    cache.context.reset();
                    return convert_with_components(request.modelPath.string().c_str(), nullptr, nullptr, nullptr,
                        nullptr, nullptr, output.string().c_str(), SD_TYPE_Q8_0,
                        "^(first_stage_model|vae)\\.=f16", false, result.threads);
                }, [&] {
                    if (callbacks.stop()) throw std::runtime_error("Native model preparation interrupted.");
                });
            callbacks.preparing = false;
            effectiveModel = prepared.path;
            result.q8CacheUsed = true;
            result.diskCacheHit = prepared.hit;
            result.modelBytes = prepared.bytes;
        } else result.modelBytes = std::filesystem::file_size(effectiveModel);
        const auto model = effectiveModel.string();
        const auto effectiveIdentity = native_detail::modelIdentity(effectiveModel);
        if (cache.inspectedModelIdentity != effectiveIdentity) {
            sd_model_vae_info_t info{};
            if (!sd_model_inspect_vae(model.c_str(), &info))
                throw std::runtime_error(callbacks.failure("Cannot inspect the local model's VAE components."));
            cache.vaeInfo = info;
            cache.missingVaeFamily = (info.state == SD_VAE_MISSING || info.state == SD_VAE_INCOMPATIBLE)
                ? info.vae_family : "";
            cache.inspectedModelIdentity = effectiveIdentity;
            cache.validatedVaeIdentity.clear();
        }
        // A required decoder remains enabled even when style modifiers are off.
        // Inspect the actual mounted file, including prepared GGUF caches.
        const auto fallbackVae = !cache.missingVaeFamily.empty()
            ? native_detail::loadFallbackVae(options.resourceDirectory, cache.missingVaeFamily) : native_detail::DefaultVae{};
        if (!fallbackVae.path.empty() && cache.validatedVaeIdentity != fallbackVae.identity) {
            if (!sd_model_validate_vae(model.c_str(), fallbackVae.path.string().c_str()))
                throw std::runtime_error("The selected VAE does not match the " + cache.missingVaeFamily
                    + " tensor contract: " + fallbackVae.path.string());
            cache.validatedVaeIdentity = fallbackVae.identity;
            if (std::getenv("IILD_NATIVE_DIAGNOSTICS"))
                std::fprintf(stderr, "iiLocalDiffusion VAE auto-mount-v2: model=%s family=%s source=fallback-validated path=%s\n",
                    cache.vaeInfo.model_family, cache.vaeInfo.vae_family, fallbackVae.path.string().c_str());
        }
        auto modifierIdentity = defaults.identity + fallbackVae.identity;
        for (const auto &lora : options.loras) modifierIdentity += ':' + native_detail::modelIdentity(lora.path);
        const auto identity = sourceIdentity + ':' + effectiveIdentity + ':' + modifierIdentity
            + (backend == NativeComputeBackend::Cpu ? ":cpu" : ":automatic");
        if (cache.identity != identity) {
            if (cache.context && std::getenv("IILD_NATIVE_DIAGNOSTICS"))
                std::fputs("iiLocalDiffusion cache: model identity changed\n", stderr);
            cache.context.reset();
        }
        // Do not rebuild a warm engine for small fluctuations in free memory.
        // Explicit pressure/background release invalidates it independently.
        auto budget = cache.budget;
        if (!cache.context) {
#if defined(__APPLE__)
            const auto limits = native_detail::resourceLimits();
            budget = native_detail::memoryBudget(limits, IILD_NATIVE_CPU_VAE);
            if (std::getenv("IILD_NATIVE_DIAGNOSTICS"))
                std::fprintf(stderr, "iiLocalDiffusion resources: physical=%llu recommended=%llu available=%llu budget=%llu policy=%s\n",
                    static_cast<unsigned long long>(limits.physical), static_cast<unsigned long long>(limits.recommended),
                    static_cast<unsigned long long>(limits.available), static_cast<unsigned long long>(budget),
                    IILD_NATIVE_CPU_VAE ? "ios-cpu-vae-headroom" : "automatic");
#else
            budget = native_detail::memoryBudget({});
#endif
            if (std::getenv("IILD_NATIVE_DIAGNOSTICS")) {
                if (const auto ceiling = std::getenv("IILD_NATIVE_MAX_MEMORY_MIB"))
                    budget = native_detail::diagnosticMemoryCeiling(budget, ceiling);
            }
            if (budget < 512ull * 1024 * 1024)
                throw std::runtime_error("There is not enough available memory to load this model. Try again after closing other apps.");
        }
        result.modelCacheHit = !!cache.context;
        result.memoryBudgetBytes = budget;
        const auto loadStarted = std::chrono::steady_clock::now();
        sd_ctx_params_t contextParameters;
        sd_ctx_params_init(&contextParameters);
        contextParameters.model_path = model.c_str();
        const auto vaePath = fallbackVae.path.string();
        if (!vaePath.empty()) contextParameters.vae_path = vaePath.c_str();
        contextParameters.embeddings = embeddings.data();
        contextParameters.embedding_count = static_cast<uint32_t>(embeddings.size());
        contextParameters.n_threads = result.threads;
        contextParameters.enable_mmap = true;
#if IILD_NATIVE_CPU_VAE
        // iOS Metal VAE decoding can lose command buffers during GPU recovery.
        // Keep tiled decoding and its weights on CPU; other modules still use
        // automatic compute placement (Metal on Apple). An explicit backend
        // disables upstream auto-fit, so keep their weights reloadable from
        // the mapped file instead of accumulating non-evictable GPU residency.
        contextParameters.backend = "vae=cpu";
        contextParameters.params_backend = "te=disk,diffusion=disk,vae=cpu";
#endif
        if (backend == NativeComputeBackend::Cpu) {
            contextParameters.backend = "cpu";
            contextParameters.params_backend = "cpu";
        }
        // Apply adapters at runtime and preserve read-only checkpoint mappings.
        contextParameters.lora_apply_mode = LORA_APPLY_AT_RUNTIME;
        contextParameters.flash_attn = true;
        contextParameters.diffusion_flash_attn = true;
        contextParameters.eager_load = false;
        contextParameters.disable_prefetch = false;
        const auto budgetGiB = std::to_string(static_cast<double>(budget) / (1024.0 * 1024 * 1024));
        contextParameters.max_vram = budgetGiB.c_str();
        if (progress) progress({NativeGenerationStage::Loading});
        if (!cache.context) {
            cache.context.reset(new_sd_ctx(&contextParameters));
            cache.identity = identity;
            cache.budget = budget;
        }
        auto *context = cache.context.get();
        result.modelLoadMilliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - loadStarted).count();
        if (cancelled) { result.cancelled = true; return result; }
        if (timedOut()) throw std::runtime_error("Native image generation exceeded its time limit.");
        if (!context || !sd_ctx_supports_image_generation(context))
            throw std::runtime_error(callbacks.failure("This model could not be loaded by the native image engine."));
        if (prepareOnly) {
            if (native_detail::modelIdentity(request.modelPath) != sourceIdentity)
                throw std::runtime_error("The local model changed during preparation. Try again.");
            static const bool preparationCleanup = [] {
                std::atexit([] { releaseNativeDiffusionCache(); });
                return true;
            }();
            (void)preparationCleanup;
            cacheUse.successful = true;
            return result;
        }
        callbacks.context = context;
        sd_img_gen_params_t parameters;
        sd_img_gen_params_init(&parameters);
        parameters.prompt = request.prompt.c_str();
        const std::string family = sd_get_model_family(context);
        auto selectedLoras = options.loras;
        if (selectedLoras.empty() && options.defaultModifiers) {
            const auto canonicalFamily = native_detail::canonicalLoraFamily(family);
            for (const auto &fallback : defaults.loras)
                if (std::find(fallback.families.begin(), fallback.families.end(), canonicalFamily) != fallback.families.end())
                    selectedLoras.push_back({fallback.path, fallback.scale});
        }
        std::vector<std::string> loraPaths;
        for (const auto &lora : selectedLoras) loraPaths.push_back(lora.path.string());
        std::vector<sd_lora_t> loras;
        for (std::size_t i = 0; i < selectedLoras.size(); ++i)
            loras.push_back({false, selectedLoras[i].strength, loraPaths[i].c_str()});
        parameters.loras = loras.data();
        parameters.lora_count = static_cast<uint32_t>(loras.size());
        std::vector<std::string> negativeTokens;
        for (const auto &embedding : defaults.embeddings)
            if (family == "sdxl-base" || (family == "sd15" && embedding.sd15))
                negativeTokens.push_back(embedding.token);
        const auto negativePrompt = native_detail::appendNegativeTokens(options.negativePrompt, negativeTokens);
        parameters.negative_prompt = negativePrompt.c_str();
        if (std::getenv("IILD_NATIVE_DIAGNOSTICS"))
            std::fprintf(stderr, "iiLocalDiffusion defaults: family=%s negative_embeddings=%zu loras=%zu fallback=%d\n",
                family.c_str(), negativeTokens.size(), loras.size(),
                options.loras.empty() && !selectedLoras.empty());
        // Public dimensions describe the final image. The engine aligns the
        // half-size base to its model grid, decodes it, resizes with Lanczos,
        // VAE-encodes it, then runs a real second diffusion pass.
        parameters.width = request.width / 2;
        parameters.height = request.height / 2;
        parameters.hires.enabled = true;
        parameters.hires.upscaler = SD_HIRES_UPSCALER_LANCZOS;
        parameters.hires.scale = 2.0f;
        parameters.hires.target_width = (request.width + 63) / 64 * 64;
        parameters.hires.target_height = (request.height + 63) / 64 * 64;
        parameters.hires.denoising_strength = 0.35f;
        // Native hires.steps counts active refinement steps, unlike the full
        // Diffusers schedule. Never allow a small request to skip denoising.
        parameters.hires.steps = std::max(1, int(static_cast<float>(request.steps) * parameters.hires.denoising_strength));
        if (std::getenv("IILD_NATIVE_DIAGNOSTICS"))
            std::fprintf(stderr, "iiLocalDiffusion Hires: requested=%dx%d base=%dx%d final-canvas=%dx%d lanczos strength=0.35 steps=%d\n",
                request.width, request.height, parameters.width, parameters.height,
                parameters.hires.target_width, parameters.hires.target_height, parameters.hires.steps);
        parameters.seed = request.seed;
        parameters.batch_count = 1;
        if (initial) {
            if (initial->rgb.size() != std::size_t(initial->width) * initial->height * 3
                || initial->width != request.width || initial->height != request.height
                || !std::isfinite(strength) || strength <= 0 || strength > 1)
                throw std::runtime_error("Invalid unified model image bridge.");
            parameters.init_image = {static_cast<uint32_t>(initial->width), static_cast<uint32_t>(initial->height),
                                     3, const_cast<uint8_t *>(initial->rgb.data())};
            parameters.strength = strength;
        }
        parameters.sample_params.sample_steps = request.steps;
        parameters.sample_params.sample_method = sd_get_default_sample_method(context);
        parameters.sample_params.scheduler = sd_get_default_scheduler(context, parameters.sample_params.sample_method);
        const auto vaePolicy = native_detail::vaeDecodePolicy(family,
            parameters.hires.target_width, parameters.hires.target_height, budget);
        parameters.vae_tiling_params.enabled = vaePolicy.tiled;
        parameters.vae_tiling_params.tile_size_x = parameters.vae_tiling_params.tile_size_y = vaePolicy.tile;
        parameters.vae_tiling_params.target_overlap = vaePolicy.overlap;
        if (std::getenv("IILD_NATIVE_DIAGNOSTICS"))
            std::fprintf(stderr, "iiLocalDiffusion VAE policy: adaptive-sdxl-v1 family=%s mount=%s tile=%d tiled=%d overlap=%.2f\n",
                family.c_str(), vaePath.empty() ? "embedded" : "family-fallback", vaePolicy.tile,
                vaePolicy.tiled, vaePolicy.overlap);
        struct Images {
            sd_image_t *data = nullptr;
            int count = 0;
            ~Images() { if (data) free_sd_images(data, count); }
        } images;
        if (progress) progress({NativeGenerationStage::Encoding});
        bool ok = false;
        {
            Elapsed timing{result.generationMilliseconds};
            ok = generate_image(context, &parameters, &images.data, &images.count);
        }
        if (cancelled) { result.cancelled = true; return result; }
        if (timedOut()) throw std::runtime_error("Native image generation exceeded its time limit.");
        if (!ok || !images.data || images.count != 1 || !images.data[0].data || images.data[0].channel != 3)
            throw std::runtime_error(callbacks.failure("The native engine did not return a complete RGB image."));
        const auto &decoded = images.data[0];
        if (decoded.width != static_cast<unsigned>(parameters.hires.target_width)
            || decoded.height != static_cast<unsigned>(parameters.hires.target_height))
            throw std::runtime_error("The native engine returned an unexpected image size: expected "
                + std::to_string(parameters.hires.target_width) + "x" + std::to_string(parameters.hires.target_height)
                + ", received " + std::to_string(decoded.width) + "x" + std::to_string(decoded.height) + ".");
        result.width = request.width;
        result.height = request.height;
        const auto outputWidth = static_cast<std::size_t>(result.width);
        const auto outputHeight = static_cast<std::size_t>(result.height);
        const auto rowBytes = outputWidth * 3;
        const auto sourceStride = static_cast<std::size_t>(decoded.width) * 3;
        const auto left = (decoded.width - outputWidth) / 2;
        const auto top = (decoded.height - outputHeight) / 2;
        result.rgb.resize(rowBytes * outputHeight);
        for (std::size_t y = 0; y < outputHeight; ++y)
            std::copy_n(decoded.data + (top + y) * sourceStride + left * 3,
                        rowBytes, result.rgb.data() + y * rowBytes);
        // A sync client may atomically replace the source during generation.
        if (native_detail::modelIdentity(request.modelPath) != sourceIdentity
            || native_detail::modelIdentity(effectiveModel) != effectiveIdentity)
            throw std::runtime_error("The local model changed during generation. Try again.");
        auto finalModifierIdentity = options.defaultModifiers
            ? native_detail::loadGenerationDefaults(options.resourceDirectory).identity : std::string{};
        if (!cache.missingVaeFamily.empty())
            finalModifierIdentity += native_detail::loadFallbackVae(options.resourceDirectory, cache.missingVaeFamily).identity;
        for (const auto &lora : options.loras) finalModifierIdentity += ':' + native_detail::modelIdentity(lora.path);
        if (finalModifierIdentity != modifierIdentity)
            throw std::runtime_error("Generation defaults or LoRA changed during generation. Try again.");
        // Lazy backends register process-exit destructors during inference,
        // after new_sd_ctx. Register our cleanup only after that first complete
        // inference so retained GPU weights are destroyed before the registries.
        static const bool cleanupRegistered = [] {
            std::atexit([] { releaseNativeDiffusionCache(); });
            return true;
        }();
        (void)cleanupRegistered;
        cacheUse.successful = true;
#else
        (void)progress;
        throw std::runtime_error("This iiLocalDiffusion build does not include native image inference.");
#endif
    } catch (const std::exception &error) {
        result.rgb.clear();
        if (cancelled) result.cancelled = true;
        else if (request.timeoutMilliseconds > 0 && timedOut())
            result.error = "Native image generation exceeded its time limit.";
        else result.error = error.what();
    }
    return result;
}

static NativeGenerationResult dispatchNativeImage(const NativeGenerationRequest &request,
    const NativeGenerationOptions &options, const std::atomic_bool &cancelled,
    const NativeProgressCallback &progress, const std::shared_ptr<NativeExecutionControl> &control,
    bool prepareOnly, NativeComputeBackend backend = NativeComputeBackend::Automatic,
    const NativePreviewCallback &preview = {})
{
    std::error_code error;
    if (!std::filesystem::is_directory(request.modelPath, error))
        return nativeImage(request, options, cancelled, progress, control, prepareOnly, backend, preview);
    NativeGenerationResult result;
    try {
        if (cancelled) { result.cancelled = true; return result; }
        if (!options.loras.empty()) throw std::runtime_error("Add LoRAs to their compatible member when building the unified model.");
        const auto model = native_detail::loadUnifiedModel(request.modelPath);
        const auto started = std::chrono::steady_clock::now();
        const auto pausedStart = control ? control->pausedDuration() : NativeExecutionControl::Clock::duration::zero();
        double loadTime = 0, generationTime = 0, preparationTime = 0;
        int previewSequence = 0;
        for (std::size_t i = 0; i < model.stages.size(); ++i) {
            if (cancelled || (control && !control->waitUntilRunnable(cancelled))) {
                result.rgb.clear(); result.cancelled = true; return result;
            }
            const auto &stage = model.stages[i];
            if (i && stage.strength == 0) continue;
            auto member = request;
            member.modelPath = stage.model;
            const auto paused = control ? control->pausedDuration() - pausedStart : NativeExecutionControl::Clock::duration::zero();
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started - paused).count();
            if (elapsed >= request.timeoutMilliseconds) throw std::runtime_error("Unified image generation exceeded its time limit.");
            member.timeoutMilliseconds -= static_cast<int>(elapsed);
            model.verify();
            auto next = nativeImage(member, options, cancelled, progress, control, prepareOnly, backend,
                preview ? NativePreviewCallback([&](const NativeGenerationPreview &frame) {
                    auto unified = frame; unified.sequence = ++previewSequence; preview(unified);
                }) : NativePreviewCallback{}, i ? &result : nullptr, stage.strength);
            if (next.cancelled || !next.error.empty()) {
                if (!next.error.empty()) next.error = "Unified stage " + std::to_string(i + 1) + ": " + next.error;
                return next;
            }
            loadTime += next.modelLoadMilliseconds;
            generationTime += next.generationMilliseconds;
            preparationTime += next.preparationMilliseconds;
            result = std::move(next);
            // Foreground preparation retains the first context for the real run.
            if (prepareOnly) break;
        }
        model.verify();
        result.modelLoadMilliseconds = loadTime;
        result.generationMilliseconds = generationTime;
        result.preparationMilliseconds = preparationTime;
    } catch (const std::exception &failure) {
        result.rgb.clear(); result.error = failure.what();
    }
    return result;
}

NativeGenerationResult generateNativeImageWithOptions(const NativeGenerationRequest &request,
    const NativeGenerationOptions &options, const std::atomic_bool &cancelled,
    const NativeProgressCallback &progress, const std::shared_ptr<NativeExecutionControl> &control)
{
    return dispatchNativeImage(request, options, cancelled, progress, control, false);
}

NativeGenerationResult generateNativeImageWithBackend(const NativeGenerationRequest &request,
    NativeComputeBackend backend, const std::atomic_bool &cancelled,
    const NativeProgressCallback &progress, const std::shared_ptr<NativeExecutionControl> &control)
{
    return dispatchNativeImage(request, {}, cancelled, progress, control, false, backend);
}

NativeGenerationResult generateNativeImageWithOptions(const NativeGenerationRequest &request,
    const NativeGenerationOptions &options, NativeComputeBackend backend, const std::atomic_bool &cancelled,
    const NativeProgressCallback &progress, const std::shared_ptr<NativeExecutionControl> &control)
{
    return dispatchNativeImage(request, options, cancelled, progress, control, false, backend);
}

NativeGenerationResult prepareNativeImageModel(const NativeGenerationRequest &request,
    const NativeGenerationOptions &options, const std::atomic_bool &cancelled,
    const NativeProgressCallback &progress)
{
    return dispatchNativeImage(request, options, cancelled, progress, {}, true);
}
NativeGenerationResult generateNativeImageWithPreview(const NativeGenerationRequest &request,
    const NativeGenerationOptions &options, NativeComputeBackend backend, const std::atomic_bool &cancelled,
    const NativeProgressCallback &progress, const NativePreviewCallback &preview,
    const std::shared_ptr<NativeExecutionControl> &control)
{
    return dispatchNativeImage(request, options, cancelled, progress, control, false, backend, preview);
}
}
