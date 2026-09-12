#include "NativeDiffusion.hpp"
#include "NativeCachePolicy.hpp"
#include "NativeDiskCache.hpp"
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
    std::uint64_t budget = 0;
};
EngineCache &engineCache() { static EngineCache cache; return cache; }
thread_local bool ownsEngine = false;
}
#endif
bool nativeDiffusionAvailable() noexcept { return IILD_HAS_NATIVE_DIFFUSION; }
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
            sd_ctx_t *context = nullptr;
            bool preparing = false;
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
                sd_set_log_callback(nullptr, nullptr);
            }
        } callbacks{cancelled, progress, stopped};
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
        const auto identity = sourceIdentity + ':' + effectiveIdentity;
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
            budget = native_detail::memoryBudget(limits);
            if (std::getenv("IILD_NATIVE_DIAGNOSTICS"))
                std::fprintf(stderr, "iiLocalDiffusion resources: physical=%llu recommended=%llu available=%llu budget=%llu\n",
                    static_cast<unsigned long long>(limits.physical), static_cast<unsigned long long>(limits.recommended),
                    static_cast<unsigned long long>(limits.available), static_cast<unsigned long long>(budget));
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
        contextParameters.n_threads = result.threads;
        contextParameters.enable_mmap = true;
        // This API takes a checkpoint, not mutable LoRA merges. AUTO otherwise
        // requests writable mappings even when no LoRA is present.
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
        callbacks.context = context;
        sd_img_gen_params_t parameters;
        sd_img_gen_params_init(&parameters);
        parameters.prompt = request.prompt.c_str();
        // The pinned engine aligns UNet requests to VAE(8) * UNet(8).
        // Make its internal canvas explicit, then center-crop the returned RGB
        // to preserve our public 8-pixel output-size contract without rescaling.
        parameters.width = (request.width + 63) / 64 * 64;
        parameters.height = (request.height + 63) / 64 * 64;
        parameters.seed = request.seed;
        parameters.batch_count = 1;
        parameters.sample_params.sample_steps = request.steps;
        parameters.sample_params.sample_method = sd_get_default_sample_method(context);
        parameters.sample_params.scheduler = sd_get_default_scheduler(context, parameters.sample_params.sample_method);
        parameters.vae_tiling_params.enabled = true;
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
        if (decoded.width != static_cast<unsigned>(parameters.width)
            || decoded.height != static_cast<unsigned>(parameters.height))
            throw std::runtime_error("The native engine returned an unexpected image size: expected "
                + std::to_string(parameters.width) + "x" + std::to_string(parameters.height)
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
}
