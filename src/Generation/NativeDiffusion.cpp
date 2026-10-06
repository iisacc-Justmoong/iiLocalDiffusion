#include "NativeDiffusion.hpp"
#include "NativePose.hpp"
#include "NativeCachePolicy.hpp"
#include "NativeDiskCache.hpp"
#include "NativeVaePolicy.hpp"
#include "NativeTelemetry.hpp"
#include "GenerationDefaults.hpp"
#include "UnifiedModel.hpp"
#include <cmath>
#include <algorithm>
#include <memory>
#include <map>
#include <regex>
#include <set>
#include <cctype>
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
    std::unique_ptr<sd_ctx_t, decltype(&free_sd_ctx)> refiner{nullptr, free_sd_ctx};
    std::string refinerIdentity, inspectedRefinerIdentity;
    sd_model_vae_info_t refinerVaeInfo{};
    std::unique_ptr<adetailer_ctx_t, decltype(&free_adetailer_ctx)> detailer{nullptr, free_adetailer_ctx};
    void reset() { detailer.reset(); refiner.reset(); context.reset(); refinerIdentity.clear(); ipIdentity.clear(); }
    std::string identity;
    std::string ipIdentity; // Retain loaded IP resources when a same-context request disables its inputs.
    std::string inspectedModelIdentity;
    sd_model_vae_info_t vaeInfo{};
    std::string missingVaeFamily;
    std::string validatedVaeIdentity;
    std::uint64_t budget = 0;
    std::vector<std::string> placementLogs;
};
EngineCache &engineCache() { static EngineCache cache; return cache; }
thread_local bool ownsEngine = false;
}
#endif
bool nativeDiffusionAvailable() noexcept { return IILD_HAS_NATIVE_DIFFUSION; }
ImageParameters nativeImageParameterDefaults(const std::filesystem::path &modelPath) {
    auto defaults = ImageParameters::defaults();
#if IILD_HAS_NATIVE_DIFFUSION
    sd_model_vae_info_t info{};
    if (sd_model_inspect_vae(modelPath.string().c_str(), &info) && info.model_family
        && std::string_view(info.model_family) == "krea2") {
        defaults.values["steps"] = std::int64_t(52);
        // Quick requests retain the common native CFG while using Krea2 steps.
        defaults.values["cfgScale"] = 7.0;
    }
#endif
    return defaults;
}
std::filesystem::path nativeGenerationResourceDirectory() { return native_detail::generationResourceDirectory(); }
void releaseNativeDiffusionCache() noexcept {
    releaseNativePoseCache();
#if IILD_HAS_NATIVE_DIFFUSION
    auto &cache = engineCache();
    ++cache.releaseEpoch;
    // May be called from a progress callback; never recurse into its lock.
    if (ownsEngine) return;
    std::unique_lock lock(cache.mutex, std::try_to_lock);
    if (lock.owns_lock()) {
        if (cache.context && std::getenv("IILD_NATIVE_DIAGNOSTICS"))
            std::fputs("iiLocalDiffusion cache: explicit idle release\n", stderr);
        cache.reset();
        sd_release_resident_model_memory();
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
    const NativePreviewCallback &preview = {}, const NativeGenerationResult *initial = nullptr, float strength = 1.0f,
    const std::filesystem::path &explicitVae = {}, const NativeModelComponents *components = nullptr,
    const NativeSamplingControls *sampling = nullptr, bool residentWeights = false,
    const NativeAdvancedControls *advanced = nullptr)
{
    NativeGenerationResult result;
    native_detail::NativeTelemetry telemetry;
    telemetry.note(residentWeights ? "weight_storage=anonymous" : "weight_storage=file-backed");
    telemetry.note("model=" + request.modelPath.filename().string() + " width=" + std::to_string(request.width)
        + " height=" + std::to_string(request.height) + " steps=" + std::to_string(request.steps)
        + " prepare_only=" + std::to_string(prepareOnly));
    const auto started = std::chrono::steady_clock::now();
    const auto pausedAtStart = control ? control->pausedDuration() : NativeExecutionControl::Clock::duration::zero();
    const auto timedOut = [&] {
        const auto paused = control ? control->pausedDuration() - pausedAtStart : NativeExecutionControl::Clock::duration::zero();
        return std::chrono::steady_clock::now() - started - paused >= std::chrono::milliseconds(request.timeoutMilliseconds);
    };
    const auto stopped = [&] {
        return cancelled || (control && !control->waitUntilRunnable(cancelled)) || timedOut();
    };
    const auto identifyWeight = [residentWeights](const std::filesystem::path &path) {
        return residentWeights ? native_detail::modelMetadataIdentity(path) : native_detail::modelIdentity(path);
    };
    try {
        if (advanced) {
            const std::vector<std::string> samplers{"auto", "euler", "heun", "euler_a", "dpmpp_2m", "dpmpp_sde", "ddim"};
            const std::vector<std::string> schedulers{"auto", "normal", "karras", "exponential", "sgm_uniform"};
            if (std::find(samplers.begin(), samplers.end(), advanced->sampler) == samplers.end()
                || std::find(schedulers.begin(), schedulers.end(), advanced->scheduler) == schedulers.end()
                || advanced->clipSkip < 0 || advanced->clipSkip > 12
                || !std::isfinite(advanced->eta) || advanced->eta < 0 || advanced->eta > 1
                || !std::isfinite(advanced->imageStrength) || advanced->imageStrength < 0 || advanced->imageStrength > 1
                || advanced->references.size() > 20
                || !std::isfinite(advanced->denoiseStrength) || advanced->denoiseStrength < 0 || advanced->denoiseStrength > 1
                || (advanced->detailer && advanced->denoiseStrength == 0)
                || !std::isfinite(advanced->refinerSwitch) || advanced->refinerSwitch < 0 || advanced->refinerSwitch > 1
                || (advanced->hires && (advanced->denoiseStrength == 0
                    || (advanced->upscaler != "lanczos" && advanced->upscaler != "nearest"
                        && advanced->upscaler != "bilinear" && advanced->upscaler != "bicubic"
                        && advanced->upscaler != "4x-ultra"))))
                throw std::runtime_error("Invalid or unsupported native advanced controls.");
            for (const auto &image : advanced->references)
                if (image.width < 1 || image.height < 1 || image.width > 4096 || image.height > 4096
                    || image.rgb.size() != std::size_t(image.width) * image.height * 3)
                    throw std::runtime_error("Invalid native reference RGB image.");
            if (advanced->controls.size() > 64)
                throw std::runtime_error("At most 64 applied ControlNets are supported.");
            if (advanced->ipAdapters.size() > 64)
                throw std::runtime_error("At most 64 IP-Adapters are supported.");
            for (const auto &item : advanced->ipAdapters) {
                const auto validImage = [](const NativeReferenceImage &image) {
                    return image.width > 0 && image.height > 0 && image.width <= 2048 && image.height <= 2048
                        && image.rgb.size() == std::size_t(image.width) * image.height * 3;
                };
                if (!validImage(item.image) || !std::isfinite(item.weight) || item.weight < 0 || item.weight > 2)
                    throw std::runtime_error("Invalid IP-Adapter RGB image or strength.");
                if ((item.mask.width || item.mask.height || !item.mask.rgb.empty()) && !validImage(item.mask))
                    throw std::runtime_error("Invalid IP-Adapter regional mask.");
            }
            for (const auto &item : advanced->controls) {
                const auto &image = item.image;
                if ((item.process != "Canny" && item.process != "Tile" && item.process != "Pose") || !std::isfinite(item.weight)
                    || item.weight < 0 || item.weight > 2 || image.width < 1 || image.height < 1
                    || image.width > 2048 || image.height > 2048
                    || image.rgb.size() != std::size_t(image.width) * image.height * 3)
                    throw std::runtime_error("Invalid native ControlNet process, weight or RGB image.");
                if (item.process == "Pose" && (!nativePoseAvailable()
                    || !item.poseDetector.is_absolute() || !item.poseModel.is_absolute()))
                    throw std::runtime_error("Pose requires native ONNX support and local detector/pose models.");
                const auto &mask = item.mask;
                if ((mask.width != 0 || mask.height != 0 || !mask.rgb.empty())
                    && (mask.width < 1 || mask.height < 1 || mask.width > 2048 || mask.height > 2048
                        || mask.rgb.size() != std::size_t(mask.width) * mask.height * 3))
                    throw std::runtime_error("Invalid ControlNet regional mask RGB image.");
            }
        }
        if (cancelled) { result.cancelled = true; telemetry.finish(result.error.empty() && !result.cancelled, result.cancelled ? "cancelled" : result.error); return result; }
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
        if (residentWeights && !request.q8CacheDirectory.empty())
            throw std::runtime_error("Memory-resident model loading cannot use a disk-backed Q8 cache.");
        if (sampling) {
            if (sampling->sampler < NativeSampler::Automatic || sampling->sampler > NativeSampler::Heun
                || (sampling->flowShift != std::numeric_limits<float>::infinity()
                    && (!std::isfinite(sampling->flowShift) || sampling->flowShift < 0 || sampling->flowShift > 4)))
                throw std::runtime_error("Invalid native sampler or flow shift.");
            const auto &sigmas = sampling->customSigmas;
            if (!sigmas.empty()) {
                if (sigmas.size() != std::size_t(request.steps + 1) || sigmas.back() != 0)
                    throw std::runtime_error("Native custom sigmas require steps + 1 entries ending in zero.");
                for (std::size_t i = 0; i < sigmas.size(); ++i)
                    if (!std::isfinite(sigmas[i]) || sigmas[i] < 0 || (i && sigmas[i - 1] <= sigmas[i]))
                        throw std::runtime_error("Native custom sigmas must be finite and strictly descending.");
            }
        }
        std::string componentIdentity;
        const auto identifyComponents = [&]() {
            std::string identity;
            if (components) for (const auto *path : {&components->clipL, &components->clipG, &components->t5xxl, &components->llm, &components->vae}) {
                identity += '|';
                if (path->empty()) continue;
                if (!path->is_absolute() || !std::filesystem::is_regular_file(*path)
                    || std::filesystem::canonical(*path) != *path)
                    throw std::runtime_error("Choose canonical local component weight files.");
                identity += identifyWeight(*path);
            }
            return identity;
        };
        if (components && (!std::isfinite(components->guidanceScale) || components->guidanceScale < 0
            || !std::isfinite(components->distilledGuidance) || components->distilledGuidance < 0))
            throw std::runtime_error("Invalid native guidance parameters.");
        componentIdentity = identifyComponents();
        const auto identifyEmbeddings = [&]() {
            std::string identity;
            std::set<std::string> names;
            if (advanced) {
                if (advanced->embeddings.size() > 64) throw std::runtime_error("At most 64 textual embeddings are allowed.");
                for (const auto &embedding : advanced->embeddings) {
                    if (!std::regex_match(embedding.token, std::regex("[a-z][a-z0-9_]{0,127}"))
                        || !names.insert(embedding.token).second || !embedding.path.is_absolute()
                        || !std::filesystem::is_regular_file(embedding.path)
                        || std::filesystem::canonical(embedding.path) != embedding.path)
                        throw std::runtime_error("Textual embeddings require unique lowercase tokens and canonical local weight files.");
                    identity += ':' + embedding.token + ':' + identifyWeight(embedding.path);
                }
            }
            return identity;
        };
        const auto embeddingIdentity = identifyEmbeddings();
        const auto identifyControls = [&]() {
            std::string identity;
            if (advanced) for (const auto &item : advanced->controls) {
                if (!item.model.is_absolute() || !std::filesystem::is_regular_file(item.model)
                    || std::filesystem::canonical(item.model) != item.model)
                    throw std::runtime_error("Choose a canonical local ControlNet weight file.");
                identity += ":controlnet=" + identifyWeight(item.model);
            }
            return identity;
        };
        const auto controlIdentity = identifyControls();
        const auto identifyIPAdapters = [&]() {
            std::string identity;
            if (advanced) for (const auto &item : advanced->ipAdapters) {
                for (const auto *path : {&item.model, &item.vision}) {
                    if (!path->is_absolute() || !std::filesystem::is_regular_file(*path)
                        || std::filesystem::canonical(*path) != *path)
                        throw std::runtime_error("Choose canonical local IP-Adapter and CLIP vision weight files.");
                    identity += ":ip=" + identifyWeight(*path);
                }
            }
            return identity;
        };
        const auto ipIdentity = identifyIPAdapters();
        std::string upscalerPath, upscalerIdentity;
        if (advanced && advanced->hires && advanced->upscaler == "4x-ultra") {
            const auto &path = advanced->upscalerModel;
            if (!path.is_absolute() || !std::filesystem::is_regular_file(path)
                || std::filesystem::canonical(path) != path)
                throw std::runtime_error("Choose a canonical local 4x ESRGAN model.");
            upscalerPath = path.string();
            upscalerIdentity = ":upscaler=" + identifyWeight(path);
        }
        std::string detailerPath, detailerIdentity;
        const bool useRefiner = advanced && advanced->refiner && advanced->refinerSwitch < 1.f;
        std::string refinerPath, refinerIdentity, refinerVaeIdentity;
        std::filesystem::path refinerVaePath;
        if (advanced && advanced->refiner) {
            const auto& path = advanced->refinerModel;
            if (!path.is_absolute() || !std::filesystem::is_regular_file(path)
                || std::filesystem::canonical(path) != path)
                throw std::runtime_error("Choose a canonical local SDXL Refiner checkpoint.");
            if (useRefiner) {
                refinerPath = path.string();
                refinerIdentity = identifyWeight(path);
            }
        }
        if (advanced && advanced->detailer) {
            const auto &path = advanced->detailerModel;
            if (!path.is_absolute() || !std::filesystem::is_regular_file(path)
                || std::filesystem::canonical(path) != path)
                throw std::runtime_error("Choose a canonical local converted YOLOv8 detector.");
            detailerPath = path.string();
            detailerIdentity = ":detailer=" + identifyWeight(path);
        }
        if (components && (components->prediction < NativePrediction::Automatic || components->prediction > NativePrediction::VPrediction))
            throw std::runtime_error("Invalid native prediction type.");
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
            if (cancelled) { result.cancelled = true; telemetry.finish(result.error.empty() && !result.cancelled, result.cancelled ? "cancelled" : result.error); return result; }
            if (timedOut()) throw std::runtime_error("Native image generation exceeded its time limit.");
        }
        if (control) control->waitUntilRunnable(cancelled);
        if (cancelled) { result.cancelled = true; telemetry.finish(result.error.empty() && !result.cancelled, result.cancelled ? "cancelled" : result.error); return result; }
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
            native_detail::NativeTelemetry &telemetry;
            EngineCache &cache;
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
        } callbacks{cancelled, progress, stopped, preview, telemetry, cache};
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
                    cache.reset();
                    if (cache.releaseEpoch != epoch) sd_release_resident_model_memory();
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
                state.telemetry.computing();
                if (state.progress) state.progress({NativeGenerationStage::Computing, state.completedGraphBatches, 0});
                return !state.stop();
            }, &callbacks);
        }
        sd_set_log_callback([](sd_log_level_t level, const char *text, void *opaque) {
            auto &state = *static_cast<Callbacks *>(opaque);
            if (text) {
                state.telemetry.log(text);
                if (std::string_view(text).find("IILD_BACKEND ") != std::string_view::npos) {
                    const std::lock_guard lock(state.logMutex);
                    state.cache.placementLogs.emplace_back(text);
                }
            }
            if (text && std::getenv("IILD_NATIVE_DIAGNOSTICS")) std::fputs(text, stderr);
            if (text && std::string_view(text).find("decoding ") != std::string_view::npos) {
                state.telemetry.stage("vae-decode");
                if (state.progress) state.progress({NativeGenerationStage::Decoding});
            }
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
            if (stage == SD_PROGRESS_SAMPLE) { state.samplingTotal = total; state.telemetry.stage("denoise", step, total); }
            else if (stage == SD_PROGRESS_DECODE) state.telemetry.stage("vae-decode", step, total);
            else state.telemetry.loading(step, total);
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
        const auto sourceIdentity = identifyWeight(request.modelPath);
        const auto defaults = options.defaultModifiers
            ? native_detail::loadGenerationDefaults(options.resourceDirectory) : native_detail::GenerationDefaults{};
        std::map<std::string, std::string> embeddingSources;
        std::set<std::string> explicitEmbeddingNames;
        for (const auto &embedding : defaults.embeddings) embeddingSources[embedding.token] = embedding.path.string();
        if (advanced) for (const auto &embedding : advanced->embeddings) {
            embeddingSources[embedding.token] = embedding.path.string();
            explicitEmbeddingNames.insert(embedding.token);
        }
        std::vector<sd_embedding_t> embeddings;
        for (const auto &[token, path] : embeddingSources) embeddings.push_back({token.c_str(), path.c_str()});
        auto effectiveModel = request.modelPath;
        if (!request.q8CacheDirectory.empty()) {
            Elapsed timing{result.preparationMilliseconds};
            callbacks.preparing = true;
            if (progress) progress({NativeGenerationStage::Preparing});
            const auto prepared = native_detail::prepareQ8Cache(request.modelPath, request.q8CacheDirectory,
                sourceIdentity, [&](const auto &output) {
                    // Conversion needs CPU working memory; evict previous GPU
                    // weights before starting its bounded streaming workers.
                    cache.reset();
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
        const auto effectiveIdentity = effectiveModel == request.modelPath
            ? sourceIdentity : identifyWeight(effectiveModel);
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
        const auto fallbackVae = !explicitVae.empty()
            ? native_detail::DefaultVae{explicitVae, identifyWeight(explicitVae)}
            : !cache.missingVaeFamily.empty()
                ? native_detail::loadFallbackVae(options.resourceDirectory, cache.missingVaeFamily) : native_detail::DefaultVae{};
        if (!fallbackVae.path.empty() && cache.validatedVaeIdentity != fallbackVae.identity) {
            // Newer engine families (for example Krea 2) have no SDK auto-mount
            // contract. Their explicitly supplied VAE is validated as part of
            // new_sd_ctx's full tensor assembly; never invent a fallback family.
            if (cache.vaeInfo.state != SD_VAE_UNSUPPORTED
                && !sd_model_validate_vae(model.c_str(), fallbackVae.path.string().c_str()))
                throw std::runtime_error("The selected VAE does not match the " + std::string(cache.vaeInfo.vae_family)
                    + " tensor contract: " + fallbackVae.path.string());
            cache.validatedVaeIdentity = fallbackVae.identity;
            if (std::getenv("IILD_NATIVE_DIAGNOSTICS"))
                std::fprintf(stderr, "iiLocalDiffusion VAE auto-mount-v2: model=%s family=%s source=%s path=%s\n",
                    cache.vaeInfo.model_family, cache.vaeInfo.vae_family,
                    explicitVae.empty() ? "fallback-validated" : "package-explicit", fallbackVae.path.string().c_str());
        }
        auto modifierIdentity = defaults.identity + fallbackVae.identity;
        for (const auto &lora : options.loras) modifierIdentity += ':' + identifyWeight(lora.path);
        const auto identity = sourceIdentity + ':' + effectiveIdentity + ':' + modifierIdentity + ':' + componentIdentity + embeddingIdentity + controlIdentity + upscalerIdentity + detailerIdentity
            + (components ? ":prediction=" + std::to_string(static_cast<int>(components->prediction)) : "")
            + (backend == NativeComputeBackend::Cpu ? ":cpu" : ":automatic")
            + (residentWeights ? ":resident-weights" : ":mapped-weights");
        if (cache.identity != identity || (!ipIdentity.empty() && cache.ipIdentity != ipIdentity)) {
            if (cache.context && std::getenv("IILD_NATIVE_DIAGNOSTICS"))
                std::fputs("iiLocalDiffusion cache: model identity changed\n", stderr);
            cache.reset();
        }
        // Do not rebuild a warm engine for small fluctuations in free memory.
        // Only explicit runtime release invalidates retained source residency.
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
            if (!native_detail::mayAttemptModelLoad(budget, residentWeights))
                throw std::runtime_error("There is not enough available memory to load this model. Try again after closing other apps.");
        }
        result.modelCacheHit = !!cache.context;
        result.memoryBudgetBytes = budget;
        telemetry.note("memory_budget_bytes=" + std::to_string(budget) + " cache_hit=" + std::to_string(result.modelCacheHit));
        const auto loadStarted = std::chrono::steady_clock::now();
        sd_ctx_params_t contextParameters;
        sd_ctx_params_init(&contextParameters);
        contextParameters.model_path = model.c_str();
        if (!ipIdentity.empty()) {
            const std::string inspectedFamily = cache.vaeInfo.model_family;
            if (inspectedFamily != "sd15" && inspectedFamily != "sdxl-base")
                throw std::runtime_error("Native IP-Adapter requires an SD 1.5 or SDXL base model.");
        }
        const bool useMultiControl = advanced && advanced->controls.size() > 1;
        const auto controlPath = advanced && !advanced->controls.empty()
            ? advanced->controls.front().model.string() : std::string{};
        if (!controlPath.empty()) {
            const std::string inspectedFamily = cache.vaeInfo.model_family;
            if (inspectedFamily != "sd15" && inspectedFamily != "sdxl-base")
                throw std::runtime_error("Native ControlNet currently requires an SD 1.5 or SDXL base model.");
            if (!useMultiControl) contextParameters.control_net_path = controlPath.c_str();
        }
        if (components && components->prediction != NativePrediction::Automatic)
            contextParameters.prediction = components->prediction == NativePrediction::Epsilon ? EPS_PRED : V_PRED;
        const auto clipL = components ? components->clipL.string() : std::string{};
        const auto clipG = components ? components->clipG.string() : std::string{};
        const auto t5 = components ? components->t5xxl.string() : std::string{};
        const auto llm = components ? components->llm.string() : std::string{};
        if (!clipL.empty()) contextParameters.clip_l_path = clipL.c_str();
        if (!clipG.empty()) contextParameters.clip_g_path = clipG.c_str();
        if (!t5.empty()) contextParameters.t5xxl_path = t5.c_str();
        if (!llm.empty()) contextParameters.llm_path = llm.c_str();
        const auto vaePath = fallbackVae.path.string();
        if (!vaePath.empty()) contextParameters.vae_path = vaePath.c_str();
        contextParameters.embeddings = embeddings.data();
        contextParameters.embedding_count = static_cast<uint32_t>(embeddings.size());
        contextParameters.n_threads = result.threads;
        contextParameters.enable_mmap = true;
        contextParameters.memory_resident_model = residentWeights;
#if IILD_NATIVE_CPU_VAE
        // iOS Metal VAE decoding can lose command buffers during GPU recovery.
        // Keep tiled decoding and its weights on CPU; other modules still use
        // automatic compute placement (Metal on Apple). An explicit backend
        // disables upstream auto-fit, so keep their weights reloadable from
        // the mapped file instead of accumulating non-evictable GPU residency.
        contextParameters.backend = "vae=cpu";
        contextParameters.params_backend = residentWeights ? "vae=cpu" : "te=disk,diffusion=disk,vae=cpu";
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
            cache.placementLogs.clear();
            std::vector<std::pair<std::string, std::string>> ipPaths;
            if (advanced) for (const auto &item : advanced->ipAdapters)
                ipPaths.emplace_back(item.model.string(), item.vision.string());
            std::vector<sd_ip_adapter_model_t> ipModels;
            for (const auto &[modelPath, visionPath] : ipPaths)
                ipModels.push_back({modelPath.c_str(), visionPath.c_str()});
            cache.context.reset(ipModels.empty() ? new_sd_ctx(&contextParameters)
                : new_sd_ctx_with_ip_adapters(&contextParameters, ipModels.data(), static_cast<uint32_t>(ipModels.size())));
            cache.identity = identity;
            cache.ipIdentity = ipIdentity;
            cache.budget = budget;
        }
        for (const auto &placement : cache.placementLogs) telemetry.log(placement);
        auto *context = cache.context.get();
        result.modelLoadMilliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - loadStarted).count();
        if (cancelled) { result.cancelled = true; telemetry.finish(result.error.empty() && !result.cancelled, result.cancelled ? "cancelled" : result.error); return result; }
        if (timedOut()) throw std::runtime_error("Native image generation exceeded its time limit.");
        if (!context || !sd_ctx_supports_image_generation(context))
            throw std::runtime_error(callbacks.failure("This model could not be loaded by the native image engine."));
        if (!useMultiControl && !controlPath.empty() && !sd_ctx_has_control_net(context))
            throw std::runtime_error("The selected ControlNet could not be loaded for this model.");
        if (useMultiControl) {
            std::vector<std::string> paths;
            for (const auto& item : advanced->controls) paths.push_back(item.model.string());
            std::vector<const char*> pointers;
            for (const auto& path : paths) pointers.push_back(path.c_str());
            if (!sd_prepare_control_nets(context, pointers.data(), static_cast<uint32_t>(pointers.size())))
                throw std::runtime_error(callbacks.failure("Cannot prepare every selected ControlNet."));
            if (stopped()) throw std::runtime_error("ControlNet preparation interrupted.");
            result.modelLoadMilliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - loadStarted).count();
            telemetry.note("resident_controlnets_ready=" + std::to_string(pointers.size()));
        }
        if (useRefiner) {
            if (std::string(sd_get_model_family(context)) != "sdxl-base")
                throw std::runtime_error("Refiner requires an SDXL Base model.");
            if (cache.inspectedRefinerIdentity != refinerIdentity) {
                if (!sd_model_inspect_vae(refinerPath.c_str(), &cache.refinerVaeInfo)
                    || std::string(cache.refinerVaeInfo.model_family) != "sdxl-refiner")
                    throw std::runtime_error("The selected checkpoint is not an SDXL Refiner model.");
                cache.inspectedRefinerIdentity = refinerIdentity;
            }
            auto refinerVae = fallbackVae;
            if (refinerVae.path.empty() && cache.refinerVaeInfo.state != SD_VAE_EMBEDDED)
                refinerVae = native_detail::loadFallbackVae(options.resourceDirectory, "sdxl-base");
            refinerVaePath = refinerVae.path;
            refinerVaeIdentity = refinerVaePath.empty() ? std::string{} : identifyWeight(refinerVaePath);
            const auto refinerKey = refinerIdentity + ':' + identity + ':' + refinerVae.identity + ':' + refinerVaeIdentity;
            const bool refinerHit = cache.refiner && cache.refinerIdentity == refinerKey;
            result.modelCacheHit = result.modelCacheHit && refinerHit;
            if (!refinerHit) {
                if (!refinerVaePath.empty() && !sd_model_validate_vae(refinerPath.c_str(), refinerVaePath.string().c_str()))
                    throw std::runtime_error("The selected VAE does not match the Refiner SDXL tensor contract.");
                auto refinerParameters = contextParameters;
                refinerParameters.model_path = refinerPath.c_str();
                refinerParameters.clip_l_path = refinerParameters.clip_g_path = nullptr;
                refinerParameters.t5xxl_path = refinerParameters.llm_path = nullptr;
                refinerParameters.control_net_path = nullptr;
                refinerParameters.prediction = PREDICTION_COUNT;
                const auto vae = refinerVaePath.string();
                refinerParameters.vae_path = vae.empty() ? nullptr : vae.c_str();
                // Automatic Base embeddings are architecture-specific. Only
                // explicitly chosen embeddings are shared with the Refiner.
                std::vector<sd_embedding_t> refinerEmbeddings;
                for (const auto& embedding : embeddings)
                    if (explicitEmbeddingNames.contains(embedding.name)) refinerEmbeddings.push_back(embedding);
                refinerParameters.embeddings = refinerEmbeddings.data();
                refinerParameters.embedding_count = static_cast<uint32_t>(refinerEmbeddings.size());
                cache.refiner.reset(new_sd_ctx(&refinerParameters));
                cache.refinerIdentity = refinerKey;
            }
            if (!cache.refiner || !sd_ctx_can_refine(context, cache.refiner.get()))
                throw std::runtime_error(callbacks.failure("The selected Base and Refiner checkpoints are incompatible."));
            if (stopped()) throw std::runtime_error("Refiner preparation interrupted.");
            if (!sd_set_freeu(cache.refiner.get(), advanced->freeU))
                throw std::runtime_error("FreeU could not be configured on the Refiner.");
            sd_set_prompt_weighting(cache.refiner.get(), advanced->promptWeighting);
            for (const auto& embedding : advanced->embeddings)
                if (!sd_load_textual_embedding(cache.refiner.get(), embedding.token.c_str()))
                    throw std::runtime_error("The selected embedding has no compatible Refiner bigG weights: " + embedding.token);
            result.modelLoadMilliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - loadStarted).count();
            telemetry.note("resident_refiner_ready=1 cache_hit=" + std::to_string(refinerHit));
        }
        // Reapply on every request, including legacy calls sharing this context.
        if (!sd_set_freeu(context, advanced && advanced->freeU))
            throw std::runtime_error("FreeU requires a supported SD 1.5, SD 2 or SDXL UNet model.");
        if (!upscalerPath.empty()) {
            if (!sd_prepare_hires_upscaler(context, upscalerPath.c_str()))
                throw std::runtime_error(callbacks.failure("Cannot prepare the selected 4x ESRGAN upscaler."));
            if (stopped()) throw std::runtime_error("Upscaler preparation interrupted.");
            telemetry.note("resident_upscaler_ready=1");
            result.modelLoadMilliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - loadStarted).count();
        }
        if (!detailerPath.empty()) {
            if (!cache.detailer)
                cache.detailer.reset(new_resident_adetailer_ctx(detailerPath.c_str(), result.threads, contextParameters.backend));
            if (!cache.detailer)
                throw std::runtime_error(callbacks.failure("Cannot prepare the selected YOLOv8 Detailer detector."));
            if (stopped()) throw std::runtime_error("Detailer preparation interrupted.");
            telemetry.note("resident_detailer_ready=1");
            result.modelLoadMilliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - loadStarted).count();
        }
        if (residentWeights) telemetry.note("resident_model_ready=1");
        if (prepareOnly) {
            if (identifyWeight(request.modelPath) != sourceIdentity)
                throw std::runtime_error("The local model changed during preparation. Try again.");
            static const bool preparationCleanup = [] {
                std::atexit([] { releaseNativeDiffusionCache(); });
                return true;
            }();
            (void)preparationCleanup;
            cacheUse.successful = true;
            telemetry.finish(result.error.empty() && !result.cancelled, result.cancelled ? "cancelled" : result.error); return result;
        }
        callbacks.context = context;
        sd_img_gen_params_t parameters;
        sd_img_gen_params_init(&parameters);
        parameters.prompt = request.prompt.c_str();
        const std::string family = sd_get_model_family(context);
        sd_set_prompt_weighting(context, !advanced || advanced->promptWeighting);
        auto effectivePrompt = request.prompt;
        const auto mentions = [](std::string text, const std::string &token) {
            std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return char(std::tolower(c)); });
            const auto word = [](unsigned char c) { return std::isalnum(c) || c == '_'; };
            for (auto at = text.find(token); at != std::string::npos; at = text.find(token, at + 1))
                if ((at == 0 || !word(text[at - 1])) && (at + token.size() == text.size() || !word(text[at + token.size()]))) return true;
            return false;
        };
        if (advanced) for (const auto &embedding : advanced->embeddings) {
            if (!sd_load_textual_embedding(context, embedding.token.c_str()))
                throw std::runtime_error("The selected textual embedding is incompatible with this model or could not be loaded: " + embedding.token);
            if (!mentions(effectivePrompt, embedding.token) && !mentions(options.negativePrompt, embedding.token))
                effectivePrompt += " " + embedding.token;
        }
        parameters.prompt = effectivePrompt.c_str();
        // Preprocessing mutates only request-owned scratch RGB, never caller input.
        std::vector<std::vector<uint8_t>> controlPixels;
        std::vector<sd_control_input_t> controlInputs;
        sd_image_t controlMask{};
        if (advanced && !advanced->controls.empty()) {
            controlPixels.resize(advanced->controls.size());
            for (size_t index = 0; index < advanced->controls.size(); ++index) {
                const auto& item = advanced->controls[index];
                if (item.process == "Pose") {
                    if (progress) progress({NativeGenerationStage::Preparing});
                    // Bridge the generation's pause-adjusted deadline into ORT's
                    // cancellation token without mutating caller-owned cancellation.
                    std::atomic_bool poseCancelled{cancelled.load()};
                    std::jthread deadline([&](std::stop_token stop) {
                        while (!stop.stop_requested()) {
                            if (cancelled || timedOut()) { poseCancelled = true; return; }
                            std::this_thread::sleep_for(std::chrono::milliseconds(10));
                        }
                    });
                    auto pose = processNativePose(item.poseDetector,item.poseModel,item.image,
                        poseCancelled,result.threads,control);
                    controlPixels[index] = std::move(pose.image.rgb);
                    telemetry.note("pose_backend=onnx-cpu cache_hit=" + std::to_string(pose.modelCacheHit)
                        + " model_bytes=" + std::to_string(pose.modelBytes) + " threads=" + std::to_string(pose.threads));
                } else controlPixels[index] = item.image.rgb;
                sd_control_input_t input{};
                input.image = {static_cast<uint32_t>(item.image.width),
                    static_cast<uint32_t>(item.image.height), 3, controlPixels[index].data()};
                input.strength = item.weight;
                if (!item.mask.rgb.empty()) input.mask = {static_cast<uint32_t>(item.mask.width),
                    static_cast<uint32_t>(item.mask.height), 3, const_cast<uint8_t *>(item.mask.rgb.data())};
                if (item.process == "Canny"
                    && !preprocess_canny(input.image, 0.08f, 0.08f, 0.8f, 1.0f, false))
                    throw std::runtime_error("Native Canny preprocessing failed.");
                if (stopped()) throw std::runtime_error("ControlNet preprocessing interrupted.");
                controlInputs.push_back(input);
            }
            parameters.control_image = controlInputs.front().image;
            parameters.control_strength = useMultiControl ? 1.f : controlInputs.front().strength;
            if (!useMultiControl) controlMask = controlInputs.front().mask;
        }
        if (!sd_set_control_net_inputs(context, useMultiControl ? controlInputs.data() : nullptr,
                useMultiControl ? static_cast<uint32_t>(controlInputs.size()) : 0))
            throw std::runtime_error("Cannot prepare every ControlNet input.");
        // Reset each request, including legacy calls sharing a cached context.
        if (!sd_set_control_net_mask(context, controlMask))
            throw std::runtime_error("Cannot prepare ControlNet regional mask.");
        std::vector<sd_ip_adapter_input_t> ipInputs;
        if (advanced) for (const auto &item : advanced->ipAdapters) {
            sd_ip_adapter_input_t input{};
            input.slot = static_cast<uint32_t>(ipInputs.size());
            input.image = {static_cast<uint32_t>(item.image.width), static_cast<uint32_t>(item.image.height),
                3, const_cast<uint8_t *>(item.image.rgb.data())};
            input.strength = item.weight;
            if (!item.mask.rgb.empty()) input.mask = {static_cast<uint32_t>(item.mask.width),
                static_cast<uint32_t>(item.mask.height), 3, const_cast<uint8_t *>(item.mask.rgb.data())};
            ipInputs.push_back(input);
        }
        // Always reset inputs, including a legacy request reusing a resident IP context.
        if (!sd_set_ip_adapter_inputs(context, ipInputs.data(), static_cast<uint32_t>(ipInputs.size())))
            throw std::runtime_error("Cannot prepare every IP-Adapter input.");
        std::vector<sd_image_t> referenceImages;
        if (advanced && !advanced->references.empty()) {
            // Base Krea2 is an img2img denoiser. Its upstream reference preset
            // assumes separately trained Ostris edit weights; do not activate
            // that preset merely because the transformer can accept tokens.
            const int capacity = family == "krea2" ? 0 : sd_ctx_reference_image_capacity(context);
            if (advanced->references.size() > std::size_t(std::max(1, capacity)))
                throw std::runtime_error("The selected model cannot condition on this many reference images. Choose a multi-reference editing model.");
            for (const auto &image : advanced->references)
                referenceImages.push_back({static_cast<uint32_t>(image.width), static_cast<uint32_t>(image.height),
                    3, const_cast<uint8_t *>(image.rgb.data())});
            // The first reference establishes the img2img starting latent.
            // Editing models additionally receive every ordered reference.
            parameters.init_image = referenceImages.front();
            parameters.strength = advanced->imageStrength;
            if (capacity > 0) {
                parameters.ref_images = referenceImages.data();
                parameters.ref_images_count = static_cast<int>(referenceImages.size());
            }
        }
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
            if (!explicitEmbeddingNames.contains(embedding.token)
                && (family == "sdxl-base" || (family == "sd15" && embedding.sd15)))
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
        if (components) {
            parameters.sample_params.guidance.txt_cfg = components->guidanceScale;
            parameters.sample_params.guidance.distilled_guidance = components->distilledGuidance;
            parameters.hires.enabled = components->hires;
            if (!components->hires) {
                parameters.width = parameters.hires.target_width;
                parameters.height = parameters.hires.target_height;
            }
        }
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
        if (sampling && sampling->sampler != NativeSampler::Automatic)
            parameters.sample_params.sample_method = sampling->sampler == NativeSampler::Euler ? EULER_SAMPLE_METHOD : HEUN_SAMPLE_METHOD;
        parameters.sample_params.scheduler = sd_get_default_scheduler(context, parameters.sample_params.sample_method);
        if (advanced) {
            const std::map<std::string, sample_method_t> samplers{{"euler", EULER_SAMPLE_METHOD},
                {"heun", HEUN_SAMPLE_METHOD}, {"euler_a", EULER_A_SAMPLE_METHOD}, {"dpmpp_2m", DPMPP2M_SAMPLE_METHOD},
                {"dpmpp_sde", DPMPP2M_SDE_SAMPLE_METHOD}, {"ddim", DDIM_TRAILING_SAMPLE_METHOD}};
            const std::map<std::string, scheduler_t> schedulers{{"normal", DISCRETE_SCHEDULER},
                {"karras", KARRAS_SCHEDULER}, {"exponential", EXPONENTIAL_SCHEDULER}, {"sgm_uniform", SGM_UNIFORM_SCHEDULER}};
            if (advanced->sampler != "auto") parameters.sample_params.sample_method = samplers.at(advanced->sampler);
            parameters.sample_params.scheduler = advanced->scheduler == "auto"
                ? sd_get_default_scheduler(context, parameters.sample_params.sample_method) : schedulers.at(advanced->scheduler);
            parameters.clip_skip = advanced->clipSkip == 0 ? -1 : advanced->clipSkip;
            parameters.sample_params.eta = advanced->eta;
            parameters.circular_x = parameters.circular_y = advanced->seamlessTiling;
            parameters.hires.enabled = advanced->hires;
            if (advanced->hires) {
                parameters.width = request.width / 2;
                parameters.height = request.height / 2;
                const std::map<std::string, sd_hires_upscaler_t> upscalers{
                    {"nearest", SD_HIRES_UPSCALER_NEAREST}, {"bilinear", SD_HIRES_UPSCALER_BILINEAR},
                    {"bicubic", SD_HIRES_UPSCALER_BICUBIC}, {"lanczos", SD_HIRES_UPSCALER_LANCZOS},
                    {"4x-ultra", SD_HIRES_UPSCALER_MODEL}};
                parameters.hires.upscaler = upscalers.at(advanced->upscaler);
                parameters.hires.model_path = upscalerPath.c_str();
                parameters.hires.denoising_strength = advanced->denoiseStrength;
                parameters.hires.steps = std::max(1, int(request.steps * advanced->denoiseStrength));
            }
        }
        // Keep request-owned schedule storage alive and writable for the engine.
        auto customSigmas = sampling ? sampling->customSigmas : std::vector<float>{};
        if (sampling) {
            parameters.sample_params.flow_shift = sampling->flowShift;
            parameters.sample_params.custom_sigmas = customSigmas.empty() ? nullptr : customSigmas.data();
            parameters.sample_params.custom_sigmas_count = static_cast<int>(customSigmas.size());
        }
        // Match the Python Krea2 path for native app calls. Explicit schedules
        // and advanced scheduler choices remain authoritative.
        if (family == "krea2" && customSigmas.empty()
            && (!advanced || advanced->scheduler == "auto")) {
            const float mu = sampling && std::isfinite(sampling->flowShift) ? sampling->flowShift
                : .5f + float((parameters.width / 16) * (parameters.height / 16) - 256) * (.65f / 6144.f);
            const float factor = std::exp(mu);
            customSigmas.reserve(std::size_t(request.steps + 1));
            for (int i = 0; i < request.steps; ++i) {
                const float sigma = float(request.steps - i) / float(request.steps);
                customSigmas.push_back(factor * sigma / (1 + (factor - 1) * sigma));
            }
            customSigmas.push_back(0.f);
            parameters.sample_params.flow_shift = mu;
            parameters.sample_params.custom_sigmas = customSigmas.data();
            parameters.sample_params.custom_sigmas_count = static_cast<int>(customSigmas.size());
            if ((!sampling || sampling->sampler == NativeSampler::Automatic)
                && (!advanced || advanced->sampler == "auto"))
                parameters.sample_params.sample_method = EULER_SAMPLE_METHOD;
        }
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
        telemetry.stage("text-encode");
        if (progress) progress({NativeGenerationStage::Encoding});
        bool ok = false;
        {
            Elapsed timing{result.generationMilliseconds};
            ok = useRefiner
                ? generate_image_with_refiner(context, cache.refiner.get(), advanced->refinerSwitch,
                    &parameters, &images.data, &images.count, options.negativePrompt.c_str())
                : generate_image(context, &parameters, &images.data, &images.count);
        }
        if (cancelled) { result.cancelled = true; telemetry.finish(result.error.empty() && !result.cancelled, result.cancelled ? "cancelled" : result.error); return result; }
        if (timedOut()) throw std::runtime_error("Native image generation exceeded its time limit.");
        if (!ok || !images.data || images.count != 1 || !images.data[0].data || images.data[0].channel != 3)
            throw std::runtime_error(callbacks.failure("The native engine did not return a complete RGB image."));
        if (advanced && advanced->detailer) {
            if (stopped()) throw std::runtime_error("Detailer interrupted before detection.");
            // Whole-image hints/masks use different coordinates from Detailer crops.
            // Retain model resources; the next request repopulates its own inputs.
            if (!sd_set_ip_adapter_inputs(context, nullptr, 0)
                || !sd_set_control_net_inputs(context, nullptr, 0)
                || !sd_set_control_net_mask(context, {}))
                throw std::runtime_error("Cannot reset whole-image conditioning before Detailer.");
            telemetry.stage("detailer");
            // A crop preview must not replace the complete-canvas preview.
            sd_set_preview_callback(nullptr, PREVIEW_NONE, 1, false, false, nullptr);
            auto inpaint = parameters;
            inpaint.width = inpaint.height = 512;
            inpaint.strength = advanced->denoiseStrength;
            const sd_adetailer_params_t detailParameters{nullptr, nullptr, nullptr};
            Images detailed;
            double elapsed = 0;
            {
                Elapsed timing{elapsed};
                ok = adetail_image(cache.detailer.get(), context, images.data[0], &detailParameters,
                    &inpaint, &detailed.data, &detailed.count);
            }
            result.generationMilliseconds += elapsed;
            if (stopped()) throw std::runtime_error("Detailer interrupted.");
            if (!ok || !detailed.data || detailed.count != 1 || !detailed.data[0].data
                || detailed.data[0].channel != 3 || detailed.data[0].width != images.data[0].width
                || detailed.data[0].height != images.data[0].height)
                throw std::runtime_error(callbacks.failure("Detailer detection or masked regeneration failed."));
            std::swap(images.data, detailed.data);
            std::swap(images.count, detailed.count);
        }
        telemetry.stage("postprocess");
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
        if (identifyWeight(request.modelPath) != sourceIdentity
            || identifyWeight(effectiveModel) != effectiveIdentity
            || identifyComponents() != componentIdentity || identifyEmbeddings() != embeddingIdentity
            || identifyControls() != controlIdentity
            || identifyIPAdapters() != ipIdentity
            || (!upscalerPath.empty() && ":upscaler=" + identifyWeight(upscalerPath) != upscalerIdentity)
            || (!detailerPath.empty() && ":detailer=" + identifyWeight(detailerPath) != detailerIdentity))
            throw std::runtime_error("The local model changed during generation. Try again.");
        if (useRefiner && (identifyWeight(refinerPath) != refinerIdentity
            || (!refinerVaePath.empty() && identifyWeight(refinerVaePath) != refinerVaeIdentity)))
            throw std::runtime_error("The local model changed during generation. Try again.");
        auto finalModifierIdentity = options.defaultModifiers
            ? native_detail::loadGenerationDefaults(options.resourceDirectory).identity : std::string{};
        if (!explicitVae.empty())
            finalModifierIdentity += identifyWeight(explicitVae);
        else if (!cache.missingVaeFamily.empty())
            finalModifierIdentity += native_detail::loadFallbackVae(options.resourceDirectory, cache.missingVaeFamily).identity;
        for (const auto &lora : options.loras) finalModifierIdentity += ':' + identifyWeight(lora.path);
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
    telemetry.finish(result.error.empty() && !result.cancelled, result.cancelled ? "cancelled" : result.error); return result;
}

static NativeGenerationResult dispatchNativeImage(const NativeGenerationRequest &request,
    const NativeGenerationOptions &options, const std::atomic_bool &cancelled,
    const NativeProgressCallback &progress, const std::shared_ptr<NativeExecutionControl> &control,
    bool prepareOnly, NativeComputeBackend backend = NativeComputeBackend::Automatic,
    const NativePreviewCallback &preview = {}, bool residentWeights = false)
{
    std::error_code error;
    if (!std::filesystem::is_directory(request.modelPath, error)
        && !native_detail::isUnifiedModelPackage(request.modelPath))
        return nativeImage(request, options, cancelled, progress, control, prepareOnly, backend, preview,
                           nullptr, 1.0f, {}, nullptr, nullptr, residentWeights);
    NativeGenerationResult result;
    try {
        if (cancelled) { result.cancelled = true; return result; }
        if (residentWeights && native_detail::isUnifiedModelPackage(request.modelPath))
            throw std::runtime_error("Memory-resident generation requires an unpacked unified-model directory; archive extraction to disk is disabled.");
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
                }) : NativePreviewCallback{}, i ? &result : nullptr, stage.strength, stage.vae,
                nullptr, nullptr, residentWeights);
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

NativeGenerationResult generateNativeImageWithResidentWeights(const NativeGenerationRequest &request,
    const NativeGenerationOptions &options, const std::atomic_bool &cancelled,
    const NativeProgressCallback &progress, const std::shared_ptr<NativeExecutionControl> &control)
{
    return dispatchNativeImage(request, options, cancelled, progress, control, false,
                               NativeComputeBackend::Automatic, {}, true);
}

NativeGenerationResult generateNativeImageWithResidentWeights(const NativeGenerationRequest &request,
    const NativeGenerationOptions &options, NativeComputeBackend backend, const std::atomic_bool &cancelled,
    const NativeProgressCallback &progress, const NativePreviewCallback &preview,
    const std::shared_ptr<NativeExecutionControl> &control)
{
    return dispatchNativeImage(request, options, cancelled, progress, control, false, backend, preview, true);
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
NativeGenerationResult prepareNativeImageModelWithResidentWeights(const NativeGenerationRequest &request,
    const NativeGenerationOptions &options, const std::atomic_bool &cancelled,
    const NativeProgressCallback &progress)
{
    return dispatchNativeImage(request, options, cancelled, progress, {}, true,
                               NativeComputeBackend::Automatic, {}, true);
}
NativeGenerationResult generateNativeImageWithPreview(const NativeGenerationRequest &request,
    const NativeGenerationOptions &options, NativeComputeBackend backend, const std::atomic_bool &cancelled,
    const NativeProgressCallback &progress, const NativePreviewCallback &preview,
    const std::shared_ptr<NativeExecutionControl> &control)
{
    return dispatchNativeImage(request, options, cancelled, progress, control, false, backend, preview);
}
NativeGenerationResult generateNativeImageWithComponents(const NativeGenerationRequest &request,
    const NativeGenerationOptions &options, const NativeModelComponents &components,
    NativeComputeBackend backend, const std::atomic_bool &cancelled, bool prepareOnly,
    const NativeProgressCallback &progress, const NativePreviewCallback &preview,
    const std::shared_ptr<NativeExecutionControl> &control)
{
    return nativeImage(request, options, cancelled, progress, control, prepareOnly, backend,
                       preview, nullptr, 1.0f, components.vae, &components);
}
NativeGenerationResult generateNativeAdvancedImage(const NativeGenerationRequest &request,
    const NativeGenerationOptions &options, const NativeModelComponents &components,
    const NativeAdvancedControls &advanced, NativeComputeBackend backend,
    const std::atomic_bool &cancelled, const NativeProgressCallback &progress,
    const NativePreviewCallback &preview, const std::shared_ptr<NativeExecutionControl> &control)
{
    auto resolved = components;
    resolved.hires = advanced.hires;
    return nativeImage(request, options, cancelled, progress, control, false, backend,
                       preview, nullptr, 1.0f, resolved.vae, &resolved, nullptr, true, &advanced);
}
NativeGenerationResult generateNativeImageWithSampling(const NativeGenerationRequest &request,
    const NativeGenerationOptions &options, const NativeModelComponents &components,
    const NativeSamplingControls &sampling, NativeComputeBackend backend,
    const std::atomic_bool &cancelled, bool prepareOnly, const NativeProgressCallback &progress,
    const NativePreviewCallback &preview, const std::shared_ptr<NativeExecutionControl> &control)
{
    return nativeImage(request, options, cancelled, progress, control, prepareOnly, backend,
                       preview, nullptr, 1.0f, components.vae, &components, &sampling);
}
NativeGenerationResult generateNativeImageWithResidentWeights(const NativeGenerationRequest &request,
    const NativeGenerationOptions &options, const NativeModelComponents &components,
    const NativeSamplingControls &sampling, NativeComputeBackend backend,
    const std::atomic_bool &cancelled, bool prepareOnly, const NativeProgressCallback &progress,
    const NativePreviewCallback &preview, const std::shared_ptr<NativeExecutionControl> &control)
{
    return nativeImage(request, options, cancelled, progress, control, prepareOnly, backend,
                       preview, nullptr, 1.0f, components.vae, &components, &sampling, true);
}
}
