#include "NativeVideo.hpp"
#include "NativeCachePolicy.hpp"
#include "NativeEngineAccess.hpp"
#include "NativeTelemetry.hpp"
#include "ModelPackaging/ModelPackaging.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <limits>
#include <mutex>
#include <stop_token>
#include <thread>
#if IILD_HAS_NATIVE_DIFFUSION
#include <stable-diffusion.h>
#endif

namespace iiLocalDiffusion {
NativeVideoResult generateNativeVideo(const NativeVideoRequest &request, const std::atomic_bool &cancelled,
    const NativeProgressCallback &progress, const NativeVideoFrameObserver &frame)
{
    NativeVideoResult result;
    native_detail::NativeTelemetry telemetry;
    struct TraceEnd {
        native_detail::NativeTelemetry &telemetry;
        NativeVideoResult &result;
        ~TraceEnd() { telemetry.finish(result.error.empty() && !result.cancelled,
            result.cancelled ? "cancelled" : result.error); }
    } traceEnd{telemetry, result};
    try {
        if (request.width < 32 || request.height < 32 || request.width > 4096 || request.height > 4096
            || request.width % 32 || request.height % 32 || request.steps < 1 || request.steps > 1000
            || !std::isfinite(request.guidanceScale) || request.guidanceScale < 0 || request.guidanceScale > 100
            || request.shots.empty() || request.shots.size() > 256 || !frame)
            throw std::invalid_argument("Invalid native video dimensions, sampling parameters or shots");
        const auto source = std::filesystem::absolute(request.modelPath).lexically_normal();
        if (!std::filesystem::is_regular_file(source) || std::filesystem::canonical(source) != source)
            throw std::invalid_argument("Choose a local, unredirected video checkpoint");
        const auto identity = native_detail::modelMetadataIdentity(source);
        const auto validateImage = [&](const NativeReferenceImage &image) {
            if (!image.rgb.empty() && (image.width != request.width || image.height != request.height
                || image.rgb.size() != static_cast<std::size_t>(request.width) * static_cast<std::size_t>(request.height) * 3))
                throw std::invalid_argument("Video keyframes must be complete RGB at output size");
        };
        for (const auto &shot : request.shots) {
            if (shot.prompt.empty() || shot.prompt.size() > 128000 || shot.negativePrompt.size() > 128000
                || shot.frames < 2 || shot.frames > 4097 || shot.fps < 1 || shot.fps > 120
                || !std::isfinite(shot.conditioningStrength) || shot.conditioningStrength <= 0 || shot.conditioningStrength > 1)
                throw std::invalid_argument("Invalid native video shot parameters");
            validateImage(shot.firstFrame); validateImage(shot.lastFrame);
        }
        if (request.shots.front().continuePrevious)
            throw std::invalid_argument("The first video shot cannot continue a previous shot");
        const std::vector<std::string> backends{"auto", "cpu", "metal", "cuda", "rocm"};
        const std::vector<std::string> precisions{"auto", "float16", "bfloat16", "float32"};
        if (std::find(backends.begin(), backends.end(), request.backend) == backends.end()
            || std::find(precisions.begin(), precisions.end(), request.precision) == precisions.end())
            throw std::invalid_argument("Invalid native video backend or precision");
        if (cancelled) { result.cancelled = true; return result; }
#if IILD_HAS_NATIVE_DIFFUSION
        auto &mutex = native_detail::nativeEngineMutex();
        std::unique_lock lock(mutex, std::defer_lock);
        if (progress) progress({NativeGenerationStage::Waiting});
        while (!lock.try_lock_for(std::chrono::milliseconds(20))) {
            if (cancelled) { result.cancelled = true; return result; }
        }
        if (cancelled) { result.cancelled = true; return result; }
        struct EngineAccess {
            EngineAccess() { native_detail::nativeEngineOnThread(true); }
            ~EngineAccess() { native_detail::nativeEngineOnThread(false); }
        } access;
        native_detail::releaseImageCacheForVideoLocked();
        struct Callbacks {
            const std::atomic_bool &cancelled;
            const NativeProgressCallback &progress;
            native_detail::NativeTelemetry &telemetry;
            std::string error;
            std::mutex mutex;
            std::atomic_int nodes{0}, batches{0};
            ~Callbacks() {
                sd_set_abort_callback(nullptr, nullptr);
                sd_set_progress_stage_callback(nullptr, nullptr);
                sd_set_log_callback(nullptr, nullptr);
                sd_set_backend_eval_callback(nullptr, nullptr);
            }
        } callbacks{cancelled, progress, telemetry, {}, {}};
        sd_set_abort_callback([](void *data) { return static_cast<Callbacks *>(data)->cancelled.load(); }, &callbacks);
        if (request.cpu || request.cpuTextEncoding) {
            sd_set_backend_eval_callback([](ggml_tensor *, bool ask, void *data) {
                auto &state = *static_cast<Callbacks *>(data);
                if (ask) return ++state.nodes % 16 == 0;
                const auto completed = ++state.batches;
                state.telemetry.computing();
                if (state.progress) state.progress({NativeGenerationStage::Computing, completed, 0});
                return !state.cancelled.load();
            }, &callbacks);
        }
        sd_set_progress_stage_callback([](sd_progress_stage_t stage, int step, int total, float, void *data) {
            auto &state = *static_cast<Callbacks *>(data);
            if (stage == SD_PROGRESS_SAMPLE) state.telemetry.stage("denoise", step, total);
            else if (stage == SD_PROGRESS_DECODE) state.telemetry.stage("vae-decode", step, total);
            else state.telemetry.loading(step, total);
            if (state.progress) state.progress({stage == SD_PROGRESS_SAMPLE ? NativeGenerationStage::Denoising
                : stage == SD_PROGRESS_DECODE ? NativeGenerationStage::Decoding : NativeGenerationStage::Loading, step, total});
        }, &callbacks);
        sd_set_log_callback([](sd_log_level_t level, const char *text, void *data) {
            if (!text) return;
            auto &state = *static_cast<Callbacks *>(data);
            state.telemetry.log(text);
            if (level >= SD_LOG_ERROR) {
                const std::lock_guard guard(state.mutex);
                state.error += text;
                if (state.error.size() > 4000) state.error.erase(0, state.error.size() - 4000);
            }
            if (std::getenv("IILD_NATIVE_DIAGNOSTICS")) std::fputs(text, stderr);
        }, &callbacks);
        auto textEncoder = request.textEncoderPath;
        if (textEncoder.empty()) {
            if (request.assetCacheDirectory.empty()) throw std::invalid_argument("Choose a model asset cache or explicit GGUF text encoder");
            std::stop_source stop;
            textEncoder = iild::materializeModelPackageAsset(source, request.assetCacheDirectory, "text_encoder", stop.get_token(),
                [&](const auto &event) {
                    if (cancelled) stop.request_stop();
                    // The existing ABI has signed 32-bit counters. Report MiB,
                    // not bytes: the bundled Gemma encoder exceeds 2 GiB.
                    constexpr std::uint64_t unit = 1024 * 1024;
                    const auto total = std::min<std::uint64_t>((event.totalBytes + unit - 1) / unit,
                        std::numeric_limits<int>::max());
                    const auto completed = event.completedBytes == event.totalBytes ? total
                        : std::min<std::uint64_t>(event.completedBytes / unit, total);
                    telemetry.stage("preparing-model", static_cast<int>(completed), static_cast<int>(total));
                    if (progress) progress({NativeGenerationStage::Preparing,
                        static_cast<int>(completed), static_cast<int>(total)});
                });
        }
        const auto encoderIdentity = native_detail::modelMetadataIdentity(textEncoder);
        if (cancelled) { result.cancelled = true; return result; }
        result.threads = request.threads > 0 ? request.threads : std::max(1, sd_get_num_physical_cores());
#if defined(__APPLE__)
        result.memoryBudgetBytes = native_detail::memoryBudget(native_detail::resourceLimits(), IILD_NATIVE_CPU_VAE);
#else
        result.memoryBudgetBytes = native_detail::memoryBudget({});
#endif
        if (!native_detail::mayAttemptModelLoad(result.memoryBudgetBytes, false))
            throw std::runtime_error("Insufficient available memory for native video generation");
        telemetry.note("video=1 memory_budget_bytes=" + std::to_string(result.memoryBudgetBytes));
        telemetry.stage("model-load");
        if (progress) progress({NativeGenerationStage::Loading});
        const auto model = source.string(), encoder = textEncoder.string();
        const auto budget = std::to_string(double(result.memoryBudgetBytes) / (1024.0 * 1024 * 1024));
        sd_ctx_params_t contextParameters; sd_ctx_params_init(&contextParameters);
        contextParameters.model_path = model.c_str(); contextParameters.llm_path = encoder.c_str();
        contextParameters.n_threads = result.threads; contextParameters.enable_mmap = true;
        contextParameters.flash_attn = contextParameters.diffusion_flash_attn = true;
        contextParameters.eager_load = false; contextParameters.max_vram = budget.c_str();
        if (request.cpu) contextParameters.backend = contextParameters.params_backend = "cpu";
        std::string backend = request.backend == "rocm" ? "HIP" : request.backend;
        if (!request.cpu && backend != "auto") contextParameters.backend = backend.c_str();
        if (!request.cpu && request.cpuTextEncoding) {
            backend = backend == "auto" ? "te=cpu" : "te=cpu,diffusion=" + backend + ",vae=" + backend;
            contextParameters.backend = backend.c_str();
        }
        if (request.precision != "auto") contextParameters.wtype = str_to_sd_type(request.precision == "float16" ? "f16"
            : request.precision == "bfloat16" ? "bf16" : "f32");
        std::unique_ptr<sd_ctx_t, decltype(&free_sd_ctx)> context(new_sd_ctx(&contextParameters), free_sd_ctx);
        if (!context || !sd_ctx_supports_video_generation(context.get()))
            throw std::runtime_error("This checkpoint is not supported by the native video engine. " + callbacks.error);
        NativeReferenceImage previous;
        for (std::size_t index = 0; index < request.shots.size(); ++index) {
            if (cancelled) { result.cancelled = true; return result; }
            const auto &shot = request.shots[index];
            const auto &first = shot.continuePrevious ? previous : shot.firstFrame;
            sd_vid_gen_params_t parameters; sd_vid_gen_params_init(&parameters);
            parameters.prompt = shot.prompt.c_str(); parameters.negative_prompt = shot.negativePrompt.c_str();
            parameters.width = request.width; parameters.height = request.height;
            parameters.video_frames = shot.frames; parameters.fps = shot.fps; parameters.seed = shot.seed;
            parameters.sample_params.sample_steps = request.steps;
            parameters.sample_params.guidance.txt_cfg = request.guidanceScale;
            parameters.strength = shot.conditioningStrength; parameters.vae_tiling_params.enabled = request.vaeTiling;
            const auto image = [](const NativeReferenceImage &value) -> sd_image_t {
                if (value.rgb.empty()) return {};
                return {static_cast<std::uint32_t>(value.width), static_cast<std::uint32_t>(value.height), 3,
                    const_cast<std::uint8_t *>(value.rgb.data())};
            };
            parameters.init_image = image(first); parameters.end_image = image(shot.lastFrame);
            struct Frames {
                sd_image_t *data = nullptr; int count = 0;
                ~Frames() { if (data) free_sd_images(data, count); }
            } frames;
            telemetry.stage("text-encode");
            if (progress) progress({NativeGenerationStage::Encoding});
            const bool generated = generate_video(context.get(), &parameters, &frames.data, &frames.count, nullptr);
            if (cancelled) { result.cancelled = true; return result; }
            if (!generated || !frames.data || frames.count != shot.frames)
                throw std::runtime_error("Native video generation returned incomplete frames. " + callbacks.error);
            for (int position = 0; position < frames.count; ++position) {
                if (cancelled) { result.cancelled = true; return result; }
                const auto &output = frames.data[position];
                if (!output.data || output.width != static_cast<unsigned>(request.width)
                    || output.height != static_cast<unsigned>(request.height) || output.channel != 3)
                    throw std::runtime_error("Native video returned an invalid RGB frame");
                const std::span<const std::uint8_t> rgb(output.data,
                    static_cast<std::size_t>(request.width) * static_cast<std::size_t>(request.height) * 3);
                frame(index, position, request.width, request.height, rgb);
                if (cancelled) { result.cancelled = true; return result; }
                if (position == frames.count - 1) {
                    previous.width = request.width; previous.height = request.height;
                    previous.rgb.assign(rgb.begin(), rgb.end());
                }
            }
            result.frameCounts.push_back(frames.count);
        }
        if (native_detail::modelMetadataIdentity(source) != identity
            || native_detail::modelMetadataIdentity(textEncoder) != encoderIdentity)
            throw std::runtime_error("Video model resources changed during generation");
#else
        (void)progress; (void)frame;
        throw std::runtime_error("This SDK was built without native video inference");
#endif
    } catch (const std::exception &error) {
        result.cancelled = cancelled.load();
        if (!result.cancelled) result.error = error.what();
    }
    return result;
}
}
