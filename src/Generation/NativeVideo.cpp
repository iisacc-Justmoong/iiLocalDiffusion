#include "NativeVideo.hpp"
#include "NativeCachePolicy.hpp"
#include "NativeModelPaths.hpp"
#include "NativeEngineAccess.hpp"
#include "NativeTelemetry.hpp"
#include "ModelPackaging/ModelInspection.hpp"
#include "ModelPackaging/ModelPackaging.hpp"
#include "ModelPackaging/ModelGGUFInspection.hpp"
#include "ModelPackaging/ModelWanNames.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <limits>
#include <mutex>
#include <optional>
#include <stop_token>
#include <thread>
#include <utility>
#if IILD_HAS_NATIVE_DIFFUSION
#include <stable-diffusion.h>
#endif

namespace iiLocalDiffusion {
static NativeVideoResult generateNativeVideoImpl(const NativeVideoRequest &request, const NativeVideoComponents &components,
    const std::atomic_bool &cancelled,
    const NativeProgressCallback &progress, const NativeVideoFrameObserver &frame, bool lowVram)
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
        if (!native_detail::isCanonicalOrShardIndex(source))
            throw std::invalid_argument("Choose a local video checkpoint or a validated shard index");
        const auto isOnnx = [](const std::filesystem::path &path) {
            auto ext = path.extension().u8string();
            for (auto &c : ext) if (c >= u8'A' && c <= u8'Z') c += u8'a' - u8'A';
            return ext == u8".onnx";
        };
        if (isOnnx(source) || (!request.textEncoderPath.empty() && isOnnx(request.textEncoderPath))
            || (!components.vaePath.empty() && isOnnx(components.vaePath))
            || (!components.highNoiseModelPath.empty() && isOnnx(components.highNoiseModelPath))
            || (!components.visionEncoderPath.empty() && isOnnx(components.visionEncoderPath)))
            throw std::invalid_argument("ONNX video checkpoints and components are not supported");
        const auto inspectIndex = [](const std::filesystem::path &path) -> std::optional<iild::SafetensorsIndex> {
            const auto container=iild::detectModelContainer(path);
            if (container!="safetensors-index" && !iild::hasSafetensorsIndexSuffix(path.filename().string())) return std::nullopt;
            if (container=="safetensors" || container=="gguf") return std::nullopt;
            return iild::inspectSafetensorsIndex(path);
        };
        const auto modelIndex=inspectIndex(source);
        const auto identity = native_detail::modelMetadataIdentity(source);
        const auto sourceFormat=iild::detectModelContainer(source);
        const auto embeddedMain=sourceFormat=="safetensors" ? iild::detail::inspectEmbeddedMainGguf(source)
                                                           : std::optional<iild::detail::EmbeddedGgufMain>{};
        // Output modality overrides do not change the underlying model family.
        // Unknown architectures retain the existing LLM behavior and must still
        // pass the native engine's complete component validation.
        const bool wan=iild::inspectModel(source).family=="Wan";
        const auto packageAssets=(sourceFormat=="safetensors" && (wan || embeddedMain))
            ? iild::detail::inspectIncludedPackageAssets(source) : iild::detail::PackageAssets{};
        const auto &packageRoles=packageAssets.roles;
        // Bare and Diffusers-style Wan weights need a diffusion mount. Keep the
        // original mount as well for an embedded encoder or VAE.
        constexpr std::array<std::string_view,2> bareWanNames{"patch_embedding.weight","blocks.0.cross_attn.norm_k.weight"};
        bool mountWanDiffusion=false;
        if (wan) {
            std::array<bool,bareWanNames.size()> foundBare{};
            iild::detail::WanDiffusersSignature diffusersSignature;
            const auto inspectName = [&](std::string_view name) {
                for (std::size_t i=0;i<bareWanNames.size();++i) if (name==bareWanNames[i]) foundBare[i]=true;
                diffusersSignature.add(name);
            };
            if (embeddedMain) {
                for (const auto &name:embeddedMain->catalog.tensorNames) inspectName(name);
            } else if (modelIndex) {
                for (const auto &tensor:modelIndex->tensors) inspectName(tensor.name);
            } else if (sourceFormat=="safetensors" || sourceFormat=="gguf") {
                std::vector<std::string> ownedNames;
                ownedNames.reserve(bareWanNames.size()+diffusersSignature.prefixes.size()*diffusersSignature.markers.size());
                for (auto name:bareWanNames) ownedNames.emplace_back(name);
                for (auto prefix:diffusersSignature.prefixes) for (auto marker:diffusersSignature.markers)
                    ownedNames.push_back(std::string(prefix)+std::string(marker));
                std::vector<std::string_view> candidates; candidates.reserve(ownedNames.size());
                for (const auto &name:ownedNames) candidates.push_back(name);
                for (const auto &name:iild::probeModelTensorNames(source,candidates).matches) inspectName(name);
            }
            mountWanDiffusion=std::all_of(foundBare.begin(),foundBare.end(),[](bool found) {return found;})
                || diffusersSignature.prefix().has_value();
        }
        const auto validOptionalFloat=[](float value,float maximum) {
            return std::isfinite(value) && (value == -1 || (value >= 0 && value <= maximum));
        };
        if (!validOptionalFloat(components.moeBoundary,1) || !validOptionalFloat(components.highNoiseGuidance,100)
            || (components.highNoiseSteps != -1 && (components.highNoiseSteps < 1 || components.highNoiseSteps > 1000)))
            throw std::invalid_argument("Invalid high-noise video sampling controls");
        const float moeBoundary=components.moeBoundary != -1 ? components.moeBoundary
            : wan && packageAssets.wanMoeBoundary ? *packageAssets.wanMoeBoundary : -1;
        const bool expertControls=moeBoundary != -1 || components.highNoiseSteps != -1 || components.highNoiseGuidance != -1;
        if (expertControls && components.highNoiseModelPath.empty() &&
            !(wan && packageRoles.contains("high_noise_model")))
            throw std::invalid_argument("High-noise sampling controls require a high-noise model");
        if (!wan && (!components.highNoiseModelPath.empty() || !components.visionEncoderPath.empty()))
            throw std::invalid_argument("High-noise and vision video components require a recognized Wan checkpoint");
        constexpr std::array<std::string_view,12> embeddedEncoderNames{
            "text_encoders.llm.model.embed_tokens.weight",
            "text_encoders.llm.model.norm.weight",
            "text_encoders.llm.model.layers.0.self_attn.q_proj.weight",
            "text_encoders.llm.token_embd.weight",
            "text_encoders.llm.output_norm.weight",
            "text_encoders.llm.blk.0.attn_q.weight",
            "text_encoders.t5xxl.transformer.shared.weight",
            "text_encoders.t5xxl.transformer.encoder.final_layer_norm.weight",
            "text_encoders.t5xxl.transformer.encoder.block.0.layer.0.SelfAttention.q.weight",
            "text_encoders.t5xxl.transformer.token_embd.weight",
            "text_encoders.t5xxl.transformer.enc.output_norm.weight",
            "text_encoders.t5xxl.transformer.enc.blk.0.attn_q.weight"};
        bool embeddedEncoder=false;
        if (request.textEncoderPath.empty()) {
            std::vector<std::string> found;
            bool embeddedPackage=false;
            if (embeddedMain) {
                for (const auto &name:embeddedMain->catalog.tensorNames)
                    if (std::find(embeddedEncoderNames.begin(),embeddedEncoderNames.end(),name)!=embeddedEncoderNames.end())
                        found.push_back(name);
                embeddedPackage=packageRoles.contains("text_encoder");
            } else if (modelIndex) {
                for (const auto &tensor:modelIndex->tensors)
                    if (std::find(embeddedEncoderNames.begin(),embeddedEncoderNames.end(),tensor.name)!=embeddedEncoderNames.end())
                        found.push_back(tensor.name);
            } else {
                const auto format=iild::detectModelContainer(source);
                if (format=="safetensors" || format=="gguf") {
                    auto probe=iild::probeModelTensorNames(source,embeddedEncoderNames);
                    found=std::move(probe.matches);embeddedPackage=probe.embeddedPackage;
                }
            }
            const auto has=[&](std::string_view name) {
                return std::find(found.begin(),found.end(),name)!=found.end();
            };
            const std::size_t offset=wan?6:0;
            embeddedEncoder=!embeddedPackage && (has(embeddedEncoderNames[offset]) || has(embeddedEncoderNames[offset+3]))
                && (has(embeddedEncoderNames[offset+1]) || has(embeddedEncoderNames[offset+4]))
                && (has(embeddedEncoderNames[offset+2]) || has(embeddedEncoderNames[offset+5]));
        }
        if (native_detail::modelMetadataIdentity(source)!=identity)
            throw std::runtime_error("Video model changed during component inspection");
        if (modelIndex && request.textEncoderPath.empty() && !embeddedEncoder)
            throw std::invalid_argument("A standalone indexed video model requires an explicit compatible text encoder");
        const auto totalSize = [](const std::filesystem::path &path, const std::optional<iild::SafetensorsIndex> &index) {
            auto size=std::filesystem::file_size(path);
            if (index) for (const auto &shard : index->shards) {
                if (shard.sizeBytes>std::numeric_limits<std::uintmax_t>::max()-size)
                    throw std::runtime_error("Video model shard sizes exceed the supported range");
                size+=shard.sizeBytes;
            }
            return size;
        };
        const auto modelSize=totalSize(source,modelIndex);
        struct ComponentSnapshot {
            std::filesystem::path path;
            std::string identity;
            std::optional<iild::SafetensorsIndex> index;
            std::uintmax_t bytes=0;
        };
        const auto snapshot = [&](const std::filesystem::path &path) {
            ComponentSnapshot value;
            if (path.empty()) return value;
            value.path=std::filesystem::absolute(path).lexically_normal();
            if (!native_detail::isCanonicalOrShardIndex(value.path))
                throw std::invalid_argument("Choose local video component files or validated shard indexes");
            value.index=inspectIndex(value.path);
            value.identity=native_detail::modelMetadataIdentity(value.path);
            value.bytes=totalSize(value.path,value.index);
            return value;
        };
        std::array componentFiles{snapshot(components.vaePath),snapshot(components.highNoiseModelPath),
                                   snapshot(components.visionEncoderPath),ComponentSnapshot{}};
        const auto verifyComponents = [&] {
            return std::all_of(componentFiles.begin(),componentFiles.end(),[](const ComponentSnapshot &file) {
                return file.path.empty() || (native_detail::modelMetadataIdentity(file.path)==file.identity
                    && (!file.index || iild::verifySafetensorsIndex(*file.index)));
            });
        };
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
            std::string error, warnings;
            std::mutex mutex;
            std::atomic_int nodes{0}, batches{0};
            std::string failureContext() {
                const std::lock_guard guard(mutex);
                std::string context;
                if (!warnings.empty()) context += "Engine warnings:\n" + warnings + "\n";
                if (!error.empty()) context += "Engine errors:\n" + error;
                return context;
            }
            void clearFailureContext() {
                const std::lock_guard guard(mutex);
                error.clear(); warnings.clear();
            }
            ~Callbacks() {
                sd_set_abort_callback(nullptr, nullptr);
                sd_set_progress_stage_callback(nullptr, nullptr);
                sd_set_log_callback(nullptr, nullptr);
                sd_set_backend_eval_callback(nullptr, nullptr);
            }
        } callbacks{cancelled, progress, telemetry, {}, {}, {}};
        sd_set_abort_callback([](void *data) { return static_cast<Callbacks *>(data)->cancelled.load(); }, &callbacks);
        const auto enableCpuProgress = [&] {
            sd_set_backend_eval_callback([](ggml_tensor *, bool ask, void *data) {
                auto &state = *static_cast<Callbacks *>(data);
                if (ask) return ++state.nodes % 16 == 0;
                const auto completed = ++state.batches;
                state.telemetry.computing();
                if (state.progress) state.progress({NativeGenerationStage::Computing, completed, 0});
                return !state.cancelled.load();
            }, &callbacks);
        };
        if (request.cpu || request.cpuTextEncoding || lowVram) enableCpuProgress();
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
            state.telemetry.log(text, level >= SD_LOG_WARN);
            if (level >= SD_LOG_WARN) {
                const std::lock_guard guard(state.mutex);
                auto &details = level >= SD_LOG_ERROR ? state.error : state.warnings;
                const std::size_t limit = level >= SD_LOG_ERROR ? 2400 : 1200;
                details += text;
                if (details.size() > limit) details.erase(0, details.size() - limit);
            }
            if (std::getenv("IILD_NATIVE_DIAGNOSTICS")) std::fputs(text, stderr);
        }, &callbacks);
        const auto materialize = [&](const std::string &role,const std::filesystem::path &relative=std::filesystem::path{}) {
            if (request.assetCacheDirectory.empty()) throw std::invalid_argument("Choose a model asset cache or provide the required video component explicitly");
            std::stop_source stop;
            const auto observer = [&](const auto &event) {
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
                };
            return relative.empty() ? iild::materializeModelPackageAsset(source,request.assetCacheDirectory,role,stop.get_token(),observer)
                                    : iild::materializeModelPackageAssetByPath(source,request.assetCacheDirectory,relative,stop.get_token(),observer);
        };
        if (embeddedMain) componentFiles[3]=snapshot(materialize("model",embeddedMain->relativePath));
        auto textEncoder = request.textEncoderPath;
        if (textEncoder.empty() && !embeddedEncoder) textEncoder=materialize("text_encoder");
        // A non-LTX package keeps its separate VAE under a component namespace.
        // Only an included Wan VAE is mounted; duplicates already in the main
        // model stay in place, and ambiguous roles are rejected by materialize.
        if (wan && components.vaePath.empty() && packageRoles.contains("video_vae"))
            componentFiles[0]=snapshot(materialize("video_vae"));
        if (wan && components.highNoiseModelPath.empty() && packageRoles.contains("high_noise_model"))
            componentFiles[1]=snapshot(materialize("high_noise_model"));
        if (wan && components.visionEncoderPath.empty() && packageRoles.contains("vision_encoder"))
            componentFiles[2]=snapshot(materialize("vision_encoder"));
        const auto encoderIndex=textEncoder.empty() ? std::optional<iild::SafetensorsIndex>{} : inspectIndex(textEncoder);
        const auto encoderSize=textEncoder.empty() ? std::uintmax_t{0} : totalSize(textEncoder,encoderIndex);
        const auto encoderIdentity=textEncoder.empty() ? std::string{} : native_detail::modelMetadataIdentity(textEncoder);
        if (native_detail::modelMetadataIdentity(source)!=identity || (modelIndex && !iild::verifySafetensorsIndex(*modelIndex))
            || (encoderIndex && !iild::verifySafetensorsIndex(*encoderIndex)) || !verifyComponents())
            throw std::runtime_error("Video model shards changed during preparation");
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
        const auto model = (componentFiles[3].path.empty() ? source : componentFiles[3].path).string(), encoder = textEncoder.string();
        const auto vae=componentFiles[0].path.string(), high=componentFiles[1].path.string(), vision=componentFiles[2].path.string();
        const auto budget = std::to_string(double(result.memoryBudgetBytes) / (1024.0 * 1024 * 1024));
        const bool streamModel = native_detail::needsDiskBackedStreaming(
            modelSize, result.memoryBudgetBytes) ||
            native_detail::needsDiskBackedStreaming(encoderSize, result.memoryBudgetBytes) ||
            std::any_of(componentFiles.begin(),componentFiles.end(),[&](const auto &file) {
                return native_detail::needsDiskBackedStreaming(file.bytes,result.memoryBudgetBytes);
            });
        sd_ctx_params_t contextParameters; sd_ctx_params_init(&contextParameters);
        contextParameters.model_path = model.c_str();
        if (mountWanDiffusion) contextParameters.diffusion_model_path=model.c_str();
        if (wan) contextParameters.t5xxl_path = encoder.empty() ? nullptr : encoder.c_str();
        else contextParameters.llm_path = encoder.empty() ? nullptr : encoder.c_str();
        contextParameters.vae_path=vae.empty() ? nullptr : vae.c_str();
        contextParameters.high_noise_diffusion_model_path=high.empty() ? nullptr : high.c_str();
        contextParameters.clip_vision_path=vision.empty() ? nullptr : vision.c_str();
        contextParameters.n_threads = result.threads; contextParameters.enable_mmap = true;
        contextParameters.flash_attn = contextParameters.diffusion_flash_attn = true;
        contextParameters.eager_load = false; contextParameters.max_vram = budget.c_str();
        if (request.cpu) contextParameters.backend = contextParameters.params_backend = "cpu";
        std::string backend = request.backend == "rocm" ? "HIP" : request.backend;
        const bool explicitGpuPlacement = !request.cpu && backend != "cpu" && (request.cpuTextEncoding || backend != "auto");
        if (!request.cpu && backend != "auto") contextParameters.backend = backend.c_str();
        if (!request.cpu && (request.cpuTextEncoding || lowVram)) {
            backend = backend == "auto" ? (lowVram ? "te=cpu,vae=cpu" : "te=cpu")
                : "te=cpu,diffusion=" + backend + ",vae=" + (lowVram ? std::string("cpu") : backend);
            contextParameters.backend = backend.c_str();
        }
        // Any explicit placement disables the engine's auto-fit. Keep video
        // diffusion weights reloadable so unused segments can be evicted within
        // the live GPU budget rather than accumulating until allocation fails.
        if (explicitGpuPlacement) contextParameters.params_backend = "diffusion=disk";
        // Large packages may include a separate text encoder. Keep both major
        // components reloadable while preserving the requested compute backend.
        // Keep the engine's auto-fit and VAE retry behavior for an entirely
        // automatic request; it already chooses disk when residency will not
        // fit. Explicit placement disables auto-fit and needs this policy.
        if (streamModel && (request.cpu || request.cpuTextEncoding || request.backend != "auto")) {
            contextParameters.params_backend = "te=disk,diffusion=disk";
            telemetry.note("video model exceeds memory budget; using disk-backed segmented weights");
        }
        if (lowVram) {
            contextParameters.params_backend = "te=disk,diffusion=disk,vae=disk";
            contextParameters.disable_prefetch = true;
            telemetry.note("video low-vram: CPU text/VAE, disk-backed weights, bounded segment residency");
        }
        if (request.precision != "auto") contextParameters.wtype = str_to_sd_type(request.precision == "float16" ? "f16"
            : request.precision == "bfloat16" ? "bf16" : "f32");
        std::unique_ptr<sd_ctx_t, decltype(&free_sd_ctx)> context(new_sd_ctx(&contextParameters), free_sd_ctx);
        if (!context || !sd_ctx_supports_video_generation(context.get()))
            throw std::runtime_error("This checkpoint is not supported by the native video engine. " + callbacks.failureContext());
        NativeReferenceImage previous;
        bool cpuDiskFallback = false;
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
            if (moeBoundary != -1) parameters.moe_boundary=moeBoundary;
            if (components.highNoiseSteps != -1) parameters.high_noise_sample_params.sample_steps=components.highNoiseSteps;
            if (components.highNoiseGuidance != -1) parameters.high_noise_sample_params.guidance.txt_cfg=components.highNoiseGuidance;
            parameters.strength = shot.conditioningStrength; parameters.vae_tiling_params.enabled = request.vaeTiling || lowVram;
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
            callbacks.clearFailureContext();
            bool generated = generate_video(context.get(), &parameters, &frames.data, &frames.count, nullptr);
            if (cancelled) { result.cancelled = true; return result; }
            const auto firstFailure = callbacks.failureContext();
            if ((!generated || !frames.data || frames.count != shot.frames) &&
                !cpuDiskFallback && !request.cpu && request.backend == "auto" &&
                firstFailure.find("model manager cannot make enough memory available on") != std::string::npos) {
                // GPU graph workspace cannot be swapped to disk. Release the GPU
                // context and let the OS page CPU compute memory while the model
                // manager reloads diffusion weights from their original mapping.
                if (frames.data) free_sd_images(frames.data, frames.count);
                frames.data = nullptr; frames.count = 0;
                context.reset();
                if (cancelled) { result.cancelled = true; return result; }
                contextParameters.backend = "cpu";
                contextParameters.params_backend = lowVram ? "te=disk,diffusion=disk,vae=disk" : "te=disk,diffusion=disk";
                enableCpuProgress();
                telemetry.note("video fallback: CPU compute with disk-backed text and diffusion weights after GPU budget failure");
                if (progress) progress({NativeGenerationStage::Waiting});
                if (cancelled) { result.cancelled = true; return result; }
                callbacks.clearFailureContext();
                context.reset(new_sd_ctx(&contextParameters));
                cpuDiskFallback = true;
                if (!context || !sd_ctx_supports_video_generation(context.get()))
                    throw std::runtime_error("CPU disk-backed video fallback could not initialize. " + firstFailure + callbacks.failureContext());
                generated = generate_video(context.get(), &parameters, &frames.data, &frames.count, nullptr);
                if (cancelled) { result.cancelled = true; return result; }
                if (!generated || !frames.data || frames.count != shot.frames)
                    throw std::runtime_error("CPU disk-backed video fallback failed. Initial GPU failure:\n" +
                        firstFailure + "Fallback failure:\n" + callbacks.failureContext());
            }
            if (!generated || !frames.data || frames.count != shot.frames)
                throw std::runtime_error("Native video generation returned incomplete frames. " + callbacks.failureContext());
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
            || (!textEncoder.empty() && native_detail::modelMetadataIdentity(textEncoder) != encoderIdentity)
            || (modelIndex && !iild::verifySafetensorsIndex(*modelIndex))
            || (encoderIndex && !iild::verifySafetensorsIndex(*encoderIndex)) || !verifyComponents())
            throw std::runtime_error("Video model resources changed during generation");
#else
        (void)progress; (void)frame; (void)lowVram;
        throw std::runtime_error("This SDK was built without native video inference");
#endif
    } catch (const std::exception &error) {
        result.cancelled = cancelled.load();
        if (!result.cancelled) result.error = error.what();
    }
    return result;
}
NativeVideoResult generateNativeVideo(const NativeVideoRequest &request, const std::atomic_bool &cancelled,
    const NativeProgressCallback &progress, const NativeVideoFrameObserver &frame) {
    return generateNativeVideoImpl(request, {}, cancelled, progress, frame, false);
}
NativeVideoResult generateNativeVideoLowVram(const NativeVideoRequest &request, const std::atomic_bool &cancelled,
    const NativeProgressCallback &progress, const NativeVideoFrameObserver &frame) {
    return generateNativeVideoImpl(request, {}, cancelled, progress, frame, true);
}
NativeVideoResult generateNativeVideoWithComponents(const NativeVideoRequest &request, const NativeVideoComponents &components,
    const std::atomic_bool &cancelled, const NativeProgressCallback &progress, const NativeVideoFrameObserver &frame) {
    return generateNativeVideoImpl(request, components, cancelled, progress, frame, false);
}
NativeVideoResult generateNativeVideoLowVramWithComponents(const NativeVideoRequest &request, const NativeVideoComponents &components,
    const std::atomic_bool &cancelled, const NativeProgressCallback &progress, const NativeVideoFrameObserver &frame) {
    return generateNativeVideoImpl(request, components, cancelled, progress, frame, true);
}
}
