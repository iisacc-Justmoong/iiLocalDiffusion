#include "Generation/NativeDiffusion.hpp"
#include "Generation/NativeImageBridge.hpp"
#include "Generation/NativeCachePolicy.hpp"
#include <stable-diffusion.h>
#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <future>
#include <thread>

// The real adapter is compiled into this test. Only the upstream C API is a
// fixture: mimic SDXL alignment, identifiable RGB pixels, and failure cases.
struct sd_ctx_t {};
namespace {
sd_log_cb_t logCallback = nullptr;
void *logData = nullptr;
sd_progress_stage_cb_t progressCallback = nullptr;
void *progressData = nullptr;
sd_abort_cb_t abortCallback = nullptr;
void *abortData = nullptr;
sd_graph_eval_callback_t graphCallback = nullptr;
void *graphData = nullptr;
sd_preview_cb_t previewCallback = nullptr;
void *previewData = nullptr;
enum class Output { valid, failed, engineError, refinementFailed, wrongSize, wrongChannels, missing, multiple };
Output output = Output::valid;
int allocatedImages = 0;
int generationCalls = 0;
int imageBridgeCalls = 0;
unsigned embeddingCount = 0;
unsigned loraCount = 0;
float loraStrength = 0;
std::string negativePrompt;
std::string loraPath;
std::string modelFamily = "sdxl-base";
bool missingVae = false;
bool validExternalVae = true;
int vaeValidationCalls = 0;
bool expectCpu = false;
std::string selectedVae;
constexpr auto warning = "No valid VAE specified with --vae or --force-sdxl-vae-conv-scale flag set, using Conv2D scale 0.031";
void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}
}
#if defined(__APPLE__)
namespace iiLocalDiffusion::native_detail {
ResourceLimits resourceLimits() { return {8ull << 30, 4ull << 30, 6ull << 30, true}; }
}
#endif
extern "C" {
void sd_set_log_callback(sd_log_cb_t callback, void *data) { logCallback = callback; logData = data; }
void sd_set_progress_stage_callback(sd_progress_stage_cb_t callback, void *data) { progressCallback = callback; progressData = data; }
void sd_set_abort_callback(sd_abort_cb_t callback, void *data) { abortCallback = callback; abortData = data; }
void sd_set_backend_eval_callback(sd_graph_eval_callback_t callback, void *data) { graphCallback = callback; graphData = data; }
void sd_set_preview_callback(sd_preview_cb_t callback, preview_t mode, int interval, bool denoised, bool noisy, void *data) {
    if (callback) require(mode == PREVIEW_PROJ && interval == 1 && denoised && !noisy,
        "Live previews must use real denoised latents without extra VAE work");
    previewCallback = callback; previewData = data;
}
int32_t sd_get_num_physical_cores() { return 2; }
void sd_ctx_params_init(sd_ctx_params_t *parameters) { *parameters = {}; parameters->auto_fit = true; }
sd_ctx_t *new_sd_ctx(const sd_ctx_params_t *parameters) {
    if (expectCpu) {
        require(parameters->backend && std::string(parameters->backend) == "cpu",
                "CPU continuation must route every compute module away from the GPU");
        require(parameters->params_backend && std::string(parameters->params_backend) == "cpu",
                "CPU continuation must not reuse GPU parameter storage");
    } else {
#if IILD_NATIVE_CPU_VAE
    require(parameters->backend && std::string(parameters->backend) == "vae=cpu",
            "iOS must place VAE computation on CPU while leaving denoising automatic");
    require(parameters->params_backend
            && std::string(parameters->params_backend) == "te=disk,diffusion=disk,vae=cpu",
            "Explicit iOS placement must keep GPU weights evictable within the memory budget");
#if defined(__APPLE__)
    require(parameters->max_vram && std::stod(parameters->max_vram) == 2.25,
            "iOS must pass the shared-memory headroom budget to the real adapter");
#endif
#else
    require(!parameters->backend, "Desktop must retain automatic module placement");
    require(!parameters->params_backend, "Desktop must retain automatic parameter placement");
#endif
    }
    require(parameters->flash_attn && parameters->diffusion_flash_attn,
            "VAE placement must preserve denoiser attention settings");
    selectedVae = parameters->vae_path ? parameters->vae_path : "";
    embeddingCount = parameters->embedding_count;
    for (unsigned i = 0; i < embeddingCount; ++i)
        require(std::filesystem::is_regular_file(parameters->embeddings[i].path), "Default embedding is not a file");
    if (logCallback) logCallback(SD_LOG_WARN, warning, logData);
    return new sd_ctx_t;
}
void free_sd_ctx(sd_ctx_t *context) { delete context; }
bool sd_ctx_supports_image_generation(const sd_ctx_t *) { return true; }
const char *sd_get_model_family(const sd_ctx_t *) { return modelFamily.c_str(); }
bool sd_model_inspect_vae(const char *, sd_model_vae_info_t *info) {
    *info = {modelFamily.c_str(), modelFamily == "anima" ? "qwen-image"
        : modelFamily == "z-image" ? "flux1" : modelFamily.c_str(), missingVae ? SD_VAE_MISSING : SD_VAE_EMBEDDED};
    return true;
}
bool sd_model_validate_vae(const char *, const char *) { ++vaeValidationCalls; return validExternalVae; }
sample_method_t sd_get_default_sample_method(const sd_ctx_t *) { return static_cast<sample_method_t>(0); }
scheduler_t sd_get_default_scheduler(const sd_ctx_t *, sample_method_t) { return static_cast<scheduler_t>(0); }
void sd_img_gen_params_init(sd_img_gen_params_t *parameters) { *parameters = {}; }
void sd_cancel_generation(sd_ctx_t *, sd_cancel_mode_t) {}
bool convert_with_components(const char *, const char *, const char *, const char *, const char *,
    const char *, const char *, sd_type_t, const char *, bool, int) { return false; }
bool generate_image(sd_ctx_t *, const sd_img_gen_params_t *parameters, sd_image_t **images, int *count) {
    if (parameters->init_image.data) {
        ++imageBridgeCalls;
        require(parameters->init_image.channel == 3 && parameters->init_image.width == parameters->width * 2
            && parameters->init_image.height == parameters->height * 2 && parameters->strength == 0.35f,
            "A unified stage must receive decoded RGB and its own refinement strength");
        require(parameters->init_image.data[3] == 1 && parameters->init_image.data[5] == 1,
            "The unified bridge must carry the preceding image pixels");
    }
    require(parameters->hires.enabled, "Every native image must run Hires fix");
    require(parameters->hires.upscaler == SD_HIRES_UPSCALER_LANCZOS
        && parameters->hires.denoising_strength == 0.35f,
        "Hires fix must decode, upscale with Lanczos, and denoise at strength 0.35");
    require(parameters->hires.target_width == (parameters->width * 2 + 63) / 64 * 64
        && parameters->hires.target_height == (parameters->height * 2 + 63) / 64 * 64,
        "The engine must receive half the requested axes and the aligned final target");
    require(parameters->hires.steps == std::max(1, int(parameters->sample_params.sample_steps * 0.35f)),
        "Even a one-step request must retain actual Hires denoising");
    const auto &tiling = parameters->vae_tiling_params;
    const bool sdxl = modelFamily == "sdxl-base" || modelFamily == "sdxl-refiner";
#if defined(__APPLE__) && !IILD_NATIVE_CPU_VAE
    const int expectedTile = sdxl ? 48 : 32; // The fixture grants a 3.75 GiB budget.
#else
    const int expectedTile = 32; // Mobile fixture grants 2.25 GiB; other families retain 32.
#endif
    require(tiling.tile_size_x == expectedTile && tiling.tile_size_y == expectedTile && tiling.target_overlap == 0.5f,
            "The adapter must apply the memory-bounded VAE policy");
    require(tiling.enabled == (!sdxl || parameters->hires.target_width > expectedTile * 8
        || parameters->hires.target_height > expectedTile * 8),
            "Small SDXL canvases must decode without tile-local normalization differences");
    ++generationCalls;
    loraCount = parameters->lora_count;
    loraStrength = loraCount ? parameters->loras[0].multiplier : 0;
    loraPath = loraCount ? parameters->loras[0].path : "";
    negativePrompt = parameters->negative_prompt ? parameters->negative_prompt : "";
    *images = nullptr; *count = 0;
    if (abortCallback && abortCallback(abortData)) return false;
    if (output == Output::failed) return false;
    if (output == Output::engineError) {
        if (logCallback) logCallback(SD_LOG_ERROR, "VAE decode allocation failed", logData);
        return false;
    }
    if (output == Output::missing) return true;
    if (expectCpu) {
        require(graphCallback, "CPU work must report progress within a slow denoising step");
        for (int node = 0; node < 64; ++node)
            if (graphCallback(nullptr, true, graphData) && !graphCallback(nullptr, false, graphData)) return false;
    }
    if (previewCallback) {
        if (progressCallback) progressCallback(SD_PROGRESS_SAMPLE, 0, parameters->sample_params.sample_steps, 0, progressData);
        std::array<uint8_t, 12> pixels{12, 34, 56, 78, 90, 12, 34, 56, 78, 90, 12, 34};
        sd_image_t frame{2, 2, 3, pixels.data()};
        previewCallback(1, 1, &frame, false, previewData);
        pixels.fill(0); // Engine-owned storage dies after the callback.
    }
    if (progressCallback) progressCallback(SD_PROGRESS_SAMPLE, parameters->sample_params.sample_steps,
                                           parameters->sample_params.sample_steps, 0, progressData);
    if (abortCallback && abortCallback(abortData)) return false;
    if (output == Output::refinementFailed) {
        if (logCallback) logCallback(SD_LOG_ERROR, "Hires refinement failed", logData);
        return false;
    }
    if (previewCallback) {
        if (progressCallback) progressCallback(SD_PROGRESS_SAMPLE, 0, parameters->hires.steps, 0, progressData);
        std::array<uint8_t, 12> pixels{98, 76, 54};
        sd_image_t frame{2, 2, 3, pixels.data()};
        previewCallback(1, 1, &frame, false, previewData);
    }
    if (progressCallback) progressCallback(SD_PROGRESS_SAMPLE, parameters->hires.steps,
                                           parameters->hires.steps, 0, progressData);
    if (abortCallback && abortCallback(abortData)) return false;
    *count = output == Output::multiple ? 2 : 1;
    *images = static_cast<sd_image_t *>(std::calloc(*count, sizeof(sd_image_t)));
    allocatedImages += *count;
    for (int i = 0; i < *count; ++i) {
        auto &image = (*images)[i];
        image.width = parameters->hires.target_width;
        image.height = parameters->hires.target_height;
        if (output == Output::wrongSize) image.height -= 64;
        image.channel = output == Output::wrongChannels ? 4 : 3;
        image.data = static_cast<uint8_t *>(std::calloc(image.width * image.height, image.channel));
        for (unsigned y = 0; y < image.height; ++y)
            for (unsigned x = 0; x < image.width; ++x) {
                const auto offset = (y * image.width + x) * image.channel;
                image.data[offset] = x % 251;
                image.data[offset + 1] = y % 251;
                image.data[offset + 2] = (x + y) % 251;
            }
    }
    if (progressCallback) progressCallback(SD_PROGRESS_DECODE, 1, 1, 0, progressData);
    return true;
}
void free_sd_images(sd_image_t *images, int count) {
    for (int i = 0; i < count; ++i) std::free(images[i].data);
    std::free(images);
    allocatedImages -= count;
}
}

int main(int argc, char **argv) {
    using namespace iiLocalDiffusion;
    try {
        if (argc != 2) return 1;
        const auto directory = std::filesystem::path(argv[1]);
        std::filesystem::create_directories(directory);
        const auto model = directory / "result-fixture.safetensors";
        { std::ofstream file(model); file << "C API fixture"; }
        NativeGenerationRequest request;
        request.modelPath = std::filesystem::canonical(model);
        request.prompt = "portrait";
        {
            const auto path = request.modelPath.string();
            iild_native_request_v1 bridge{};
            bridge.size = sizeof(bridge); bridge.model = path.c_str(); bridge.prompt = "short portrait";
            bridge.negative_prompt = "blur"; bridge.width = 64; bridge.height = 120;
            bridge.steps = 10; bridge.seed = 23; bridge.prepare_only = 1;
            const auto callsBefore = generationCalls;
            auto *prepared = iild_native_generate_v1(&bridge, nullptr, nullptr);
            require(prepared && std::string(iild_native_metadata_v1(prepared)).find("\"error\":\"\"") != std::string::npos,
                "Native bridge preparation failed");
            require(generationCalls == callsBefore, "Foreground preparation sampled an image");
            iild_native_free_v1(prepared);
            bridge.prepare_only = 0;
            for (int i = 0; i < 10; ++i) {
                auto *image = iild_native_generate_v1(&bridge, nullptr, nullptr);
                require(image, "Native bridge returned null");
                const std::string metadata = iild_native_metadata_v1(image);
                size_t size = 0;
                const auto *rgb = iild_native_rgb_v1(image, &size);
                require(metadata.find("\"model_cache_hit\":true") != std::string::npos,
                    "Native bridge did not reuse its prepared context");
                require(rgb && size == 64 * 120 * 3 && negativePrompt == "blur",
                    "Native bridge lost pixels or negative prompt");
                iild_native_free_v1(image);
            }
            require(generationCalls == callsBefore + 10 && allocatedImages == 0, "Native bridge leaked batch results");
            for (const int limit : {0, 1}) {
                bridge.timeout_milliseconds = limit;
                auto *timed = iild_native_generate_v1(&bridge, [](int stage, int, int, void *) {
                    if (stage == static_cast<int>(NativeGenerationStage::Encoding))
                        std::this_thread::sleep_for(std::chrono::milliseconds(25));
                    return 0;
                }, nullptr);
                require(timed, "Native bridge timeout returned null");
                const std::string metadata = iild_native_metadata_v1(timed);
                require(limit ? metadata.find("time limit") != std::string::npos
                              : metadata.find("\"error\":\"\"") != std::string::npos,
                    "Native bridge did not respect the caller's explicit deadline policy");
                iild_native_free_v1(timed);
            }
            bridge.timeout_milliseconds = 0;
            auto *cancelledResult = iild_native_generate_v1(&bridge,
                [](int, int, int, void *) { return 1; }, nullptr);
            require(cancelledResult && std::string(iild_native_metadata_v1(cancelledResult)).find("\"cancelled\":true") != std::string::npos,
                "Native bridge cancellation was lost");
            iild_native_free_v1(cancelledResult);
            bridge.size = 0;
            auto *invalid = iild_native_generate_v1(&bridge, nullptr, nullptr);
            require(invalid && std::string(iild_native_metadata_v1(invalid)).find("ABI version") != std::string::npos,
                "Native bridge accepted an incompatible request layout");
            iild_native_free_v1(invalid);
            iild_native_release_v1();
        }
        request.steps = 10;
        std::atomic_bool cancelled{false};
        {
            std::vector<NativeGenerationPreview> frames;
            const auto result = generateNativeImageWithPreview(request, {}, NativeComputeBackend::Automatic,
                cancelled, {}, [&](const auto &frame) { frames.push_back(frame); });
            require(result.error.empty(), "Preview generation failed: " + result.error);
            require(frames.size() == 2 && frames[0].sequence == 1 && frames[1].sequence == 2,
                "Previews must span base and refinement passes");
            require(frames[0].rgb[0] == 12 && frames[1].rgb[0] == 98 && frames[0].rgb.size() == 12,
                "Preview storage must own the engine pixels");
            require(frames[0].total == 10 && frames[1].total == 3 && frames[1].step == 1,
                "Preview progress must describe its actual pass");
            require(!previewCallback, "Dangling native preview callback");
        }
        // A background CPU request must evict an automatic/GPU context, then
        // reuse only CPU residency until the caller changes backends again.
        releaseNativeDiffusionCache();
        for (int run = 0; run < 4; ++run) {
            expectCpu = run == 1 || run == 2;
            int computed = 0;
            const auto result = generateNativeImageWithBackend(request,
                expectCpu ? NativeComputeBackend::Cpu : NativeComputeBackend::Automatic,
                cancelled, [&](const auto &event) {
                    if (event.stage == NativeGenerationStage::Computing) {
                        require(event.step > computed && event.total == 0, "Invalid CPU work progress");
                        computed = event.step;
                    }
                }, {});
            require(result.error.empty(), "Backend selection failed: " + result.error);
            require(expectCpu ? computed > 0 : computed == 0, "CPU graph progress was lost or invented");
            require(!graphCallback, "Dangling graph progress callback");
            require(result.modelCacheHit == (run == 2), "Backend change reused incompatible residency");
        }
        // width, height, horizontal crop origin, vertical crop origin.
        const std::array<std::array<int, 4>, 6> sizes{{
            {1024, 1368, 0, 20}, {1368, 1024, 20, 0}, {1024, 1824, 0, 16},
            {1824, 1024, 16, 0}, {1024, 1024, 0, 0}, {88, 72, 20, 28}}};
        for (const auto &size : sizes) {
            releaseNativeDiffusionCache();
            request.width = size[0]; request.height = size[1];
            bool decoded = false;
            const auto result = generateNativeImageWithProgress(request, cancelled, [&](const auto &event) {
                decoded |= event.stage == NativeGenerationStage::Decoding;
            });
            require(result.error.empty(), "Completed image rejected: " + result.error);
            require(embeddingCount == 7 && loraCount == 1 && loraStrength == 1,
                    "Legacy native API omitted mandatory default modifiers");
            require(loraPath.find("addDetailAesthetic_v20_32.safetensors") != std::string::npos,
                    "Incorrect fallback LoRA");
            for (const auto *token : {"iild_negative_color_balance", "iild_ndxl", "iild_negative_dynamics",
                    "iild_negative_xl", "iild_negative_hand", "iild_negative_face", "iild_negative_realisticvision"})
                require(negativePrompt.find(token) != std::string::npos, "Missing default negative token");
            require(decoded && !result.cancelled, "Missing decode completion");
            require(result.width == size[0] && result.height == size[1], "Wrong public dimensions");
            require(result.rgb.size() == static_cast<size_t>(size[0] * size[1] * 3), "Wrong RGB byte count");
            for (int y = 0; y < size[1]; ++y)
                for (int x = 0; x < size[0]; ++x) {
                    const auto offset = (y * size[0] + x) * 3;
                    require(result.rgb[offset] == (x + size[2]) % 251
                        && result.rgb[offset + 1] == (y + size[3]) % 251
                        && result.rgb[offset + 2] == (x + size[2] + y + size[3]) % 251,
                        "Crop changed pixels or read the wrong row stride");
                }
            require(allocatedImages == 0, "Upstream image allocation leaked");
            std::cout << "RGB and center crop verified: " << size[0] << 'x' << size[1] << '\n';
        }
        {
            NativeGenerationOptions options;
            options.defaultModifiers = false;
            const auto previous = request.modelPath;
            const auto emptyResources = directory / "society-resources-not-synced";
            std::filesystem::create_directories(emptyResources);
            options.resourceDirectory = emptyResources;
            modelFamily = "anima";
            for (bool needsVae : {false, true}) {
                request.modelPath = directory / (needsVae ? "anima-denoiser.safetensors" : "anima-complete.safetensors");
                { std::ofstream file(request.modelPath); file << "Anima C API fixture"; }
                missingVae = needsVae;
                const auto beforeCalls = generationCalls;
                const auto result = generateNativeImageWithOptions(request, options, cancelled);
                if (needsVae) {
                    require(result.error.find("qwen-image") != std::string::npos
                        && result.error.find("generation resources") != std::string::npos
                        && result.error.find("filesystem error") == std::string::npos
                        && generationCalls == beforeCalls,
                        "A missing Anima VAE must identify the unsynced resources before inference");
                } else require(result.error.empty() && selectedVae.empty(),
                    "Complete Anima must generate with its embedded VAE and no resource manifest");
            }
            request.modelPath = previous;
            options.resourceDirectory.clear();
            for (const auto &vaeFamily : {"qwen-image", "sdxl-base", "flux1", "flux2", "anima", "z-image"}) {
            const auto qwenModel = directory / (std::string(vaeFamily) + "-result-fixture.safetensors");
            { std::ofstream file(qwenModel); file << "Qwen C API fixture"; }
            const auto previous = request.modelPath;
            request.modelPath = std::filesystem::canonical(qwenModel);
            missingVae = true;
            modelFamily = vaeFamily;
            const auto validationsBefore = vaeValidationCalls;
            const auto first = generateNativeImageWithOptions(request, options, cancelled);
            require(first.error.empty() && !selectedVae.empty() && std::filesystem::is_regular_file(selectedVae),
                    "Missing VAE did not select the installed family fallback: " + first.error);
            const auto second = generateNativeImageWithOptions(request, options, cancelled);
            require(second.error.empty() && second.modelCacheHit, "Unchanged fallback must keep the warm native context");
            require(vaeValidationCalls == validationsBefore + 1, "Cache hits must reuse the validated model/VAE pair");
            const auto beforeCalls = generationCalls;
            { std::ofstream file(qwenModel); file << "Changed model metadata requiring VAE revalidation"; }
            validExternalVae = false;
            const auto rejected = generateNativeImageWithOptions(request, options, cancelled);
            require(!rejected.error.empty() && rejected.error.find("tensor contract") != std::string::npos
                && generationCalls == beforeCalls, "An incompatible VAE must fail before inference starts");
            validExternalVae = true;
            request.modelPath = previous;
            }
            {
                const auto previous = request.modelPath;
                const auto sd3Model = directory / "unbundled-sd3.safetensors";
                { std::ofstream file(sd3Model); file << "SD3 fixture with missing VAE"; }
                request.modelPath = std::filesystem::canonical(sd3Model);
                modelFamily = "sd3";
                const auto beforeCalls = generationCalls;
                const auto unavailable = generateNativeImageWithOptions(request, options, cancelled);
                require(unavailable.error.find("Missing fallback VAE for sd3") != std::string::npos
                    && generationCalls == beforeCalls,
                    "A missing SD3 VAE must not be substituted by the same-channel FLUX or Qwen VAE");
                request.modelPath = previous;
            }
            missingVae = false;
            modelFamily = "sdxl-base";
            const auto embedded = generateNativeImageWithOptions(request, options, cancelled);
            require(embedded.error.empty() && selectedVae.empty(), "An embedded or incompatible VAE was replaced");
        }
        {
            NativeGenerationOptions options;
            options.negativePrompt = "blurred text";
            options.loras = {{request.modelPath, 0.35f}};
            auto custom = generateNativeImageWithOptions(request, options, cancelled);
            require(custom.error.empty() && loraCount == 1 && loraPath == request.modelPath.string()
                && loraStrength == 0.35f && negativePrompt.starts_with("blurred text, "), "Explicit modifier precedence failed");
            custom = generateNativeImageWithOptions(request, options, cancelled);
            require(custom.error.empty() && custom.modelCacheHit, "Unchanged defaults did not reuse the context");
            { std::ofstream file(model, std::ios::app); file << "changed fixture"; }
            custom = generateNativeImageWithOptions(request, options, cancelled);
            require(custom.error.empty() && !custom.modelCacheHit, "Changed resource retained a stale context");
            options.defaultModifiers = false;
            options.loras.clear();
            const auto baseline = generateNativeImageWithOptions(request, options, cancelled);
            require(baseline.error.empty() && embeddingCount == 0 && loraCount == 0
                && negativePrompt == "blurred text", "Explicit baseline was not respected");
            modelFamily = "sd15";
            const auto sd15 = generateNativeImageWithProgress(request, cancelled, {});
            require(sd15.error.empty() && loraCount == 0 && negativePrompt.find("iild_negative_hand") != std::string::npos
                && negativePrompt.find("iild_ndxl") == std::string::npos, "Incompatible SDXL defaults reached SD 1.x");
            modelFamily = "sdxl-base";
            options.defaultModifiers = true;
            options.resourceDirectory = directory / "missing-defaults";
            std::filesystem::create_directories(options.resourceDirectory);
            const auto missing = generateNativeImageWithOptions(request, options, cancelled);
            require(!missing.error.empty() && missing.rgb.empty(), "Missing defaults were silently ignored");
        }
        {
            // Model-family dispatch is exercised through the real public adapter;
            // these tiny safetensors headers are C API fixtures, not trained LoRAs.
            const auto resources = directory / "family-defaults";
            std::filesystem::create_directories(resources);
            const auto adapter = resources / "family.safetensors";
            { std::ofstream file(adapter, std::ios::binary);
              const std::uint64_t length = 2;
              file.write(reinterpret_cast<const char *>(&length), sizeof(length)); file << "{}"; }
            NativeGenerationOptions options;
            options.resourceDirectory = resources;
            for (const auto *family : {"sd15", "sd2", "sd3", "flux1", "flux2", "qwen-image", "z-image"}) {
                modelFamily = family;
                { std::ofstream manifest(resources / "generation-defaults.json");
                  manifest << "{\"version\":1,\"negative_embeddings\":[],\"fallback_loras\":["
                      "{\"file\":\"family.safetensors\",\"size\":10,\"scale\":0.7,\"families\":[\"" << family << "\"]}]}"; }
                auto result = generateNativeImageWithOptions(request, options, cancelled);
                require(result.error.empty() && loraCount == 1 && loraStrength == 0.7f
                    && loraPath == adapter.string(), "Native family fallback failed for " + modelFamily + ": " + result.error);
                options.loras = {{request.modelPath, 0.2f}};
                result = generateNativeImageWithOptions(request, options, cancelled);
                require(result.error.empty() && loraCount == 1 && loraStrength == 0.2f,
                    "Explicit LoRA did not replace family fallback");
                options.loras.clear();
            }
            { std::ofstream manifest(resources / "generation-defaults.json");
              manifest << R"({"version":1,"negative_embeddings":[],"fallback_loras":[
                  {"file":"family.safetensors","size":10,"scale":0.7,"families":["flux1","flux1-schnell"]}]})"; }
            const auto duplicate = generateNativeImageWithOptions(request, options, cancelled);
            require(duplicate.error.find("Duplicate fallback LoRA family") != std::string::npos,
                "Ambiguous family fallback was accepted");
            modelFamily = "sdxl-base";
        }
        for (const auto failure : {Output::failed, Output::engineError, Output::refinementFailed, Output::wrongSize,
                                  Output::wrongChannels, Output::missing, Output::multiple}) {
            releaseNativeDiffusionCache();
            output = failure;
            const auto result = generateNativeImageWithProgress(request, cancelled, {});
            require(!result.error.empty() && result.rgb.empty(), "Invalid result accepted");
            require(result.error.find(warning) == std::string::npos, "Warning hid the real failure");
            if (failure == Output::engineError)
                require(result.error.find("VAE decode allocation failed") != std::string::npos, "Lost backend error");
            if (failure == Output::refinementFailed)
                require(result.error.find("Hires refinement failed") != std::string::npos, "Lost Hires failure");
            if (failure == Output::wrongSize)
                require(result.error.find("128x128") != std::string::npos
                    && result.error.find("128x64") != std::string::npos, "Missing expected/actual dimensions");
            require(allocatedImages == 0, "Failed image allocation leaked");
            require(!logCallback && !progressCallback, "Dangling callback state");
        }
        releaseNativeDiffusionCache();
        output = Output::valid;
        {
            int passes = 0;
            const auto result = generateNativeImageWithProgress(request, cancelled, [&](const auto &event) {
                if (event.stage == NativeGenerationStage::Denoising && ++passes == 2) cancelled = true;
            });
            require(passes == 2 && result.cancelled && result.rgb.empty(),
                "Cancellation during refinement returned an earlier image as success");
            require(allocatedImages == 0 && !abortCallback, "Hires cancellation leaked backend state");
            cancelled = false;
        }
        request.width = request.height = 64;
        {
            const auto original = request.modelPath;
            const auto package = std::filesystem::canonical(directory) / "cascade.iildmodel";
            std::filesystem::create_directories(package);
            for (const auto *name : {"a.safetensors", "b.safetensors"}) {
                std::ofstream file(package / name); file << "C API fixture";
            }
            const auto writeManifest = [&](const std::string &second, double strength = 0.35) {
                std::ofstream file(package / "model_index.json");
                file << R"({"schema":"iild-unified-model-v1","_class_name":"IILDUnifiedCascade","composition":"ordered-image-refinement","stages":[)"
                     << R"({"model":"a.safetensors","size_bytes":13,"strength":1},)"
                     << "{\"model\":\"" << second << "\",\"size_bytes\":13,\"strength\":" << strength << "}]}";
            };
            writeManifest("b.safetensors");
            request.modelPath = package;
            NativeGenerationOptions options; options.defaultModifiers = false;
            const auto before = generationCalls;
            auto cascade = generateNativeImageWithOptions(request, options, cancelled);
            require(cascade.error.empty() && cascade.rgb.size() == 64 * 64 * 3
                && generationCalls == before + 2 && imageBridgeCalls == 1,
                "Unified cascade did not execute both independent model contexts: " + cascade.error);
            writeManifest("b.safetensors", 0);
            cascade = generateNativeImageWithOptions(request, options, cancelled);
            require(cascade.error.empty() && generationCalls == before + 3,
                "A zero-strength member must not regenerate the image");
            writeManifest("../result-fixture.safetensors");
            cascade = generateNativeImageWithOptions(request, options, cancelled);
            require(!cascade.error.empty() && cascade.rgb.empty() && generationCalls == before + 3,
                "A redirected member must fail before executing any model");
            writeManifest("b.safetensors");
            int loads = 0;
            cascade = generateNativeImageWithOptions(request, options, cancelled, [&](const auto &event) {
                if (event.stage == NativeGenerationStage::Encoding && ++loads == 2) cancelled = true;
            });
            require(cascade.cancelled && cascade.rgb.empty(), "A cancelled cascade published an earlier stage as success");
            cancelled = false;
            request.modelPath = original;
        }
        request.timeoutMilliseconds = 200;
        for (const bool cancelPaused : {false, true}) {
            cancelled = false;
            auto control = std::make_shared<NativeExecutionControl>();
            auto future = std::async(std::launch::async, [&] {
                return generateNativeImageWithExecutionControl(request, cancelled, [&](const auto &event) {
                    if (event.stage == NativeGenerationStage::Encoding) control->setPaused(true);
                }, control);
            });
            const auto waitDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            while (!control->isWaiting() && std::chrono::steady_clock::now() < waitDeadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            const bool parked = control->isWaiting();
            // Stay paused longer than the entire inference timeout. The same
            // request must finish after resuming, without an expired deadline.
            const bool blocked = future.wait_for(std::chrono::milliseconds(300)) == std::future_status::timeout;
            if (cancelPaused) cancelled = true;
            else control->setPaused(false);
            if (!parked || !blocked) { cancelled = true; control->setPaused(false); }
            const auto result = future.get();
            require(parked && blocked, "Paused engine submitted further compute");
            if (cancelPaused) require(result.cancelled && result.rgb.empty(), "Paused cancellation did not unwind");
            else require(result.error.empty() && !result.rgb.empty(), "Pause consumed the inference deadline: " + result.error);
            require(allocatedImages == 0 && !abortCallback, "Paused request leaked engine state");
        }
        std::cout << "Paused native requests resume without timeout and cancel while parked\n";
        releaseNativeDiffusionCache();
        std::filesystem::remove(model);
        std::cout << "Malformed images rejected; warning/error separation and cleanup verified\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
