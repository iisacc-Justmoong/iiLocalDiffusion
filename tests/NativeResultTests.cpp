#include "Generation/NativeDiffusion.hpp"
#include "Generation/NativePose.hpp"
#include "Generation/NativeImageBridge.hpp"
#include "Generation/NativeCachePolicy.hpp"
#include <stable-diffusion.h>
#include <zip.h>
#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <future>
#include <thread>
#include <map>

// The real adapter is compiled into this test. The upstream C API and Pose
// boundary are fixtures; NativePoseTests separately executes the real ONNX path.
namespace iiLocalDiffusion {
bool nativePoseAvailable() noexcept { return true; }
void releaseNativePoseCache() noexcept {}
NativePoseResult processNativePose(const std::filesystem::path &detector, const std::filesystem::path &pose,
    const NativeReferenceImage &image, const std::atomic_bool &, unsigned threads,
    const std::shared_ptr<NativeExecutionControl> &)
{
    if (detector.filename() != "detector.onnx" || pose.filename() != "pose.onnx") throw std::runtime_error("Wrong Pose resources");
    auto hint = image; std::fill(hint.rgb.begin(),hint.rgb.end(),67);
    return {std::move(hint),false,128,threads};
}
}
// The C API
// fixture: mimic SDXL alignment, identifiable RGB pixels, and failure cases.
struct sd_ctx_t { bool refiner = false; std::vector<std::string> multiPaths; unsigned ipCount = 0; };
struct adetailer_ctx_t {};
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
int runtimeReleaseCalls = 0;
int imageBridgeCalls = 0;
unsigned embeddingCount = 0;
std::map<std::string, std::string> registeredEmbeddings;
std::vector<std::string> loadedEmbeddings;
std::string actualPrompt;
bool actualPromptWeighting = true, embeddingLoadSucceeds = true;
bool actualFreeU = false, freeUAvailable = true;
bool upscalerAvailable = true;
std::string preparedUpscaler;
int upscalerPrepareCalls = 0;
int detailerLoads = 0, detailerCalls = 0, liveDetailers = 0;
bool detailerLoadSucceeds = true, detailerRunSucceeds = true, detailerExpected = false;
std::atomic_bool *detailerCancellation = nullptr;
std::string refinerFixturePath, refinerFamily = "sdxl-refiner", refinerNegative;
int refinerLoads = 0, refinerCalls = 0, liveRefiners = 0;
float actualRefinerSwitch = -1;
bool refinerLoadSucceeds = true, refinerRunSucceeds = true, refinerCompatible = true;
bool refinerPromptWeighting = true, refinerFreeU = false, refinerEmbeddingSucceeds = true;
std::map<std::string, std::string> refinerEmbeddings;
std::atomic_bool *refinerCancellation = nullptr;
bool replaceRefinerDuringGeneration = false;
std::string selectedControlNet;
bool controlNetAvailable = true, cannySucceeds = true;
int cannyCalls = 0, controlMarker = -1;
float controlWeight = -1;
std::vector<uint8_t> selectedControlMask;
bool controlMaskSucceeds = true;
int multiControlLoads = 0;
bool multiControlLoadSucceeds = true, multiControlInputsSucceed = true;
std::vector<std::string> preparedMultiPaths;
std::vector<int> multiControlMarkers;
std::vector<float> multiControlStrengths;
std::vector<std::vector<uint8_t>> multiControlMasks;
int ipLoads = 0;
bool ipLoadSucceeds = true, ipInputsSucceed = true;
std::vector<std::pair<std::string, std::string>> ipPaths;
std::vector<int> ipMarkers;
std::vector<int> ipAtGeneration;
std::vector<float> ipStrengths;
std::vector<std::vector<uint8_t>> ipMasks;
std::filesystem::path replaceIPDuringGeneration;
unsigned loraCount = 0;
float loraStrength = 0;
std::string negativePrompt;
std::string loraPath;
std::string modelFamily = "sdxl-base";
bool missingVae = false;
bool validExternalVae = true;
int vaeValidationCalls = 0;
bool expectCpu = false;
bool expectResidentModel = false;
std::string selectedVae;
std::string selectedClipL, selectedClipG, selectedT5, selectedLlm;
bool expectHires = true;
float actualCfg = 0, actualDistilled = 0, actualFlowShift = 0;
sample_method_t actualSampler{};
scheduler_t actualScheduler{};
int actualClipSkip = 0;
bool actualSeamless = false;
int referenceCapacity = 0, expectedReferenceCount = -1;
float actualImageStrength = -1;
std::vector<int> actualReferenceMarkers;
float actualEta = 0, expectedDenoise = 0.35f;
sd_hires_upscaler_t expectedUpscaler = SD_HIRES_UPSCALER_LANCZOS;
std::vector<float> actualSigmas;
prediction_t selectedPrediction = PREDICTION_COUNT;
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
    require(parameters->memory_resident_model == expectResidentModel,
            "Resident model mode was not propagated to the native backend");
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
            && std::string(parameters->params_backend) == (expectResidentModel ? "vae=cpu" : "te=disk,diffusion=disk,vae=cpu"),
            "Resident iOS mode must not place model parameters in a disk backend");
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
    if (!refinerFixturePath.empty() && parameters->model_path == refinerFixturePath) {
        require(!parameters->clip_l_path && !parameters->clip_g_path && !parameters->t5xxl_path
            && !parameters->llm_path && !parameters->control_net_path
            && parameters->prediction == PREDICTION_COUNT,
            "Refiner inherited Base-specific components or prediction override");
        ++refinerLoads;
        if (!refinerLoadSucceeds) return nullptr;
        refinerEmbeddings.clear();
        for (unsigned i = 0; i < parameters->embedding_count; ++i)
            refinerEmbeddings[parameters->embeddings[i].name] = parameters->embeddings[i].path;
        ++liveRefiners;
        return new sd_ctx_t{true};
    }
    selectedVae = parameters->vae_path ? parameters->vae_path : "";
    selectedControlNet = parameters->control_net_path ? parameters->control_net_path : "";
    selectedClipL = parameters->clip_l_path ? parameters->clip_l_path : "";
    selectedClipG = parameters->clip_g_path ? parameters->clip_g_path : "";
    selectedT5 = parameters->t5xxl_path ? parameters->t5xxl_path : "";
    selectedLlm = parameters->llm_path ? parameters->llm_path : "";
    selectedPrediction = parameters->prediction;
    embeddingCount = parameters->embedding_count;
    registeredEmbeddings.clear(); loadedEmbeddings.clear();
    for (unsigned i = 0; i < embeddingCount; ++i)
        registeredEmbeddings[parameters->embeddings[i].name] = parameters->embeddings[i].path;
    for (unsigned i = 0; i < embeddingCount; ++i)
        require(std::filesystem::is_regular_file(parameters->embeddings[i].path), "Default embedding is not a file");
    if (logCallback) logCallback(SD_LOG_WARN, warning, logData);
    return new sd_ctx_t;
}
void free_sd_ctx(sd_ctx_t *context) { if (context && context->refiner) --liveRefiners; delete context; }
sd_ctx_t *new_sd_ctx_with_ip_adapters(const sd_ctx_params_t *params, const sd_ip_adapter_model_t *models, uint32_t count) {
    ++ipLoads; ipPaths.clear();
    require(count && count <= 64 && models && params->memory_resident_model, "IP resources were not resident");
    for (uint32_t i = 0; i < count; ++i) ipPaths.emplace_back(models[i].adapter_path, models[i].vision_path);
    if (!ipLoadSucceeds) return nullptr;
    auto *ctx = new_sd_ctx(params); ctx->ipCount = count; return ctx;
}
bool sd_set_ip_adapter_inputs(sd_ctx_t *ctx, const sd_ip_adapter_input_t *inputs, uint32_t count) {
    if (!ipInputsSucceed) return false;
    ipMarkers.clear(); ipStrengths.clear(); ipMasks.clear();
    for (uint32_t i = 0; i < count; ++i) {
        const auto &input = inputs[i];
        require(input.slot == i && i < ctx->ipCount && input.image.data && input.image.channel == 3,
            "IP slot or RGB forwarding was incorrect");
        ipMarkers.push_back(input.image.data[0]); ipStrengths.push_back(input.strength); ipMasks.emplace_back();
        if (input.mask.data) ipMasks.back().assign(input.mask.data,
            input.mask.data + size_t(input.mask.width) * input.mask.height * input.mask.channel);
    }
    return true;
}
void sd_release_resident_model_memory() { ++runtimeReleaseCalls; }
bool sd_ctx_supports_image_generation(const sd_ctx_t *) { return true; }
const char *sd_get_model_family(const sd_ctx_t *ctx) { return ctx->refiner ? refinerFamily.c_str() : modelFamily.c_str(); }
bool sd_ctx_can_refine(const sd_ctx_t *base, const sd_ctx_t *refiner) {
    return base && !base->refiner && refiner && refiner->refiner && refinerCompatible;
}
bool sd_model_inspect_vae(const char *path, sd_model_vae_info_t *info) {
    if (!refinerFixturePath.empty() && path == refinerFixturePath) {
        *info = {refinerFamily.c_str(), "sdxl-base", SD_VAE_EMBEDDED};
        return true;
    }
    *info = {modelFamily.c_str(), modelFamily == "anima" ? "qwen-image"
        : modelFamily == "z-image" ? "flux1" : modelFamily.c_str(), missingVae ? SD_VAE_MISSING : SD_VAE_EMBEDDED};
    return true;
}
bool sd_model_validate_vae(const char *, const char *) { ++vaeValidationCalls; return validExternalVae; }
sample_method_t sd_get_default_sample_method(const sd_ctx_t *) { return static_cast<sample_method_t>(0); }
int sd_ctx_reference_image_capacity(const sd_ctx_t *) { return referenceCapacity; }
bool sd_ctx_has_control_net(const sd_ctx_t *) { return controlNetAvailable && !selectedControlNet.empty(); }
bool sd_prepare_control_nets(sd_ctx_t *ctx, const char *const *paths, uint32_t count) {
    if (!multiControlLoadSucceeds) return false;
    std::vector<std::string> next;
    for (uint32_t i = 0; i < count; ++i) next.emplace_back(paths[i]);
    if (ctx->multiPaths != next) {
        ++multiControlLoads;
        ctx->multiPaths = next;
    }
    preparedMultiPaths = std::move(next);
    return true;
}
bool sd_set_control_net_inputs(sd_ctx_t *ctx, const sd_control_input_t *inputs, uint32_t count) {
    if (!multiControlInputsSucceed) return false;
    require(!count || (inputs && count == ctx->multiPaths.size()), "Multi-ControlNet input/model counts differ");
    multiControlMarkers.clear(); multiControlStrengths.clear(); multiControlMasks.clear();
    for (uint32_t i = 0; i < count; ++i) {
        const auto &input = inputs[i];
        require(input.image.data && input.image.channel == 3, "Missing independent control RGB");
        multiControlMarkers.push_back(input.image.data[0]);
        multiControlStrengths.push_back(input.strength);
        multiControlMasks.emplace_back();
        if (input.mask.data) multiControlMasks.back().assign(input.mask.data,
            input.mask.data + size_t(input.mask.width) * input.mask.height * input.mask.channel);
    }
    return true;
}
bool sd_set_control_net_mask(sd_ctx_t *, sd_image_t mask) {
    selectedControlMask.clear();
    if (mask.data) selectedControlMask.assign(mask.data, mask.data + size_t(mask.width) * mask.height * mask.channel);
    return controlMaskSucceeds;
}
bool preprocess_canny(sd_image_t image, float high, float low, float weak, float strong, bool inverse) {
    require(high == 0.08f && low == 0.08f && weak == 0.8f && strong == 1 && !inverse,
        "Unexpected native Canny preprocessing contract");
    ++cannyCalls; image.data[0] = 255;
    return cannySucceeds;
}
void sd_set_prompt_weighting(sd_ctx_t *ctx, bool enabled) {
    (ctx->refiner ? refinerPromptWeighting : actualPromptWeighting) = enabled;
}
bool sd_set_freeu(sd_ctx_t *ctx, bool enabled) {
    (ctx->refiner ? refinerFreeU : actualFreeU) = enabled;
    return !enabled || freeUAvailable;
}
bool sd_prepare_hires_upscaler(sd_ctx_t *, const char *path) {
    ++upscalerPrepareCalls;
    preparedUpscaler = path;
    return upscalerAvailable;
}
adetailer_ctx_t *new_resident_adetailer_ctx(const char *path, int threads, const char *backend) {
    require(path && std::filesystem::is_regular_file(path) && threads > 0, "Invalid Detailer preparation request");
    if (expectCpu) require(backend && std::string(backend) == "cpu", "Detailer must honor CPU execution");
    ++detailerLoads;
    if (!detailerLoadSucceeds) return nullptr;
    ++liveDetailers;
    return new adetailer_ctx_t;
}
void free_adetailer_ctx(adetailer_ctx_t *ctx) { if (ctx) { --liveDetailers; delete ctx; } }
bool adetail_image(adetailer_ctx_t *ctx, sd_ctx_t *, sd_image_t input, const sd_adetailer_params_t *parameters,
    const sd_img_gen_params_t *inpaint, sd_image_t **images, int *count) {
    require(ctx && liveDetailers == 1 && input.data && input.channel == 3, "Detailer did not receive the generated image");
    require(ipMarkers.empty() && multiControlMarkers.empty() && selectedControlMask.empty(),
        "Whole-image IP/ControlNet conditioning leaked into Detailer crop coordinates");
    require(inpaint->width == 512 && inpaint->height == 512 && inpaint->strength == .25f,
        "Detailer lost crop dimensions or submitted denoise strength");
    require(parameters && !parameters->prompt && !parameters->negative_prompt && !previewCallback,
        "Detailer must inherit prompts without publishing crop-only previews");
    ++detailerCalls;
    *count = 1;
    *images = static_cast<sd_image_t *>(std::calloc(1, sizeof(sd_image_t)));
    ++allocatedImages;
    **images = input;
    (*images)->data = static_cast<uint8_t *>(std::calloc(input.width * input.height, 3));
    std::fill_n((*images)->data, input.width * input.height * 3, uint8_t(177));
    if (detailerCancellation) *detailerCancellation = true;
    return detailerRunSucceeds;
}
bool sd_load_textual_embedding(sd_ctx_t *ctx, const char *name) {
    if (ctx->refiner) return refinerEmbeddingSucceeds && refinerEmbeddings.contains(name);
    loadedEmbeddings.emplace_back(name);
    return embeddingLoadSucceeds && registeredEmbeddings.contains(name);
}
scheduler_t sd_get_default_scheduler(const sd_ctx_t *, sample_method_t) { return static_cast<scheduler_t>(0); }
void sd_img_gen_params_init(sd_img_gen_params_t *parameters) { *parameters = {}; }
void sd_cancel_generation(sd_ctx_t *, sd_cancel_mode_t) {}
bool convert_with_components(const char *, const char *, const char *, const char *, const char *,
    const char *, const char *, sd_type_t, const char *, bool, int) { return false; }
bool generate_image(sd_ctx_t *, const sd_img_gen_params_t *parameters, sd_image_t **images, int *count) {
    require(!detailerExpected || liveDetailers == 1, "Detailer weights must be prepared before base generation");
    if (parameters->hires.enabled && parameters->hires.upscaler == SD_HIRES_UPSCALER_MODEL)
        require(upscalerPrepareCalls > 0 && parameters->hires.model_path
            && preparedUpscaler == parameters->hires.model_path,
            "Learned upscaler must be prepared before any image generation");
    actualPrompt = parameters->prompt ? parameters->prompt : "";
    controlMarker = parameters->control_image.data ? parameters->control_image.data[0] : -1;
    controlWeight = parameters->control_strength;
    if (expectedReferenceCount >= 0) {
        require(parameters->init_image.data && parameters->init_image.data[0] == 11,
            "Initial reference pixels were not passed to img2img");
        require(parameters->ref_images_count == expectedReferenceCount, "Reference count was lost");
        actualImageStrength = parameters->strength;
        actualReferenceMarkers.clear();
        for (int i = 0; i < parameters->ref_images_count; ++i)
            actualReferenceMarkers.push_back(parameters->ref_images[i].data[0]);
    } else if (parameters->init_image.data) {
        ++imageBridgeCalls;
        require(parameters->init_image.channel == 3 && parameters->init_image.width == parameters->width * 2
            && parameters->init_image.height == parameters->height * 2 && parameters->strength == 0.35f,
            "A unified stage must receive decoded RGB and its own refinement strength");
        require(parameters->init_image.data[3] == 1 && parameters->init_image.data[5] == 1,
            "The unified bridge must carry the preceding image pixels");
    }
    require(parameters->hires.enabled == expectHires, "Native image must honor its Hires contract");
    actualFlowShift = parameters->sample_params.flow_shift;
    actualSampler = parameters->sample_params.sample_method;
    actualScheduler = parameters->sample_params.scheduler;
    actualClipSkip = parameters->clip_skip;
    actualSeamless = parameters->circular_x && parameters->circular_y;
    actualEta = parameters->sample_params.eta;
    actualSigmas.clear();
    if (parameters->sample_params.custom_sigmas_count)
        actualSigmas.assign(parameters->sample_params.custom_sigmas,
            parameters->sample_params.custom_sigmas + parameters->sample_params.custom_sigmas_count);
    actualCfg = parameters->sample_params.guidance.txt_cfg;
    actualDistilled = parameters->sample_params.guidance.distilled_guidance;
    require(parameters->hires.upscaler == expectedUpscaler
        && parameters->hires.denoising_strength == expectedDenoise,
        "Hires fix must decode, upscale with Lanczos, and denoise at strength 0.35");
    const int factor = expectHires ? 2 : 1;
    require(parameters->hires.target_width == (parameters->width * factor + 63) / 64 * 64
        && parameters->hires.target_height == (parameters->height * factor + 63) / 64 * 64,
        "The engine must receive half the requested axes and the aligned final target");
    require(parameters->hires.steps == std::max(1, int(parameters->sample_params.sample_steps * expectedDenoise)),
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
    ipAtGeneration = ipMarkers;
    if (!replaceIPDuringGeneration.empty()) { std::ofstream file(replaceIPDuringGeneration); file << "replaced IP resources during inference"; }
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
bool generate_image_with_refiner(sd_ctx_t *base, sd_ctx_t *refiner, float switchAt,
    const sd_img_gen_params_t *parameters, sd_image_t **images, int *count, const char *negative) {
    require(sd_ctx_can_refine(base, refiner) && liveRefiners == 1,
        "Refiner weights must be prepared before any image generation");
    ++refinerCalls; actualRefinerSwitch = switchAt;
    refinerNegative = negative ? negative : parameters->negative_prompt;
    const bool generated = generate_image(base, parameters, images, count);
    if (generated) for (int i = 0; i < *count; ++i)
        for (size_t pixel = 0; pixel < size_t((*images)[i].width) * (*images)[i].height; ++pixel)
            (*images)[i].data[pixel * (*images)[i].channel] = 221;
    if (refinerCancellation) *refinerCancellation = true;
    if (replaceRefinerDuringGeneration) { std::ofstream file(refinerFixturePath); file << "changed refiner weights"; }
    return generated && refinerRunSucceeds;
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
            expectResidentModel = true;
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
            {
                iild_native_request_v2 split{};
                // Desktop component-aware workers must stage source weights once,
                // including prepare-only and V3 custom sampling requests.
                expectResidentModel = true;
                split.size = sizeof(split); split.image = bridge;
                split.clip_l = split.clip_g = split.t5xxl = split.llm = split.vae = path.c_str();
                split.guidance_scale = 1.25f; split.distilled_guidance = 3.5f;
                split.prediction = 2;
                expectHires = false;
                releaseNativeDiffusionCache();
                split.image.prepare_only = 1;
                const auto callsBeforeResidentPreparation = generationCalls;
                auto *image = iild_native_generate_v2(&split, nullptr, nullptr, nullptr);
                require(image && std::string(iild_native_metadata_v1(image)).find("\"error\":\"\"") != std::string::npos
                    && generationCalls == callsBeforeResidentPreparation,
                    "Resident V2 preparation must not sample an image");
                iild_native_free_v1(image);
                split.image.prepare_only = 0;
                image = iild_native_generate_v2(&split, nullptr, nullptr, nullptr);
                require(image && std::string(iild_native_metadata_v1(image)).find("\"error\":\"\"") != std::string::npos,
                    "Component-aware native generation failed");
                require(std::string(iild_native_metadata_v1(image)).find("\"model_cache_hit\":true") != std::string::npos
                    && std::string(iild_native_metadata_v1(image)).find("\"weight_storage\":\"anonymous\"") != std::string::npos
                    && std::string(iild_native_metadata_v1(image)).find("\"weight_lifetime\":\"runtime\"") != std::string::npos,
                    "Resident desktop generation must reuse its prepared anonymous weights");
                require(selectedClipL == path && selectedClipG == path && selectedT5 == path
                    && selectedLlm == path && selectedVae == path,
                    "Split model component paths did not reach the native engine");
                require(actualCfg == 1.25f && actualDistilled == 3.5f,
                    "Family sampling settings did not reach the native engine");
                require(selectedPrediction == V_PRED, "Explicit v_prediction was lost during context construction");
                size_t size = 0;
                require(iild_native_rgb_v1(image, &size) && size == 64 * 120 * 3,
                    "V2 single pass must retain exact requested RGB dimensions");
                iild_native_free_v1(image);
                iild_native_request_v3 controlled{};
                controlled.size = sizeof(controlled); controlled.image = split;
                controlled.sampler = 2; controlled.flow_shift = .85f;
                std::vector<float> sigmas;
                for (int i = bridge.steps; i >= 0; --i) sigmas.push_back(float(i) / bridge.steps);
                controlled.sigmas = sigmas.data(); controlled.sigma_count = sigmas.size();
                image = iild_native_generate_v3(&controlled, nullptr, nullptr, nullptr);
                require(image && std::string(iild_native_metadata_v1(image)).find("\"error\":\"\"") != std::string::npos,
                    "V3 sampling request failed");
                require(actualSampler == HEUN_SAMPLE_METHOD && actualFlowShift == .85f && actualSigmas == sigmas,
                    "V3 sampler, shift or custom schedule did not reach the engine");
                require(std::string(iild_native_metadata_v1(image)).find("\"model_cache_hit\":true") != std::string::npos,
                    "Per-request scheduling should not reload model weights");
                iild_native_free_v1(image);
                // Krea V3 uses the same output contract as QuickGenerate:
                // align the inference canvas, then preserve requested RGB size.
                for (const auto &size : std::array<std::array<int, 2>, 5>{{
                         {1024, 1024}, {1368, 1024}, {1024, 1368},
                         {1824, 1024}, {1024, 1824}}}) {
                    auto rectangular = controlled;
                    rectangular.image.image.width = size[0];
                    rectangular.image.image.height = size[1];
                    image = iild_native_generate_v3(&rectangular, nullptr, nullptr, nullptr);
                    require(image && std::string(iild_native_metadata_v1(image)).find("\"error\":\"\"") != std::string::npos,
                        "V3 QuickGenerate rectangle failed");
                    size_t bytes = 0;
                    const auto *rgb = iild_native_rgb_v1(image, &bytes);
                    require(rgb && bytes == size_t(size[0]) * size[1] * 3,
                        "V3 must preserve QuickGenerate output dimensions");
                    require(rgb[0] == ((size[0] + 63) / 64 * 64 - size[0]) / 2
                         && rgb[1] == ((size[1] + 63) / 64 * 64 - size[1]) / 2,
                        "V3 must center-crop without stretching RGB");
                    iild_native_free_v1(image);
                }
                const int callsBeforeInvalid = generationCalls;
                for (int failure = 0; failure < 5; ++failure) {
                    auto invalid = controlled;
                    if (failure == 0) invalid.sigmas = nullptr;
                    if (failure == 1) invalid.sigma_count -= 1;
                    if (failure == 2) invalid.flow_shift = std::numeric_limits<float>::quiet_NaN();
                    if (failure == 3) invalid.sampler = 3;
                    if (failure == 4) invalid.image.size = 0;
                    image = iild_native_generate_v3(&invalid, nullptr, nullptr, nullptr);
                    require(image && std::string(iild_native_metadata_v1(image)).find("\"error\":\"\"") == std::string::npos,
                        "Invalid V3 controls were accepted");
                    iild_native_free_v1(image);
                }
                require(generationCalls == callsBeforeInvalid, "Invalid V3 controls reached inference");
                // Only the component binding changed: it must still invalidate residency.
                split.llm = nullptr;
                image = iild_native_generate_v2(&split, nullptr, nullptr, nullptr);
                require(image && std::string(iild_native_metadata_v1(image)).find("\"model_cache_hit\":false") != std::string::npos,
                    "Changed component binding reused the old context");
                iild_native_free_v1(image);
                split.size = 0;
                image = iild_native_generate_v2(&split, nullptr, nullptr, nullptr);
                require(image && std::string(iild_native_metadata_v1(image)).find("ABI version") != std::string::npos,
                    "V2 accepted an invalid ABI size");
                iild_native_free_v1(image);
                expectHires = true;
            }
            bridge.size = 0;
            auto *invalid = iild_native_generate_v1(&bridge, nullptr, nullptr);
            require(invalid && std::string(iild_native_metadata_v1(invalid)).find("ABI version") != std::string::npos,
                "Native bridge accepted an incompatible request layout");
            iild_native_free_v1(invalid);
            iild_native_release_v1();
            expectResidentModel = false;
        }
        request.steps = 10;
        std::atomic_bool cancelled{false};
        {
            NativeAdvancedControls advanced;
            NativeGenerationOptions modifiers;
            modifiers.defaultModifiers = false;
            NativeModelComponents components;
            components.guidanceScale = 6.5f;
            expectResidentModel = true;
            expectHires = false;
            const std::vector<std::pair<std::string, sample_method_t>> samplers{
                {"auto", EULER_SAMPLE_METHOD}, {"euler", EULER_SAMPLE_METHOD}, {"heun", HEUN_SAMPLE_METHOD},
                {"euler_a", EULER_A_SAMPLE_METHOD}, {"dpmpp_2m", DPMPP2M_SAMPLE_METHOD},
                {"dpmpp_sde", DPMPP2M_SDE_SAMPLE_METHOD}, {"ddim", DDIM_TRAILING_SAMPLE_METHOD}};
            const std::vector<std::pair<std::string, scheduler_t>> schedulers{{"auto", DISCRETE_SCHEDULER},
                {"normal", DISCRETE_SCHEDULER}, {"karras", KARRAS_SCHEDULER},
                {"exponential", EXPONENTIAL_SCHEDULER}, {"sgm_uniform", SGM_UNIFORM_SCHEDULER}};
            for (const auto &[name, expected] : samplers) {
                advanced.sampler = name;
                for (const auto &[schedule, expectedSchedule] : schedulers) {
                    advanced.scheduler = schedule;
                    advanced.clipSkip = 2;
                    advanced.seamlessTiling = true;
                    advanced.eta = 0.65f;
                    const auto image = generateNativeAdvancedImage(request, modifiers, components, advanced,
                        NativeComputeBackend::Automatic, cancelled);
                    require(image.error.empty() && !image.rgb.empty(), image.error);
                    require(actualSampler == expected && actualScheduler == expectedSchedule
                        && actualClipSkip == 2 && actualEta == 0.65f && actualCfg == 6.5f && actualSeamless,
                        "Advanced sampler/scheduler/CLIP/eta/CFG did not reach the native C API");
                }
            }
            advanced.hires = expectHires = true;
            advanced.denoiseStrength = expectedDenoise = 0.25f;
            for (const auto &[name, mode] : std::vector<std::pair<std::string, sd_hires_upscaler_t>>{
                {"nearest", SD_HIRES_UPSCALER_NEAREST}, {"bilinear", SD_HIRES_UPSCALER_BILINEAR},
                {"bicubic", SD_HIRES_UPSCALER_BICUBIC}, {"lanczos", SD_HIRES_UPSCALER_LANCZOS}}) {
                advanced.upscaler = name; expectedUpscaler = mode;
                const auto refined = generateNativeAdvancedImage(request, modifiers, components, advanced,
                    NativeComputeBackend::Automatic, cancelled);
                require(refined.error.empty() && !refined.rgb.empty(), refined.error);
            }
            advanced.upscaler = "4x-ultra";
            expectedUpscaler = SD_HIRES_UPSCALER_MODEL;
            auto beforeLearned = generationCalls;
            require(!generateNativeAdvancedImage(request, modifiers, components, advanced,
                NativeComputeBackend::Automatic, cancelled).error.empty() && generationCalls == beforeLearned,
                "Missing learned weights must fail before base generation");
            advanced.upscalerModel = request.modelPath; // C API fixture, not an ESRGAN quality claim.
            auto learned = generateNativeAdvancedImage(request, modifiers, components, advanced,
                NativeComputeBackend::Automatic, cancelled);
            require(learned.error.empty() && !learned.rgb.empty() && !learned.modelCacheHit, learned.error);
            learned = generateNativeAdvancedImage(request, modifiers, components, advanced,
                NativeComputeBackend::Automatic, cancelled);
            require(learned.error.empty() && learned.modelCacheHit, "Learned weights broke warm cache reuse");
            upscalerAvailable = false;
            beforeLearned = generationCalls;
            require(!generateNativeAdvancedImage(request, modifiers, components, advanced,
                NativeComputeBackend::Automatic, cancelled).error.empty() && generationCalls == beforeLearned,
                "Invalid learned weights must never fall back to interpolation");
            upscalerAvailable = true;
            const auto before = generationCalls;
            advanced.upscaler = "unknown";
            require(!generateNativeAdvancedImage(request, modifiers, components, advanced,
                NativeComputeBackend::Automatic, cancelled).error.empty() && generationCalls == before,
                "Invalid advanced controls reached native inference");
            expectedDenoise = 0.35f;
            expectedUpscaler = SD_HIRES_UPSCALER_LANCZOS;
            advanced.hires = expectHires = false;
            advanced.references = {{64, 32, std::vector<uint8_t>(64 * 32 * 3, 11)}};
            expectedReferenceCount = 0;
            for (float amount : {0.0f, 0.65f, 1.0f}) {
                advanced.imageStrength = amount;
                const auto image = generateNativeAdvancedImage(request, modifiers, components, advanced,
                    NativeComputeBackend::Automatic, cancelled);
                require(image.error.empty() && actualImageStrength == amount, "Reference strength did not reach img2img: " + image.error);
            }
            advanced.references.push_back({32, 64, std::vector<uint8_t>(32 * 64 * 3, 22)});
            const auto beforeUnsupported = generationCalls;
            require(!generateNativeAdvancedImage(request, modifiers, components, advanced,
                NativeComputeBackend::Automatic, cancelled).error.empty() && generationCalls == beforeUnsupported,
                "A non-reference model silently accepted multiple images");
            referenceCapacity = 20; expectedReferenceCount = 2;
            const auto multiple = generateNativeAdvancedImage(request, modifiers, components, advanced,
                NativeComputeBackend::Automatic, cancelled);
            require(multiple.error.empty() && actualReferenceMarkers == std::vector<int>({11, 22}),
                "Ordered reference pixels did not reach the editing model: " + multiple.error);
            advanced.references.front().rgb.pop_back();
            require(!generateNativeAdvancedImage(request, modifiers, components, advanced,
                NativeComputeBackend::Automatic, cancelled).error.empty(), "Truncated reference RGB was accepted");
            referenceCapacity = 0; expectedReferenceCount = -1; expectHires = true;
            expectResidentModel = false;
        }
        {
            releaseNativeDiffusionCache();
            expectHires = false; expectResidentModel = true;
            const auto model = directory / "selected-control.safetensors";
            { std::ofstream file(model); file << "control fixture"; }
            NativeAdvancedControls advanced;
            advanced.controls.push_back({std::filesystem::canonical(model),
                {16, 16, std::vector<uint8_t>(16 * 16 * 3, 41)}, "Canny", 0.7f});
            NativeGenerationOptions options; options.defaultModifiers = false;
            NativeModelComponents components;
            auto result = generateNativeAdvancedImage(request, options, components, advanced,
                NativeComputeBackend::Automatic, cancelled);
            require(result.error.empty() && selectedControlNet == model.string() && controlMarker == 255
                && controlWeight == 0.7f && advanced.controls.front().image.rgb.front() == 41,
                "ControlNet model, private Canny pixels or strength were lost: " + result.error);
            advanced.controls.front().process = "Pose";
            advanced.controls.front().poseDetector = directory / "detector.onnx";
            advanced.controls.front().poseModel = directory / "pose.onnx";
            result = generateNativeAdvancedImage(request, options, components, advanced,
                NativeComputeBackend::Automatic, cancelled);
            require(result.error.empty() && controlMarker == 67 && advanced.controls.front().image.rgb.front() == 41,
                "Pose-conditioned pixels did not reach ControlNet: " + result.error);
            advanced.controls.front().process = "Canny";
            advanced.controls.front().mask = {2, 1, {0, 0, 0, 255, 255, 255}};
            result = generateNativeAdvancedImage(request, options, components, advanced,
                NativeComputeBackend::Automatic, cancelled);
            require(result.error.empty() && result.modelCacheHit
                && selectedControlMask == advanced.controls.front().mask.rgb,
                "Regional mask was lost or reloaded model weights");
            advanced.controls.front().mask.rgb.pop_back();
            const auto beforeInvalidMask = generationCalls;
            require(!generateNativeAdvancedImage(request, options, components, advanced,
                NativeComputeBackend::Automatic, cancelled).error.empty() && generationCalls == beforeInvalidMask,
                "Malformed mask reached sampling");
            advanced.controls.front().mask = {};
            controlMaskSucceeds = false;
            require(!generateNativeAdvancedImage(request, options, components, advanced,
                NativeComputeBackend::Automatic, cancelled).error.empty() && generationCalls == beforeInvalidMask,
                "Failed mask preparation reached sampling");
            controlMaskSucceeds = true;
            result = generateNativeAdvancedImage(request, options, components, advanced,
                NativeComputeBackend::Automatic, cancelled);
            require(result.error.empty() && selectedControlMask.empty(), "Disabled mask leaked into a later request");
            const auto beforeCanny = cannyCalls;
            advanced.controls.front().process = "Tile";
            advanced.controls.front().weight = 0;
            result = generateNativeAdvancedImage(request, options, components, advanced,
                NativeComputeBackend::Automatic, cancelled);
            require(result.error.empty() && result.modelCacheHit && controlMarker == 41
                && controlWeight == 0 && cannyCalls == beforeCanny,
                "Tile must preserve pixels and reuse weights without Canny preprocessing");
            { std::ofstream file(model); file << "changed ControlNet fixture"; }
            result = generateNativeAdvancedImage(request, options, components, advanced,
                NativeComputeBackend::Automatic, cancelled);
            require(result.error.empty() && !result.modelCacheHit, "Changed ControlNet reused stale weights");
            auto before = generationCalls;
            controlNetAvailable = false;
            result = generateNativeAdvancedImage(request, options, components, advanced,
                NativeComputeBackend::Automatic, cancelled);
            require(!result.error.empty() && generationCalls == before, "Missing ControlNet reached inference");
            controlNetAvailable = true;
            cannySucceeds = false; advanced.controls.front().process = "Canny";
            result = generateNativeAdvancedImage(request, options, components, advanced,
                NativeComputeBackend::Automatic, cancelled);
            require(!result.error.empty() && generationCalls == before, "Failed preprocessing reached inference");
            cannySucceeds = true;
            const auto secondModel = directory / "second-control.safetensors";
            { std::ofstream file(secondModel); file << "second control fixture"; }
            advanced.controls.front().weight = 0.5f;
            advanced.controls.front().mask = {1, 1, {255, 255, 255}};
            advanced.controls.push_back({std::filesystem::canonical(secondModel),
                {8, 8, std::vector<uint8_t>(8 * 8 * 3, 73)}, "Tile", 1.25f,
                {1, 1, {128, 128, 128}}});
            result = generateNativeAdvancedImage(request, options, components, advanced,
                NativeComputeBackend::Automatic, cancelled);
            require(result.error.empty() && !result.modelCacheHit && selectedControlNet.empty()
                && preparedMultiPaths == std::vector<std::string>({model.string(), secondModel.string()})
                && multiControlMarkers == std::vector<int>({255, 73})
                && multiControlStrengths == std::vector<float>({0.5f, 1.25f})
                && multiControlMasks == std::vector<std::vector<uint8_t>>({{255, 255, 255}, {128, 128, 128}})
                && controlWeight == 1.f && selectedControlMask.empty(),
                "Ordered independent ControlNet inputs were lost: " + result.error);
            const auto multiLoads = multiControlLoads;
            result = generateNativeAdvancedImage(request, options, components, advanced,
                NativeComputeBackend::Automatic, cancelled);
            require(result.error.empty() && result.modelCacheHit && multiControlLoads == multiLoads,
                "Warm multi-ControlNet weights were unnecessarily reloaded");
            advanced.controls.back().weight = 1.75f;
            advanced.controls.back().image.rgb.front() = 91;
            advanced.controls.back().mask = {};
            result = generateNativeAdvancedImage(request, options, components, advanced,
                NativeComputeBackend::Automatic, cancelled);
            require(result.error.empty() && result.modelCacheHit && multiControlLoads == multiLoads
                && multiControlMarkers.back() == 91 && multiControlStrengths.back() == 1.75f
                && multiControlMasks.back().empty(), "Input edits reloaded weights or retained old multi input");
            { std::ofstream file(secondModel); file << "changed second ControlNet fixture"; }
            result = generateNativeAdvancedImage(request, options, components, advanced,
                NativeComputeBackend::Automatic, cancelled);
            require(result.error.empty() && !result.modelCacheHit && multiControlLoads > multiLoads,
                "Changing the second ControlNet model reused stale weights");
            before = generationCalls;
            multiControlLoadSucceeds = false;
            result = generateNativeAdvancedImage(request, options, components, advanced,
                NativeComputeBackend::Automatic, cancelled);
            require(!result.error.empty() && generationCalls == before,
                "Failed multi-ControlNet preparation reached inference");
            multiControlLoadSucceeds = true; multiControlInputsSucceed = false;
            result = generateNativeAdvancedImage(request, options, components, advanced,
                NativeComputeBackend::Automatic, cancelled);
            require(!result.error.empty() && generationCalls == before,
                "Failed multi-ControlNet input preparation reached inference");
            multiControlInputsSucceed = true;
            advanced.controls.pop_back();
            result = generateNativeAdvancedImage(request, options, components, advanced,
                NativeComputeBackend::Automatic, cancelled);
            require(result.error.empty() && multiControlMarkers.empty() && controlWeight == .5f,
                "Multi-ControlNet inputs leaked into a single-control request");
            before = generationCalls;
            advanced.controls.front().process = "Depth";
            result = generateNativeAdvancedImage(request, options, components, advanced,
                NativeComputeBackend::Automatic, cancelled);
            require(!result.error.empty() && generationCalls == before, "An unimplemented detector reached inference");
            advanced.controls.front().process = "Tile"; advanced.controls.front().image.rgb.pop_back();
            result = generateNativeAdvancedImage(request, options, components, advanced,
                NativeComputeBackend::Automatic, cancelled);
            require(!result.error.empty() && generationCalls == before, "Invalid control RGB was accepted");
            advanced.controls.front().image.rgb.push_back(41);
            releaseNativeDiffusionCache(); modelFamily = "flux1";
            result = generateNativeAdvancedImage(request, options, components, advanced,
                NativeComputeBackend::Automatic, cancelled);
            require(!result.error.empty() && generationCalls == before, "Unsupported control architecture reached inference");
            modelFamily = "sdxl-base"; releaseNativeDiffusionCache();
            advanced.controls.clear();
            advanced.freeU = true;
            result = generateNativeAdvancedImage(request, options, components, advanced,
                NativeComputeBackend::Automatic, cancelled);
            require(result.error.empty() && selectedControlNet.empty() && controlMarker == -1 && actualFreeU,
                "Removed ControlNet leaked into the next request");
            advanced.freeU = false;
            result = generateNativeAdvancedImage(request, options, components, advanced,
                NativeComputeBackend::Automatic, cancelled);
            require(result.error.empty() && result.modelCacheHit && !actualFreeU,
                "Disabling FreeU must reset the cached model without reloading weights");
            advanced.freeU = true;
            result = generateNativeAdvancedImage(request, options, components, advanced,
                NativeComputeBackend::Automatic, cancelled);
            require(result.error.empty() && actualFreeU, "FreeU could not be re-enabled on a warm model");
            result = generateNativeImageWithResidentWeights(request, options, components, NativeSamplingControls{},
                NativeComputeBackend::Automatic, cancelled, false);
            require(result.error.empty() && result.modelCacheHit && !actualFreeU,
                "FreeU leaked into a legacy resident request sharing the same model context");
            advanced.freeU = true; freeUAvailable = false;
            before = generationCalls;
            result = generateNativeAdvancedImage(request, options, components, advanced,
                NativeComputeBackend::Automatic, cancelled);
            require(!result.error.empty() && generationCalls == before, "Unsupported FreeU silently reached sampling");
            freeUAvailable = true;
            expectHires = true; expectResidentModel = false;
        }
        {
            expectResidentModel = true;
            expectHires = false;
            releaseNativeDiffusionCache();
            const auto a = std::filesystem::canonical(directory) / "ip-first.safetensors";
            const auto b = std::filesystem::canonical(directory) / "ip-second.safetensors";
            const auto v = std::filesystem::canonical(directory) / "ip-vision.safetensors";
            for (const auto &path : {a, b, v}) { std::ofstream file(path); file << "IP resource fixture"; }
            NativeGenerationOptions options;
            NativeModelComponents components;
            NativeAdvancedControls advanced;
            advanced.ipAdapters = {{a, v, {2, 1, std::vector<uint8_t>(6, 31)}, .3f, {2, 1, {0, 0, 0, 255, 255, 255}}},
                {b, v, {1, 1, std::vector<uint8_t>(3, 71)}, .7f, {}}};
            const auto generate = [&] { return generateNativeAdvancedImage(request, options, components, advanced,
                NativeComputeBackend::Automatic, cancelled); };
            auto result = generate();
            require(result.error.empty() && ipPaths == std::vector<std::pair<std::string, std::string>>{{a.string(), v.string()}, {b.string(), v.string()}}
                && ipMarkers == std::vector<int>{31, 71} && ipStrengths == std::vector<float>{.3f, .7f}
                && ipMasks.front() == advanced.ipAdapters.front().mask.rgb && ipMasks.back().empty(),
                "Independent IP resources, raw images, masks or weights were not forwarded: " + result.error);
            const auto loads = ipLoads;
            advanced.ipAdapters.back().weight = 0; advanced.ipAdapters.back().image.rgb.front() = 81;
            result = generate();
            require(result.error.empty() && result.modelCacheHit && ipLoads == loads && ipMarkers.back() == 81
                && ipStrengths.back() == 0, "IP input edits rebuilt model resources or discarded zero weight");
            auto saved = advanced.ipAdapters;
            advanced.ipAdapters.clear(); result = generate();
            require(result.error.empty() && result.modelCacheHit && ipLoads == loads && ipMarkers.empty(),
                "Disabled IP inputs leaked or unloaded resident resources");
            result = generateNativeImageWithResidentWeights(request, options, components, NativeSamplingControls{},
                NativeComputeBackend::Automatic, cancelled, false);
            require(result.error.empty() && result.modelCacheHit && ipMarkers.empty(), "IP leaked into a legacy resident request");
            advanced.ipAdapters = saved; result = generate();
            require(result.error.empty() && result.modelCacheHit && ipLoads == loads, "Re-enabling IP reloaded resident resources");
            for (const auto &path : {b, v}) {
                { std::ofstream file(path); file << "changed resource " << path.string(); }
                result = generate();
                require(result.error.empty() && !result.modelCacheHit, "Changed IP or vision weights reused a stale context");
            }
            auto before = generationCalls;
            advanced.ipAdapters.back().image.rgb.pop_back(); result = generate();
            require(!result.error.empty() && generationCalls == before, "Invalid IP image reached sampling");
            advanced.ipAdapters = saved; advanced.ipAdapters.back().mask = {2, 1, {0}}; result = generate();
            require(!result.error.empty() && generationCalls == before, "Invalid IP mask reached sampling");
            advanced.ipAdapters = saved; advanced.ipAdapters.back().weight = std::numeric_limits<float>::quiet_NaN(); result = generate();
            require(!result.error.empty() && generationCalls == before, "Invalid IP strength reached sampling");
            advanced.ipAdapters = saved; advanced.ipAdapters.back().vision = directory / "missing-vision.safetensors"; result = generate();
            require(!result.error.empty() && generationCalls == before, "Missing vision model reached sampling");
            advanced.ipAdapters = saved; advanced.ipAdapters.resize(65, saved.front()); result = generate();
            require(!result.error.empty() && generationCalls == before, "Excess IP inputs reached sampling");
            advanced.ipAdapters = saved; releaseNativeDiffusionCache(); ipLoadSucceeds = false; result = generate();
            require(!result.error.empty() && generationCalls == before, "Failed IP loading reached sampling");
            ipLoadSucceeds = true; ipInputsSucceed = false; result = generate();
            require(!result.error.empty() && generationCalls == before, "Failed IP input preparation reached sampling");
            ipInputsSucceed = true; releaseNativeDiffusionCache(); modelFamily = "flux1"; result = generate();
            require(!result.error.empty() && generationCalls == before, "Unsupported IP model family reached sampling");
            modelFamily = "sdxl-base"; releaseNativeDiffusionCache();
            replaceIPDuringGeneration = v; result = generate(); replaceIPDuringGeneration.clear();
            require(!result.error.empty() && result.rgb.empty(), "Changed IP resources published an unverifiable output");
            result = generate(); require(result.error.empty(), "IP failed to recover after resource replacement");
            advanced.controls = {{a, saved.front().image, "Canny", .5f, {}}};
            result = generate();
            require(result.error.empty() && controlMarker == 255 && ipMarkers.front() == 31
                && advanced.ipAdapters.front().image.rgb.front() == 31,
                "Canny preprocessing replaced the original IP reference image");
            expectCpu = true;
            result = generateNativeAdvancedImage(request, options, components, advanced,
                NativeComputeBackend::Cpu, cancelled);
            require(result.error.empty() && !result.modelCacheHit && ipMarkers.front() == 31,
                "IP resources failed to transition to explicit CPU placement");
            expectCpu = false;
            advanced.controls.push_back({b, saved.back().image, "Tile", .2f, {}});
            advanced.detailer = true; advanced.detailerModel = a; detailerExpected = true;
            result = generate();
            require(result.error.empty() && result.rgb.front() == 177 && ipAtGeneration == std::vector<int>{31, 81}
                && ipMarkers.empty(), "IP + multiple ControlNets + Detailer did not separate whole-image and crop inputs: " + result.error);
            const auto retainedLoads = ipLoads;
            result = generate();
            require(result.error.empty() && result.modelCacheHit && ipLoads == retainedLoads
                && ipAtGeneration == std::vector<int>{31, 81}, "Detailer cleanup broke the next warm IP request");
            detailerExpected = false;
            releaseNativeDiffusionCache(); expectHires = true; expectResidentModel = false;
        }
        {
            expectResidentModel = true;
            expectHires = false;
            releaseNativeDiffusionCache();
            const auto fixture = directory / "refiner.safetensors";
            { std::ofstream file(fixture); file << "refiner fixture"; }
            refinerFixturePath = std::filesystem::canonical(fixture).string();
            NativeGenerationOptions refineOptions; refineOptions.defaultModifiers = true;
            refineOptions.negativePrompt = "user negative";
            NativeModelComponents refineComponents;
            NativeAdvancedControls refineControls;
            refineControls.refiner = true; refineControls.refinerSwitch = .65f;
            const auto generateRefined = [&] {
                return generateNativeAdvancedImage(request, refineOptions, refineComponents, refineControls,
                    NativeComputeBackend::Automatic, cancelled);
            };
            const auto beforeRefiner = generationCalls;
            require(!generateRefined().error.empty() && generationCalls == beforeRefiner,
                "Missing Refiner weights reached sampling");
            refineControls.refinerModel = refinerFixturePath;
            for (const auto invalid : {-0.1f, 1.1f, std::numeric_limits<float>::quiet_NaN()}) {
                refineControls.refinerSwitch = invalid;
                require(!generateRefined().error.empty() && generationCalls == beforeRefiner,
                    "Invalid Refiner switch reached sampling");
            }
            refineControls.refinerSwitch = .65f;
            auto refined = generateRefined();
            require(refined.error.empty() && refined.rgb.front() == 221 && allocatedImages == 0
                && actualRefinerSwitch == .65f && refinerNegative == "user negative",
                "Refiner output or controls were not forwarded: " + refined.error);
            require(refinerEmbeddings.empty() && negativePrompt.find("iild_ndxl") != std::string::npos
                && refinerNegative.find("iild_ndxl") == std::string::npos,
                "Base automatic negative embeddings leaked into Refiner");
            const auto loads = refinerLoads;
            refineControls.refinerSwitch = .25f;
            refined = generateRefined();
            require(refined.error.empty() && refined.modelCacheHit && refinerLoads == loads
                && actualRefinerSwitch == .25f, "Switch adjustment reloaded Refiner weights");
            const auto calls = refinerCalls;
            refineControls.refiner = false;
            refined = generateRefined();
            require(refined.error.empty() && refined.modelCacheHit && refinerCalls == calls && liveRefiners == 1,
                "Disabling Refiner executed or unloaded the resident model");
            refineControls.refiner = true; refineControls.refinerSwitch = 1.f;
            refined = generateRefined();
            require(refined.error.empty() && refinerCalls == calls && liveRefiners == 1,
                "Switch endpoint one executed or unloaded Refiner");
            refineControls.refinerSwitch = 0.f;
            refineControls.freeU = true; refineControls.promptWeighting = false;
            refined = generateRefined();
            require(refined.error.empty() && refined.modelCacheHit && refinerLoads == loads
                && actualRefinerSwitch == 0.f && refinerFreeU && !refinerPromptWeighting,
                "Warm Refiner lost per-request controls");
            refinerRunSucceeds = false;
            refined = generateRefined();
            require(!refined.error.empty() && refined.rgb.empty() && allocatedImages == 0 && liveRefiners == 0,
                "Failed Refiner published Base output or leaked buffers");
            refinerRunSucceeds = true; refinerLoadSucceeds = false;
            const auto beforeLoadFailure = generationCalls;
            require(!generateRefined().error.empty() && generationCalls == beforeLoadFailure,
                "Refiner load failure reached sampling");
            refinerLoadSucceeds = true; refinerCompatible = false;
            require(!generateRefined().error.empty() && generationCalls == beforeLoadFailure,
                "Incompatible Refiner pair reached sampling");
            refinerCompatible = true;
            modelFamily = "sd1"; releaseNativeDiffusionCache();
            require(!generateRefined().error.empty() && generationCalls == beforeLoadFailure,
                "Non-SDXL Base reached Refiner sampling");
            modelFamily = "sdxl-base"; releaseNativeDiffusionCache();
            refinerCancellation = &cancelled;
            refined = generateRefined();
            require(refined.cancelled && refined.rgb.empty() && allocatedImages == 0 && liveRefiners == 0,
                "Late Refiner cancellation published output or leaked buffers");
            refinerCancellation = nullptr; cancelled = false;
            replaceRefinerDuringGeneration = true;
            refined = generateRefined();
            require(!refined.error.empty() && refined.rgb.empty() && allocatedImages == 0,
                "Changed Refiner source published inconsistent output");
            replaceRefinerDuringGeneration = false;
            const auto embeddingFixture = directory / "refiner-embedding.safetensors";
            { std::ofstream file(embeddingFixture); file << "explicit embedding"; }
            refineControls.embeddings.push_back({"user_refiner", std::filesystem::canonical(embeddingFixture)});
            refined = generateRefined();
            require(refined.error.empty() && refinerEmbeddings.size() == 1
                && refinerEmbeddings.contains("user_refiner"), "Explicit Refiner embedding was not registered");
            refinerEmbeddingSucceeds = false;
            const auto beforeEmbeddingFailure = generationCalls;
            require(!generateRefined().error.empty() && generationCalls == beforeEmbeddingFailure,
                "Incompatible Refiner embedding reached sampling");
            refinerEmbeddingSucceeds = true;
            refined = generateRefined();
            require(refined.error.empty() && liveRefiners == 1, "Refiner could not recover after validation failure");
            releaseNativeDiffusionCache();
            require(liveRefiners == 0, "Explicit release retained Refiner context");
            refinerFixturePath.clear();
            NativeGenerationOptions detailOptions; detailOptions.defaultModifiers = false;
            NativeModelComponents detailComponents;
            NativeAdvancedControls detailControls;
            detailControls.detailer = true;
            auto beforeDetail = generationCalls;
            auto detailResult = generateNativeAdvancedImage(request, detailOptions, detailComponents, detailControls,
                NativeComputeBackend::Automatic, cancelled);
            require(!detailResult.error.empty() && generationCalls == beforeDetail, "Missing detector reached generation");
            detailControls.detailerModel = request.modelPath; // Weight contents belong to the separate real detector test.
            detailerExpected = true;
            detailResult = generateNativeAdvancedImage(request, detailOptions, detailComponents, detailControls,
                NativeComputeBackend::Automatic, cancelled);
            require(detailResult.error.empty() && detailResult.rgb.front() == 177 && allocatedImages == 0,
                "Detailer output was not published or buffers leaked");
            const auto loaded = detailerLoads;
            detailResult = generateNativeAdvancedImage(request, detailOptions, detailComponents, detailControls,
                NativeComputeBackend::Automatic, cancelled);
            require(detailResult.error.empty() && detailResult.modelCacheHit && detailerLoads == loaded,
                "Warm Detailer reloaded its detector");
            detailerRunSucceeds = false;
            detailResult = generateNativeAdvancedImage(request, detailOptions, detailComponents, detailControls,
                NativeComputeBackend::Automatic, cancelled);
            require(!detailResult.error.empty() && detailResult.rgb.empty() && allocatedImages == 0 && liveDetailers == 0,
                "Failed Detailer silently returned the base image or leaked resources");
            detailerRunSucceeds = true; detailerLoadSucceeds = false;
            beforeDetail = generationCalls;
            detailResult = generateNativeAdvancedImage(request, detailOptions, detailComponents, detailControls,
                NativeComputeBackend::Automatic, cancelled);
            require(!detailResult.error.empty() && generationCalls == beforeDetail, "Unloaded detector reached base generation");
            detailerLoadSucceeds = true; detailerCancellation = &cancelled;
            detailResult = generateNativeAdvancedImage(request, detailOptions, detailComponents, detailControls,
                NativeComputeBackend::Automatic, cancelled);
            require(detailResult.cancelled && detailResult.rgb.empty() && allocatedImages == 0 && liveDetailers == 0,
                "Cancelled Detailer published output or leaked resources");
            detailerCancellation = nullptr; cancelled = false; detailerExpected = false;
            const auto callsBeforeDisabled = detailerCalls;
            detailControls.detailer = false;
            detailResult = generateNativeAdvancedImage(request, detailOptions, detailComponents, detailControls,
                NativeComputeBackend::Automatic, cancelled);
            require(detailResult.error.empty() && detailerCalls == callsBeforeDisabled, "Disabled Detailer still ran");
            const auto embedding = directory / "selected-embedding.safetensors";
            { std::ofstream file(embedding); file << "embedding fixture"; }
            const auto originalPrompt = request.prompt;
            request.prompt = "coastalness";
            NativeGenerationOptions options;
            options.defaultModifiers = false;
            NativeModelComponents components;
            NativeAdvancedControls controls;
            controls.promptWeighting = false;
            controls.embeddings.push_back({"coastal", std::filesystem::canonical(embedding)});
            auto result = generateNativeAdvancedImage(request, options, components, controls,
                NativeComputeBackend::Automatic, cancelled);
            require(result.error.empty() && embeddingCount == 1 && !actualPromptWeighting
                && actualPrompt == "coastalness coastal" && loadedEmbeddings == std::vector<std::string>{"coastal"},
                "Selected embedding or literal prompt policy did not reach the native engine: " + result.error);
            options.negativePrompt = "(coastal:0.8)";
            controls.promptWeighting = true;
            result = generateNativeAdvancedImage(request, options, components, controls,
                NativeComputeBackend::Automatic, cancelled);
            require(result.error.empty() && result.modelCacheHit && actualPromptWeighting && actualPrompt == "coastalness",
                "Weighting toggle must reuse the context, and an explicit negative embedding must not be added positively");
            { std::ofstream file(embedding); file << "changed embedding fixture"; }
            result = generateNativeAdvancedImage(request, options, components, controls,
                NativeComputeBackend::Automatic, cancelled);
            require(result.error.empty() && !result.modelCacheHit, "Changed embedding weights reused a stale context");
            embeddingLoadSucceeds = false;
            const auto before = generationCalls;
            result = generateNativeAdvancedImage(request, options, components, controls,
                NativeComputeBackend::Automatic, cancelled);
            require(!result.error.empty() && generationCalls == before, "An incompatible embedding silently reached inference");
            embeddingLoadSucceeds = true;
            controls.embeddings.push_back(controls.embeddings.front());
            require(!generateNativeAdvancedImage(request, options, components, controls,
                NativeComputeBackend::Automatic, cancelled).error.empty(), "Duplicate embedding tokens were accepted");
            request.prompt = originalPrompt;
            expectHires = true; expectResidentModel = false;
        }
        {
            expectResidentModel = true;
            const auto resident = generateNativeImageWithResidentWeights(request, {}, cancelled);
            expectResidentModel = false;
            require(resident.error.empty() && !resident.modelCacheHit,
                "Resident model mode failed or reused a nonresident context: " + resident.error);
            request.q8CacheDirectory = directory / "resident-q8-must-not-exist";
            const auto callsBeforeQ8Rejection = generationCalls;
            expectResidentModel = true;
            const auto rejected = generateNativeImageWithResidentWeights(request, {}, cancelled);
            expectResidentModel = false;
            request.q8CacheDirectory.clear();
            require(!rejected.error.empty() && rejected.error.find("disk-backed Q8 cache") != std::string::npos
                && generationCalls == callsBeforeQ8Rejection
                && !std::filesystem::exists(directory / "resident-q8-must-not-exist"),
                "Resident mode must reject the disk-backed Q8 cache before writing or inference");
        }
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
            require(embeddingCount == 7 && loraCount == 0,
                    "Native defaults must keep negative embeddings without injecting a LoRA");
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
            const auto releaseCount = runtimeReleaseCalls;
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
            require(runtimeReleaseCalls == releaseCount, "Generation failure evicted runtime source memory");
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
            const auto writeManifest = [&](const std::string &second, double strength = 0.35, const std::string &vae = "") {
                std::ofstream file(package / "model_index.json");
                file << R"({"schema":"iild-unified-model-v1","container":"zip-stored-v1","_class_name":"IILDUnifiedCascade","composition":"ordered-image-refinement","stages":[)"
                     << R"({"model":"a.safetensors","size_bytes":13,"strength":1},)"
                     << "{\"model\":\"" << second << "\",\"size_bytes\":13,\"strength\":" << strength
                     << (vae.empty() ? "" : ",\"vae\":\"" + vae + "\"") << "}]}";
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
            const auto packaged = std::filesystem::canonical(directory) / "packaged.iildmodel";
            if (auto *archive = zip_open(packaged.string().c_str(), 0, 'w')) {
                for (const auto *name : {"a.safetensors", "b.safetensors", "model_index.json"}) {
                    require(zip_entry_open(archive, name) == 0
                        && zip_entry_fwrite(archive, (package / name).string().c_str()) == 0
                        && zip_entry_close(archive) == 0, "Could not build packaged unified fixture");
                }
                zip_close(archive);
            } else throw std::runtime_error("Could not open packaged unified fixture");
            request.modelPath = packaged;
            const auto packagedBefore = generationCalls;
            cascade = generateNativeImageWithOptions(request, options, cancelled);
            require(cascade.error.empty() && generationCalls == packagedBefore + 2,
                "A packaged .iildmodel file did not execute as a unified cascade: " + cascade.error);
            const auto residentPackage = generateNativeImageWithResidentWeights(request, options, cancelled);
            require(!residentPackage.error.empty() && residentPackage.error.find("archive extraction to disk is disabled") != std::string::npos
                && generationCalls == packagedBefore + 2,
                "Memory-resident generation must not materialize an archive package to disk");
            request.modelPath = package;
            int loads = 0;
            cascade = generateNativeImageWithOptions(request, options, cancelled, [&](const auto &event) {
                if (event.stage == NativeGenerationStage::Encoding && ++loads == 2) cancelled = true;
            });
            require(cascade.cancelled && cascade.rgb.empty(), "A cancelled cascade published an earlier stage as success");
            cancelled = false;
            writeManifest("b.safetensors", 0.35, "a.safetensors");
            const auto validationBefore = vaeValidationCalls;
            cascade = generateNativeImageWithOptions(request, options, cancelled);
            require(cascade.error.empty() && selectedVae == (package / "a.safetensors").string()
                && vaeValidationCalls > validationBefore,
                "A package VAE must override an embedded VAE and be validated by the generation engine: "
                + cascade.error + " selected=" + selectedVae);
            validExternalVae = false;
            writeManifest("b.safetensors", 0.35, "b.safetensors");
            cascade = generateNativeImageWithOptions(request, options, cancelled);
            require(!cascade.error.empty() && cascade.rgb.empty(), "An incompatible explicit VAE must report generation failure");
            validExternalVae = true;
            writeManifest("b.safetensors", 0.35, "../result-fixture.safetensors");
            cascade = generateNativeImageWithOptions(request, options, cancelled);
            require(!cascade.error.empty(), "An explicit VAE must stay inside its package");
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
