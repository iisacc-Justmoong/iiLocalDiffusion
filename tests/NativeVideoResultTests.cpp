#include <Generation/NativeVideo.hpp>
#include <Generation/NativeVideoBridge.hpp>
#include <Generation/NativeCachePolicy.hpp>
#include <Generation/NativeEngineAccess.hpp>
#include <ModelPackaging/ModelPackaging.hpp>
#include <iiFileProviderNative.h>
#include <stable-diffusion.h>
#include <json-c/json.h>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <chrono>

namespace {
int loaded = 0, destroyed = 0, freed = 0, generated = 0, failure = 0;
bool supports = true, continued = false;
sd_progress_stage_cb_t stageCallback = nullptr;
void *stageUser = nullptr;
sd_graph_eval_callback_t evalCallback = nullptr;
void *evalUser = nullptr;
sd_log_cb_t logCallback = nullptr;
void *logUser = nullptr;
std::string backend, paramsBackend, encoder;
bool prefetchDisabled = false, tiledDecode = false;
std::timed_mutex engineMutex;
void require(bool condition, const char *message) { if (!condition) throw std::runtime_error(message); }
std::string safetensors(const char *name, const std::string &payload) {
    auto *header = json_object_new_object();
    auto *tensor = json_object_new_object();
    auto *shape = json_object_new_array();
    auto *offsets = json_object_new_array();
    json_object_array_add(shape, json_object_new_int64(static_cast<int64_t>(payload.size())));
    json_object_array_add(offsets, json_object_new_int(0));
    json_object_array_add(offsets, json_object_new_int64(static_cast<int64_t>(payload.size())));
    json_object_object_add(tensor, "dtype", json_object_new_string("U8"));
    json_object_object_add(tensor, "shape", shape);
    json_object_object_add(tensor, "data_offsets", offsets);
    json_object_object_add(header, name, tensor);
    const std::string encoded = json_object_to_json_string_ext(header, JSON_C_TO_STRING_PLAIN);
    json_object_put(header);
    std::string result(8, '\0');
    for (unsigned i = 0; i < 8; ++i) result[i] = static_cast<char>((encoded.size() >> (8 * i)) & 255);
    return result + encoded + payload;
}
std::string ggufFixture() {
    std::string result = "GGUF";
    const auto integer = [&](std::uint64_t value, unsigned count) {
        for (unsigned i = 0; i < count; ++i) result.push_back(static_cast<char>((value >> (8 * i)) & 255));
    };
    const auto text = [&](const std::string &value) { integer(value.size(), 8); result += value; };
    integer(3, 4); integer(1, 8); integer(0, 8);
    text("token_embd.weight"); integer(1, 4); integer(1, 8); integer(0, 4); integer(0, 8);
    result.append((32 - result.size() % 32) % 32, '\0');
    integer(0x3e800000, 4);
    return result;
}
}
struct sd_ctx_t {};
extern "C" {
void sd_ctx_params_init(sd_ctx_params_t *value) { *value = {}; }
void sd_vid_gen_params_init(sd_vid_gen_params_t *value) { *value = {}; }
sd_ctx_t *new_sd_ctx(const sd_ctx_params_t *value) {
    ++loaded; backend = value->backend ? value->backend : "";
    paramsBackend = value->params_backend ? value->params_backend : "";
    prefetchDisabled = value->disable_prefetch;
    encoder = value->llm_path ? value->llm_path : "";
    return new sd_ctx_t;
}
void free_sd_ctx(sd_ctx_t *value) { ++destroyed; delete value; }
bool sd_ctx_supports_video_generation(const sd_ctx_t *) { return supports; }
int32_t sd_get_num_physical_cores() { return 2; }
void sd_set_abort_callback(sd_abort_cb_t, void *) {}
void sd_set_log_callback(sd_log_cb_t callback, void *data) { logCallback = callback; logUser = data; }
void sd_set_progress_stage_callback(sd_progress_stage_cb_t callback, void *data) { stageCallback = callback; stageUser = data; }
void sd_set_backend_eval_callback(sd_graph_eval_callback_t callback, void *data) { evalCallback = callback; evalUser = data; }
sd_type_t str_to_sd_type(const char *type) {
    return std::string(type) == "f16" ? SD_TYPE_F16 : std::string(type) == "bf16" ? SD_TYPE_BF16 : SD_TYPE_F32;
}
bool generate_video(sd_ctx_t *, const sd_vid_gen_params_t *parameters, sd_image_t **output, int *count, sd_audio_t **) {
    ++generated;
    tiledDecode = parameters->vae_tiling_params.enabled;
    if (generated > 1 && parameters->init_image.data) continued = parameters->init_image.data[0] == 42;
    require(parameters->sample_params.sample_steps == 2, "Sampling steps must reach native engine");
    require(parameters->sample_params.guidance.txt_cfg == 3, "Guidance must reach native engine");
    if (evalCallback) evalCallback(nullptr, false, evalUser);
    if (stageCallback) stageCallback(SD_PROGRESS_SAMPLE, 1, 2, 0, stageUser);
    if (logCallback && failure == 1) {
        const std::string oversized(3000, 'w');
        logCallback(SD_LOG_WARN, oversized.c_str(), logUser);
    }
    if (logCallback && failure != 4) logCallback(SD_LOG_WARN, "model manager cannot make enough memory available on Metal", logUser);
    if (failure == 3 && backend != "cpu") return false;
    if (failure == 4 || failure == 5) return false;
    if (failure == 1) {
        if (logCallback) {
            const std::string oversized(3000, 'e');
            logCallback(SD_LOG_ERROR, oversized.c_str(), logUser);
            logCallback(SD_LOG_ERROR, "segment weight preparation failed", logUser);
        }
        return false;
    }
    *count = parameters->video_frames;
    *output = static_cast<sd_image_t *>(std::calloc(static_cast<std::size_t>(*count), sizeof(sd_image_t)));
    for (int i = 0; i < *count; ++i) {
        auto &frame = (*output)[i];
        frame.width = static_cast<unsigned>(parameters->width); frame.height = static_cast<unsigned>(parameters->height);
        frame.channel = failure == 2 ? 4 : 3;
        frame.data = static_cast<std::uint8_t *>(std::malloc(static_cast<std::size_t>(frame.width) * frame.height * 3));
        std::memset(frame.data, 42, static_cast<std::size_t>(frame.width) * frame.height * 3);
    }
    return true;
}
void free_sd_images(sd_image_t *images, int count) {
    ++freed;
    for (int i = 0; i < count; ++i) std::free(images[i].data);
    std::free(images);
}
}
namespace iiLocalDiffusion::native_detail {
std::timed_mutex &nativeEngineMutex() { return engineMutex; }
void nativeEngineOnThread(bool) {}
void releaseImageCacheForVideoLocked() {}
ResourceLimits resourceLimits() { return {8ull << 30, 4ull << 30, 6ull << 30, true}; }
}
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    try {
        namespace fs = std::filesystem;
        using iiFileProvider::native::File;
        const auto root = fs::absolute(argv[1]); File::createDirectories(root);
        const auto model = root / "model.safetensors", textEncoder = root / "encoder.gguf";
        File::write(model, "fixture model"); File::write(textEncoder, "fixture encoder");
        iiLocalDiffusion::NativeVideoRequest request;
        request.modelPath = model; request.textEncoderPath = textEncoder;
        request.width = request.height = 32; request.steps = 2; request.cpu = true;
        request.shots = {{"First shot", "noise", 1, 9, 24}, {"Second shot", "noise", 2, 9, 24}};
        request.shots[1].continuePrevious = true;
        std::atomic_bool cancelled{false}; int delivered = 0;
        const auto observer = [&](std::size_t, int, int width, int height, std::span<const std::uint8_t> rgb) {
            require(width == 32 && height == 32 && rgb.size() == 32 * 32 * 3 && rgb.front() == 42, "RGB frame callback contract"); ++delivered;
        };
        std::vector<int> batches;
        auto result = iiLocalDiffusion::generateNativeVideo(request, cancelled, [&](const auto &event) {
            if (event.stage == iiLocalDiffusion::NativeGenerationStage::Computing) batches.push_back(event.step);
        }, observer);
        require(result.error.empty() && result.frameCounts == std::vector<int>{9, 9}, "All shots must complete");
        require(delivered == 18 && loaded == 1 && destroyed == 1 && freed == 2, "One context per batch; every output is freed");
        require(continued && backend == "cpu" && paramsBackend == "cpu" && encoder == textEncoder.string(),
            "Continuity, CPU and explicit encoder bindings");
        require(batches == std::vector<int>{1, 2} && !evalCallback, "Only completed graph batches report progress; callback is cleared");
        const auto legacy = root / "legacy.onnx";
        File::write(legacy, "legacy model");
        auto rejected = request; rejected.modelPath = legacy;
        result = iiLocalDiffusion::generateNativeVideo(rejected, cancelled, {}, observer);
        require(result.error.find("ONNX") != std::string::npos && loaded == 1,
            "Video model format must be rejected before native execution");
        rejected = request; rejected.textEncoderPath = legacy;
        result = iiLocalDiffusion::generateNativeVideo(rejected, cancelled, {}, observer);
        require(result.error.find("ONNX") != std::string::npos && loaded == 1,
            "Text encoder format must be rejected before native execution");
        const auto upper = root / "legacy.ONNX";
        File::write(upper, "legacy model");
        rejected = request; rejected.textEncoderPath = upper;
        result = iiLocalDiffusion::generateNativeVideo(rejected, cancelled, {}, observer);
        require(result.error.find("ONNX") != std::string::npos && loaded == 1,
            "ONNX rejection must be case insensitive");
        const auto input = root / "package-input";
        File::createDirectories(input);
        File::write(input / "main.safetensors", safetensors("model.diffusion_model.fixture", "fixture model"));
        File::write(input / "text_encoder.safetensors", safetensors("text_model.encoder.weight", "fixture encoder"));
        const auto package = root / ("package-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".safetensors");
        (void)iild::createModelPackage(input, package);
        auto embedded = request; embedded.modelPath = package; embedded.textEncoderPath.clear();
        embedded.assetCacheDirectory = root / "assets";
        for (int attempt = 0; attempt < 2; ++attempt) {
            std::vector<std::pair<int, int>> preparation;
            result = iiLocalDiffusion::generateNativeVideo(embedded, cancelled, [&](const auto &event) {
                if (event.stage == iiLocalDiffusion::NativeGenerationStage::Preparing)
                    preparation.emplace_back(event.step, event.total);
            }, observer);
            require(result.error.empty() && !preparation.empty(), "Embedded asset prepares and reuses its verified cache");
            require(fs::path(encoder).extension() == ".safetensors" && fs::is_regular_file(encoder),
                "The native engine receives a materialized safetensors encoder");
            require(preparation.front().second == 1 && preparation.back() == std::pair{1, 1},
                "Preparation and cached verification report a nonzero total and final completion");
        }
        const auto ggufInput = root / "gguf-package-input";
        File::createDirectories(ggufInput);
        File::write(ggufInput / "main.safetensors", safetensors("model.diffusion_model.fixture", "fixture model"));
        const auto ggufBytes = ggufFixture();
        File::write(ggufInput / "text_encoder.gguf", ggufBytes);
        const auto ggufPackage = root / ("gguf-package-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".safetensors");
        (void)iild::createModelPackage(ggufInput, ggufPackage);
        embedded.modelPath = ggufPackage;
        result = iiLocalDiffusion::generateNativeVideo(embedded, cancelled, {}, observer);
        require(result.error.empty() && fs::path(encoder).extension() == ".gguf" && File::read(encoder) == ggufBytes,
            "The native engine receives the original byte-exact GGUF encoder");
        failure = 2;
        result = iiLocalDiffusion::generateNativeVideo(request, cancelled, {}, observer);
        require(result.error.find("invalid RGB") != std::string::npos && loaded == destroyed, "Invalid native output cannot be published");
        failure = 1;
        result = iiLocalDiffusion::generateNativeVideo(request, cancelled, {}, observer);
        require(!result.error.empty() && loaded == destroyed, "Inference failure releases context");
        require(result.error.find("cannot make enough memory available") != std::string::npos &&
            result.error.find("segment weight preparation failed") != std::string::npos,
            "Video failures must retain bounded warning and error evidence");
        require(result.error.size() < 4000, "Warning and error evidence must remain bounded");
        failure = 0; supports = false;
        result = iiLocalDiffusion::generateNativeVideo(request, cancelled, {}, observer);
        require(result.error.find("not supported") != std::string::npos, "Manual classification cannot turn an image engine into video");
        supports = true;
        result = iiLocalDiffusion::generateNativeVideo(request, cancelled, [&](const auto &event) {
            if (event.stage == iiLocalDiffusion::NativeGenerationStage::Denoising) cancelled = true;
        }, observer);
        require(result.cancelled && result.error.empty() && loaded == destroyed && !stageCallback && !logCallback,
            "Cancellation clears callbacks and owns no stale context");
        cancelled = false;
        result = iiLocalDiffusion::generateNativeVideo(request, cancelled, {}, [&](std::size_t, int index, int, int, auto) {
            if (index == 8) cancelled = true;
        });
        require(result.cancelled && result.frameCounts.empty() && loaded == destroyed, "Cancellation on the final frame prevents shot publication");
        cancelled = false;
        auto placement = request; placement.cpu = false; placement.cpuTextEncoding = true; placement.backend = "auto";
        result = iiLocalDiffusion::generateNativeVideo(placement, cancelled, {}, observer);
        require(result.error.empty() && backend == "te=cpu" && paramsBackend == "diffusion=disk",
            "CPU text encoding must leave GPU diffusion weights reloadable when auto-fit is disabled");
        placement.cpuTextEncoding = false; placement.backend = "metal";
        result = iiLocalDiffusion::generateNativeVideo(placement, cancelled, {}, observer);
        require(result.error.empty() && backend == "metal" && paramsBackend == "diffusion=disk",
            "Explicit GPU placement must leave diffusion weights reloadable");
        placement.backend = "cpu";
        result = iiLocalDiffusion::generateNativeVideo(placement, cancelled, {}, observer);
        require(result.error.empty() && backend == "cpu" && paramsBackend.empty(),
            "A CPU backend request must not use GPU residency policy");
        placement.backend = "auto";
        result = iiLocalDiffusion::generateNativeVideo(placement, cancelled, {}, observer);
        require(result.error.empty() && backend.empty() && paramsBackend.empty(),
            "Automatic placement without overrides must keep the engine auto-fit contract");
        auto lowVram = placement; lowVram.vaeTiling = false; lowVram.shots.resize(1);
        result = iiLocalDiffusion::generateNativeVideoLowVram(lowVram, cancelled, {}, observer);
        require(result.error.empty() && backend == "te=cpu,vae=cpu" &&
                    paramsBackend == "te=disk,diffusion=disk,vae=disk" && prefetchDisabled && tiledDecode,
            "Low VRAM must offload text, VAE and weights, disable prefetch and enable tiled decode");
        lowVram.backend = "metal";
        result = iiLocalDiffusion::generateNativeVideoLowVram(lowVram, cancelled, {}, observer);
        require(result.error.empty() && backend == "te=cpu,diffusion=metal,vae=cpu" && prefetchDisabled,
            "Low VRAM must preserve an explicitly selected compute device");
        lowVram.cpu = true;
        result = iiLocalDiffusion::generateNativeVideoLowVram(lowVram, cancelled, {}, observer);
        require(result.error.empty() && backend == "cpu" && paramsBackend == "te=disk,diffusion=disk,vae=disk",
            "Low VRAM must preserve an explicit CPU request");
#if !defined(_WIN32)
        // POSIX test files are sparse; do not allocate a multi-GiB fixture on
        // platforms where resize_file may reserve the entire extent.
        const auto largeModel = root / "large-sparse-fixture.safetensors";
        File::write(largeModel, "fixture model");
        fs::resize_file(largeModel, (4ull << 30) + 1);
        auto large = placement; large.modelPath = largeModel; large.shots.resize(1);
        result = iiLocalDiffusion::generateNativeVideo(large, cancelled, {}, observer);
        require(result.error.empty() && backend.empty() && paramsBackend.empty(),
            "Large fully automatic requests must preserve engine auto-fit");
        large.cpuTextEncoding = true;
        result = iiLocalDiffusion::generateNativeVideo(large, cancelled, {}, observer);
        require(result.error.empty() && backend == "te=cpu" && paramsBackend == "te=disk,diffusion=disk",
            "A large explicitly placed video checkpoint must use disk-backed weights");
        large.cpu = true;
        result = iiLocalDiffusion::generateNativeVideo(large, cancelled, {}, observer);
        require(result.error.empty() && backend == "cpu" && paramsBackend == "te=disk,diffusion=disk",
            "A large CPU video checkpoint must stream its text and diffusion weights");
        fs::remove(largeModel);
#endif
        failure = 3;
        const auto attemptsBefore = generated;
        placement.shots.resize(1);
        int fallbackBatches = 0;
        result = iiLocalDiffusion::generateNativeVideo(placement, cancelled, [&](const auto &event) {
            if (event.stage == iiLocalDiffusion::NativeGenerationStage::Computing) ++fallbackBatches;
        }, observer);
        require(result.error.empty() && generated == attemptsBefore + 2 && backend == "cpu" &&
                    paramsBackend == "te=disk,diffusion=disk" && loaded == destroyed && fallbackBatches == 1 && !evalCallback,
            "Automatic GPU budget failure must retry with CPU compute and disk-backed diffusion weights");
        const auto otherFailure = generated;
        failure = 4;
        result = iiLocalDiffusion::generateNativeVideo(placement, cancelled, {}, observer);
        require(!result.error.empty() && generated == otherFailure + 1,
            "Unrelated failures must not trigger a disk-backed retry");
        const auto explicitFailure = generated;
        failure = 3; placement.backend = "metal";
        result = iiLocalDiffusion::generateNativeVideo(placement, cancelled, {}, observer);
        require(!result.error.empty() && generated == explicitFailure + 1 && backend == "metal",
            "Explicit GPU selection must not silently change to CPU");
        placement.backend = "auto"; failure = 5;
        result = iiLocalDiffusion::generateNativeVideo(placement, cancelled, {}, observer);
        require(result.error.find("Initial GPU failure") != std::string::npos &&
                    result.error.find("Fallback failure") != std::string::npos && loaded == destroyed,
            "Both attempts must be reported when the disk-backed fallback also fails");
        failure = 0;
        const auto modelPath = model.string(), encoderPath = textEncoder.string();
        iild_native_video_shot_v1 abiShot{};
        abiShot.prompt = "ABI shot"; abiShot.negative_prompt = "noise";
        abiShot.frames = 9; abiShot.fps = 24; abiShot.conditioning_strength = 1;
        iild_native_video_request_v1 abiRequest{};
        abiRequest.size = sizeof(abiRequest); abiRequest.model = modelPath.c_str();
        abiRequest.text_encoder = encoderPath.c_str(); abiRequest.backend = "auto";
        abiRequest.precision = "auto"; abiRequest.width = abiRequest.height = 32;
        abiRequest.steps = 2; abiRequest.guidance_scale = 3;
        abiRequest.shots = &abiShot; abiRequest.shot_count = 1;
        const auto abiFrame = [](size_t, int32_t, int32_t, int32_t, const uint8_t*, size_t, void*) { return 0; };
        auto *abi = iild_native_video_generate_low_vram_v1(&abiRequest, nullptr, abiFrame, nullptr);
        require(abi && std::string(iild_native_video_metadata_v1(abi)).find("low-vram") != std::string::npos &&
                    backend == "te=cpu,vae=cpu" && paramsBackend == "te=disk,diffusion=disk,vae=disk" && prefetchDisabled,
            "Low VRAM C ABI must reach the real video adapter without changing the request layout");
        iild_native_video_free_v1(abi);
        abi = iild_native_video_generate_v1(&abiRequest, nullptr, abiFrame, nullptr);
        require(abi && std::string(iild_native_video_metadata_v1(abi)).find("standard") != std::string::npos &&
                    backend.empty() && paramsBackend.empty() && !prefetchDisabled,
            "Existing C ABI must preserve standard placement");
        iild_native_video_free_v1(abi);
        auto *invalid = iild_native_video_generate_v1(nullptr, nullptr, nullptr, nullptr);
        require(invalid && std::string(iild_native_video_metadata_v1(invalid)).find("Invalid native video ABI") != std::string::npos, "C ABI rejects invalid request");
        iild_native_video_free_v1(invalid);
        invalid = iild_native_video_generate_low_vram_v1(nullptr, nullptr, nullptr, nullptr);
        require(invalid && std::string(iild_native_video_metadata_v1(invalid)).find("low-vram") != std::string::npos,
            "Low VRAM C ABI must report its execution mode even for invalid requests");
        iild_native_video_free_v1(invalid);
        std::cout << "Native video frames, batch, continuity, cleanup, cancellation and ABI checks passed\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
