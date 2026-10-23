#include <Generation/NativeVideo.hpp>
#include <Generation/NativeVideoBridge.hpp>
#include <Generation/NativeCachePolicy.hpp>
#include <Generation/NativeEngineAccess.hpp>
#include <iiFileProviderNative.h>
#include <stable-diffusion.h>
#include <json-c/json.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace {
int loaded = 0, destroyed = 0, freed = 0, generated = 0, failure = 0;
bool supports = true, continued = false;
sd_progress_stage_cb_t stageCallback = nullptr;
void *stageUser = nullptr;
sd_graph_eval_callback_t evalCallback = nullptr;
void *evalUser = nullptr;
std::string backend, encoder;
std::timed_mutex engineMutex;
void require(bool condition, const char *message) { if (!condition) throw std::runtime_error(message); }
}
struct sd_ctx_t {};
extern "C" {
void sd_ctx_params_init(sd_ctx_params_t *value) { *value = {}; }
void sd_vid_gen_params_init(sd_vid_gen_params_t *value) { *value = {}; }
sd_ctx_t *new_sd_ctx(const sd_ctx_params_t *value) {
    ++loaded; backend = value->backend ? value->backend : "";
    encoder = value->llm_path ? value->llm_path : "";
    return new sd_ctx_t;
}
void free_sd_ctx(sd_ctx_t *value) { ++destroyed; delete value; }
bool sd_ctx_supports_video_generation(const sd_ctx_t *) { return supports; }
int32_t sd_get_num_physical_cores() { return 2; }
void sd_set_abort_callback(sd_abort_cb_t, void *) {}
void sd_set_log_callback(sd_log_cb_t, void *) {}
void sd_set_progress_stage_callback(sd_progress_stage_cb_t callback, void *data) { stageCallback = callback; stageUser = data; }
void sd_set_backend_eval_callback(sd_graph_eval_callback_t callback, void *data) { evalCallback = callback; evalUser = data; }
sd_type_t str_to_sd_type(const char *type) {
    return std::string(type) == "f16" ? SD_TYPE_F16 : std::string(type) == "bf16" ? SD_TYPE_BF16 : SD_TYPE_F32;
}
bool generate_video(sd_ctx_t *, const sd_vid_gen_params_t *parameters, sd_image_t **output, int *count, sd_audio_t **) {
    ++generated;
    if (generated > 1 && parameters->init_image.data) continued = parameters->init_image.data[0] == 42;
    require(parameters->sample_params.sample_steps == 2, "Sampling steps must reach native engine");
    require(parameters->sample_params.guidance.txt_cfg == 3, "Guidance must reach native engine");
    if (evalCallback) evalCallback(nullptr, false, evalUser);
    if (stageCallback) stageCallback(SD_PROGRESS_SAMPLE, 1, 2, 0, stageUser);
    if (failure == 1) return false;
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
        require(continued && backend == "cpu" && encoder == textEncoder.string(), "Continuity, CPU and explicit encoder bindings");
        require(batches == std::vector<int>{1, 2} && !evalCallback, "Only completed graph batches report progress; callback is cleared");
        const char *manifest = R"({"schema":"iild-safetensors-package-v1","files":[{"status":"included","role":"text_encoder","format":"gguf","size":15,"sha256":"bbb3b575693841138a69bdfd31133e6a5cc17813acad6cb0c4f94b93accc2cb7","tensors":[{"output_key":"__iild_assets__/text_encoder/fixture","begin":0,"end":15,"sha256":"bbb3b575693841138a69bdfd31133e6a5cc17813acad6cb0c4f94b93accc2cb7"}]}]})";
        auto *header = json_tokener_parse(R"({"__iild_assets__/text_encoder/fixture":{"dtype":"U8","shape":[15],"data_offsets":[0,15]}})");
        auto *metadata = json_object_new_object();
        json_object_object_add(metadata, "iild_package_schema", json_object_new_string("iild-safetensors-package-v1"));
        json_object_object_add(metadata, "iild_package_manifest", json_object_new_string(manifest));
        json_object_object_add(header, "__metadata__", metadata);
        const std::string encoded = json_object_to_json_string_ext(header, JSON_C_TO_STRING_PLAIN);
        json_object_put(header);
        std::string packaged(8, '\0');
        for (unsigned i = 0; i < 8; ++i) packaged[i] = static_cast<char>((encoded.size() >> (8 * i)) & 255);
        packaged += encoded + "fixture encoder";
        const auto package = root / "package.safetensors";
        File::write(package, packaged);
        auto embedded = request; embedded.modelPath = package; embedded.textEncoderPath.clear();
        embedded.assetCacheDirectory = root / "assets";
        for (int attempt = 0; attempt < 2; ++attempt) {
            std::vector<std::pair<int, int>> preparation;
            result = iiLocalDiffusion::generateNativeVideo(embedded, cancelled, [&](const auto &event) {
                if (event.stage == iiLocalDiffusion::NativeGenerationStage::Preparing)
                    preparation.emplace_back(event.step, event.total);
            }, observer);
            require(result.error.empty() && !preparation.empty(), "Embedded asset prepares and reuses its verified cache");
            require(preparation.front().second == 1 && preparation.back() == std::pair{1, 1},
                "Preparation and cached verification report a nonzero total and final completion");
        }
        failure = 2;
        result = iiLocalDiffusion::generateNativeVideo(request, cancelled, {}, observer);
        require(result.error.find("invalid RGB") != std::string::npos && loaded == destroyed, "Invalid native output cannot be published");
        failure = 1;
        result = iiLocalDiffusion::generateNativeVideo(request, cancelled, {}, observer);
        require(!result.error.empty() && loaded == destroyed, "Inference failure releases context");
        failure = 0; supports = false;
        result = iiLocalDiffusion::generateNativeVideo(request, cancelled, {}, observer);
        require(result.error.find("not supported") != std::string::npos, "Manual classification cannot turn an image engine into video");
        supports = true;
        result = iiLocalDiffusion::generateNativeVideo(request, cancelled, [&](const auto &event) {
            if (event.stage == iiLocalDiffusion::NativeGenerationStage::Denoising) cancelled = true;
        }, observer);
        require(result.cancelled && result.error.empty() && loaded == destroyed && !stageCallback, "Cancellation clears callbacks and owns no stale context");
        cancelled = false;
        result = iiLocalDiffusion::generateNativeVideo(request, cancelled, {}, [&](std::size_t, int index, int, int, auto) {
            if (index == 8) cancelled = true;
        });
        require(result.cancelled && result.frameCounts.empty() && loaded == destroyed, "Cancellation on the final frame prevents shot publication");
        auto *invalid = iild_native_video_generate_v1(nullptr, nullptr, nullptr, nullptr);
        require(invalid && std::string(iild_native_video_metadata_v1(invalid)).find("Invalid native video ABI") != std::string::npos, "C ABI rejects invalid request");
        iild_native_video_free_v1(invalid);
        std::cout << "Native video frames, batch, continuity, cleanup, cancellation and ABI checks passed\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
