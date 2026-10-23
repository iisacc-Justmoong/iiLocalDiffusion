#include "NativeVideoBridge.hpp"
#include "NativeVideo.hpp"
#include <json-c/json.h>
#include <memory>
#include <stdexcept>

struct iild_native_video_result_v1 { std::string metadata; };
namespace {
std::string text(const char *value) { return value ? value : ""; }
std::string metadata(const iiLocalDiffusion::NativeVideoResult &result) {
    std::unique_ptr<json_object, decltype(&json_object_put)> root(json_object_new_object(), json_object_put);
    json_object_object_add(root.get(), "schema", json_object_new_string("iild-native-video-v1"));
    json_object_object_add(root.get(), "error", json_object_new_string(result.error.c_str()));
    json_object_object_add(root.get(), "cancelled", json_object_new_boolean(result.cancelled));
    json_object_object_add(root.get(), "threads", json_object_new_int(result.threads));
    json_object_object_add(root.get(), "memory_budget_bytes", json_object_new_uint64(result.memoryBudgetBytes));
    auto *counts = json_object_new_array();
    for (int value : result.frameCounts) json_object_array_add(counts, json_object_new_int(value));
    json_object_object_add(root.get(), "frame_counts", counts);
    return json_object_to_json_string_ext(root.get(), JSON_C_TO_STRING_PLAIN);
}
}
extern "C" iild_native_video_result_v1 *iild_native_video_generate_v1(const iild_native_video_request_v1 *input,
    iild_native_progress_v1 progress, iild_native_video_frame_v1 frame, void *user)
{
    auto result = std::make_unique<iild_native_video_result_v1>();
    iiLocalDiffusion::NativeVideoResult output;
    try {
        if (!input || input->size != sizeof(*input) || !input->shots || !input->shot_count
            || input->shot_count > 256 || !frame || input->width < 32 || input->height < 32
            || input->width > 4096 || input->height > 4096)
            throw std::invalid_argument("Invalid native video ABI request");
        iiLocalDiffusion::NativeVideoRequest request;
        request.modelPath = text(input->model); request.textEncoderPath = text(input->text_encoder);
        request.assetCacheDirectory = text(input->asset_cache);
        request.backend = input->backend ? input->backend : "auto";
        request.precision = input->precision ? input->precision : "auto";
        request.cpuTextEncoding = input->cpu_text_encoding != 0;
        request.width = input->width; request.height = input->height; request.steps = input->steps;
        request.threads = input->threads; request.guidanceScale = input->guidance_scale;
        request.cpu = input->cpu != 0; request.vaeTiling = input->vae_tiling != 0;
        const auto bytes = static_cast<std::size_t>(request.width) * static_cast<std::size_t>(request.height) * 3;
        for (std::size_t i = 0; i < input->shot_count; ++i) {
            const auto &source = input->shots[i];
            iiLocalDiffusion::NativeVideoShot shot;
            shot.prompt = text(source.prompt); shot.negativePrompt = text(source.negative_prompt);
            shot.seed = source.seed; shot.frames = source.frames; shot.fps = source.fps;
            shot.conditioningStrength = source.conditioning_strength; shot.continuePrevious = source.continue_previous != 0;
            const auto image = [&](const std::uint8_t *rgb, std::size_t size) {
                iiLocalDiffusion::NativeReferenceImage value;
                if (size) {
                    if (!rgb || size != bytes) throw std::invalid_argument("Invalid native video keyframe buffer");
                    value.width = request.width; value.height = request.height; value.rgb.assign(rgb, rgb + size);
                }
                return value;
            };
            shot.firstFrame = image(source.first_rgb, source.first_size); shot.lastFrame = image(source.last_rgb, source.last_size);
            request.shots.push_back(std::move(shot));
        }
        std::atomic_bool cancelled{false};
        output = iiLocalDiffusion::generateNativeVideo(request, cancelled,
            [&](const auto &event) {
                if (progress && progress(static_cast<int>(event.stage), event.step, event.total, user)) cancelled = true;
            }, [&](std::size_t shot, int index, int width, int height, std::span<const std::uint8_t> rgb) {
                if (frame(shot, index, width, height, rgb.data(), rgb.size(), user)) cancelled = true;
            });
    } catch (const std::exception &error) { output.error = error.what(); }
    result->metadata = metadata(output);
    return result.release();
}
extern "C" const char *iild_native_video_metadata_v1(const iild_native_video_result_v1 *result) {
    return result ? result->metadata.c_str() : "{\"error\":\"Missing native video result\"}";
}
extern "C" void iild_native_video_free_v1(iild_native_video_result_v1 *result) { delete result; }
