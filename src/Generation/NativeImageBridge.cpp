#include "NativeImageBridge.hpp"
#include "NativeDiffusion.hpp"
#include <json-c/json.h>
#include <memory>
#include <limits>
#include <stdexcept>

struct iild_native_result_v1 {
    iiLocalDiffusion::NativeGenerationResult image;
    std::string metadata;
};

extern "C" {
int iild_native_available_v1(void) { return iiLocalDiffusion::nativeDiffusionAvailable(); }
iild_native_result_v1 *iild_native_generate_v1(const iild_native_request_v1 *input,
    iild_native_progress_v1 callback, void *user)
{
    return iild_native_generate_with_preview_v1(input, callback, nullptr, user);
}
iild_native_result_v1 *iild_native_generate_with_preview_v1(const iild_native_request_v1 *input,
    iild_native_progress_v1 callback, iild_native_preview_v1 previewCallback, void *user)
{
    try {
        auto result = std::make_unique<iild_native_result_v1>();
        try {
            if (!input || input->size != sizeof(*input) || !input->model || !input->prompt
                || input->lora_count > 128 || (input->lora_count && !input->loras)
                || input->timeout_milliseconds < 0)
                throw std::invalid_argument("Invalid native bridge request or ABI version.");
            iiLocalDiffusion::NativeGenerationRequest request;
            request.modelPath = input->model;
            request.prompt = input->prompt;
            request.width = input->width; request.height = input->height; request.steps = input->steps;
            request.seed = input->seed;
            // Desktop workers have no 15-minute wall-clock cap. Keep their
            // existing lifecycle/cancellation policy; explicit C callers can
            // still choose a finite deadline. Do not inherit the mobile API default.
            request.timeoutMilliseconds = input->timeout_milliseconds > 0
                ? input->timeout_milliseconds : std::numeric_limits<int>::max();
            iiLocalDiffusion::NativeGenerationOptions options;
            if (input->negative_prompt) options.negativePrompt = input->negative_prompt;
            if (input->resources) options.resourceDirectory = input->resources;
            options.defaultModifiers = input->default_modifiers != 0;
            for (size_t i = 0; i < input->lora_count; ++i) {
                if (!input->loras[i].path) throw std::invalid_argument("Missing native LoRA path.");
                options.loras.push_back({input->loras[i].path, input->loras[i].strength});
            }
            std::atomic_bool cancelled{false};
            const auto progress = [&](const iiLocalDiffusion::NativeGenerationProgress &event) {
                if (callback && callback(static_cast<int>(event.stage), event.step, event.total, user)) cancelled = true;
            };
            result->image = input->prepare_only
                ? iiLocalDiffusion::prepareNativeImageModel(request, options, cancelled, progress)
                : iiLocalDiffusion::generateNativeImageWithPreview(request, options, iiLocalDiffusion::NativeComputeBackend::Automatic,
                    cancelled, progress, previewCallback ? iiLocalDiffusion::NativePreviewCallback([&](const auto &frame) {
                        if (previewCallback(frame.sequence, frame.step, frame.total, frame.width, frame.height,
                            frame.rgb.data(), frame.rgb.size(), user)) cancelled = true;
                    }) : iiLocalDiffusion::NativePreviewCallback{});
        } catch (const std::exception &error) { result->image.error = error.what(); }
        const auto &value = result->image;
        const std::unique_ptr<json_object, decltype(&json_object_put)> json(json_object_new_object(), json_object_put);
        if (!json) return nullptr;
        const auto field = [&](const char *name, json_object *member) { json_object_object_add(json.get(), name, member); };
        field("error", json_object_new_string(value.error.c_str()));
        field("cancelled", json_object_new_boolean(value.cancelled));
        field("width", json_object_new_int(value.width)); field("height", json_object_new_int(value.height));
        field("model_cache_hit", json_object_new_boolean(value.modelCacheHit));
        field("model_load_ms", json_object_new_double(value.modelLoadMilliseconds));
        field("generation_ms", json_object_new_double(value.generationMilliseconds));
        field("memory_budget_bytes", json_object_new_uint64(value.memoryBudgetBytes));
        field("threads", json_object_new_int(value.threads));
        result->metadata = json_object_to_json_string_ext(json.get(), JSON_C_TO_STRING_PLAIN);
        return result.release();
    } catch (...) { return nullptr; } // No C++ exception may cross the C boundary.
}
const char *iild_native_metadata_v1(const iild_native_result_v1 *result) {
    return result ? result->metadata.c_str() : nullptr;
}
const uint8_t *iild_native_rgb_v1(const iild_native_result_v1 *result, size_t *size) {
    if (size) *size = result ? result->image.rgb.size() : 0;
    return result && !result->image.rgb.empty() ? result->image.rgb.data() : nullptr;
}
void iild_native_free_v1(iild_native_result_v1 *result) { delete result; }
void iild_native_release_v1(void) { iiLocalDiffusion::releaseNativeDiffusionCache(); }
}
