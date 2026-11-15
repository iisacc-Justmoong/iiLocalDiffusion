#include "NativeImageBridge.hpp"
#include "NativeDiffusion.hpp"
#include "NativeUpscalerBridgeInternal.hpp"
#include "NativeTelemetry.hpp"
#include <json-c/json.h>
#include <memory>
#include <limits>
#include <stdexcept>

struct iild_native_result_v1 {
    iiLocalDiffusion::NativeGenerationResult image;
    iiLocalDiffusion::NativeHiresUpscalerReport hires;
    std::string metadata;
};

static iild_native_result_v1 *generate(const iild_native_request_v1 *input,
    iild_native_progress_v1 callback, iild_native_preview_v1 previewCallback, void *user,
    const iild_native_request_v2 *extended = nullptr, const iild_native_request_v3 *controlled = nullptr,
    const iild_native_hires_request_v1 *hires = nullptr)
{
    try {
        iiLocalDiffusion::native_detail::lastNativeTrace().path.clear();
        iiLocalDiffusion::native_detail::lastNativeTrace().elapsedMs = 0;
        auto result = std::make_unique<iild_native_result_v1>();
        try {
            if (!input || input->size != sizeof(*input) || !input->model || !input->prompt
                || input->lora_count > 128 || (input->lora_count && !input->loras)
                || input->timeout_milliseconds < 0)
                throw std::invalid_argument("Invalid native bridge request or ABI version.");
            if (extended && (extended->size != sizeof(*extended) || extended->hires < 0 || extended->hires > 1
                             || extended->cpu < 0 || extended->cpu > 1 || extended->prediction < 0 || extended->prediction > 2))
                throw std::invalid_argument("Invalid native V2 request or ABI version.");
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
            const iiLocalDiffusion::NativePreviewCallback preview = previewCallback ? iiLocalDiffusion::NativePreviewCallback([&](const auto &frame) {
                if (previewCallback(frame.sequence, frame.step, frame.total, frame.width, frame.height,
                    frame.rgb.data(), frame.rgb.size(), user)) cancelled = true;
            }) : iiLocalDiffusion::NativePreviewCallback{};
            if (extended) {
                iiLocalDiffusion::NativeModelComponents components;
                if (extended->clip_l) components.clipL = extended->clip_l;
                if (extended->clip_g) components.clipG = extended->clip_g;
                if (extended->t5xxl) components.t5xxl = extended->t5xxl;
                if (extended->llm) components.llm = extended->llm;
                if (extended->vae) components.vae = extended->vae;
                components.guidanceScale = extended->guidance_scale;
                components.distilledGuidance = extended->distilled_guidance;
                components.hires = extended->hires != 0;
                components.prediction = static_cast<iiLocalDiffusion::NativePrediction>(extended->prediction);
                iiLocalDiffusion::NativeSamplingControls sampling;
                if (controlled) {
                    if (controlled->sampler < 0 || controlled->sampler > 2 || controlled->sigma_count > 1001
                        || (controlled->sigma_count && !controlled->sigmas))
                        throw std::invalid_argument("Invalid native V3 sampling controls.");
                    sampling.sampler = static_cast<iiLocalDiffusion::NativeSampler>(controlled->sampler);
                    sampling.flowShift = controlled->flow_shift;
                    if (controlled->sigma_count)
                        sampling.customSigmas.assign(controlled->sigmas, controlled->sigmas + controlled->sigma_count);
                }
                const auto backend = extended->cpu ? iiLocalDiffusion::NativeComputeBackend::Cpu : iiLocalDiffusion::NativeComputeBackend::Automatic;
                if (hires) {
                    if (!components.hires || input->prepare_only)
                        throw std::invalid_argument("Hires provider requires generation with Hires enabled.");
                    const auto &configured=hires->upscaler;
                    const auto &selection=configured.upscaler;
                    auto provider=iiLocalDiffusion::native_detail::makeBridgeUpscaler(selection.image, &selection, &configured);
                    auto upscale=iiLocalDiffusion::native_detail::bridgeUpscaleOptions(selection.image);
                    result->image = iiLocalDiffusion::generateNativeResidentImageWithUpscaler(request, options, components,
                        sampling, hires->denoise_strength, *provider, upscale, backend, cancelled, &result->hires, progress, preview);
                } else result->image = iiLocalDiffusion::generateNativeImageWithResidentWeights(request, options, components, sampling,
                    backend, cancelled, input->prepare_only != 0, progress, preview);
            } else result->image = input->prepare_only
                ? iiLocalDiffusion::prepareNativeImageModelWithResidentWeights(request, options, cancelled, progress)
                : iiLocalDiffusion::generateNativeImageWithResidentWeights(request, options, iiLocalDiffusion::NativeComputeBackend::Automatic,
                    cancelled, progress, preview);
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
        field("weight_storage", json_object_new_string("anonymous"));
        field("weight_lifetime", json_object_new_string("runtime"));
        field("telemetry_path", json_object_new_string(iiLocalDiffusion::native_detail::lastNativeTrace().path.c_str()));
        field("telemetry_elapsed_ms", json_object_new_double(iiLocalDiffusion::native_detail::lastNativeTrace().elapsedMs));
        if (hires) {
            auto *record=json_object_new_object();
            json_object_object_add(record,"upscaler",json_object_new_string(result->hires.upscaler.c_str()));
            json_object_object_add(record,"applied_scale",json_object_new_double(result->hires.appliedScale));
            json_object_object_add(record,"target_width",json_object_new_int64(result->hires.targetWidth));
            json_object_object_add(record,"target_height",json_object_new_int64(result->hires.targetHeight));
            json_object_object_add(record,"resampled",json_object_new_boolean(result->hires.resampled));
            auto *passes=json_object_new_array();
            for (const auto &pass : result->hires.passes) {
                auto *item=json_object_new_object();
                json_object_object_add(item,"width",json_object_new_int64(pass.width));
                json_object_object_add(item,"height",json_object_new_int64(pass.height));
                json_object_array_add(passes,item);
            }
            json_object_object_add(record,"passes",passes);
            auto *details=json_object_new_object();
            for (const auto &[key,item] : result->hires.details)
                json_object_object_add(details,key.c_str(),json_object_new_string(item.c_str()));
            json_object_object_add(record,"details",details);
            field("hires_upscaler",record);
        }
        result->metadata = json_object_to_json_string_ext(json.get(), JSON_C_TO_STRING_PLAIN);
        return result.release();
    } catch (...) { return nullptr; } // No C++ exception may cross the C boundary.
}
extern "C" {
int iild_native_available_v1(void) { return iiLocalDiffusion::nativeDiffusionAvailable(); }
iild_native_result_v1 *iild_native_generate_v1(const iild_native_request_v1 *input,
    iild_native_progress_v1 callback, void *user) { return generate(input, callback, nullptr, user); }
iild_native_result_v1 *iild_native_generate_with_preview_v1(const iild_native_request_v1 *input,
    iild_native_progress_v1 callback, iild_native_preview_v1 preview, void *user) {
    return generate(input, callback, preview, user);
}
iild_native_result_v1 *iild_native_generate_v2(const iild_native_request_v2 *input,
    iild_native_progress_v1 callback, iild_native_preview_v1 preview, void *user) {
    // Check the outer size before dereferencing any trailing fields.
    if (!input || input->size != sizeof(*input)) return generate(nullptr, callback, preview, user);
    return generate(&input->image, callback, preview, user, input);
}
iild_native_result_v1 *iild_native_generate_v3(const iild_native_request_v3 *input,
    iild_native_progress_v1 callback, iild_native_preview_v1 preview, void *user) {
    if (!input || input->size != sizeof(*input)) return generate(nullptr, callback, preview, user);
    return generate(&input->image.image, callback, preview, user, &input->image, input);
}
iild_native_result_v1 *iild_native_generate_hires_v1(const iild_native_hires_request_v1 *input,
    iild_native_progress_v1 callback, iild_native_preview_v1 preview, void *user) {
    if (!input || input->size != sizeof(*input) || input->image.size != sizeof(input->image))
        return generate(nullptr, callback, preview, user);
    return generate(&input->image.image.image, callback, preview, user, &input->image.image, &input->image, input);
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
