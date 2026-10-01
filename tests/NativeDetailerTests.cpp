// Compile the production detector/mask/composite pipeline with only diffusion
// sampling replaced. This runs real YOLO graphs, not invented detection boxes.
#define generate_image detailer_test_generate_image
#include "detailer.cpp"
#undef generate_image
#include <filesystem>
#include <fstream>
#include <iostream>
#if defined(__APPLE__)
#include <ggml-metal.h>
#endif

namespace {
int inpaintCalls = 0;
bool abortRequested = false, failInpaint = false, cancelInpaint = false;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void writeDetector(const std::filesystem::path& path, float confidenceLogit) {
    YOLOv8Config config;
    config.out_channels.fill(4);
    for (int layer : {2, 4, 6, 8, 12, 15, 18, 21}) { config.hidden_channels[layer] = 2; config.repeats[layer] = 1; }
    config.detect_box_channels = config.detect_cls_channels = 4;
    config.reg_max = 2; config.num_classes = 1; config.valid = true;
    auto context = std::unique_ptr<ggml_context, decltype(&ggml_free)>(
        ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    require(bool(context), "Cannot allocate detector metadata");
    YOLOv8Model model(config); model.init(context.get());
    std::map<std::string, ggml_tensor*> tensors; model.get_param_tensors(tensors);
    std::ostringstream header; header << '{';
    std::vector<float> values;
    bool first = true;
    for (const auto& [name, tensor] : tensors) {
        if (!first) header << ','; first = false;
        header << '"' << name << "\":{\"dtype\":\"F32\",\"shape\":[";
        // Safetensors preserves Conv2d rank even for a one-class output. GGML
        // n_dims intentionally drops those singleton axes and cannot encode it.
        const int rank = name.ends_with(".weight") ? 4 : 1;
        for (int axis = rank - 1; axis >= 0; --axis) {
            if (axis != rank - 1) header << ',';
            header << tensor->ne[axis];
        }
        const auto offset = values.size() * sizeof(float);
        const float value = name.starts_with("model.22.cv3.") && name.ends_with(".2.bias") ? confidenceLogit : 0;
        values.insert(values.end(), size_t(ggml_nelements(tensor)), value);
        header << "],\"data_offsets\":[" << offset << ',' << values.size() * sizeof(float) << "]}";
    }
    header << '}'; auto json = header.str(); while (json.size() % 8) json += ' ';
    const uint64_t length = json.size();
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(&length), sizeof(length));
    stream.write(json.data(), json.size());
    stream.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
    require(bool(stream), "Cannot write detector fixture");
}
}
extern "C" bool detailer_test_generate_image(sd_ctx_t*, const sd_img_gen_params_t* p,
                                               sd_image_t** output, int* count) {
    ++inpaintCalls;
    require(p->init_image.data && p->mask_image.data && p->width == 64 && p->height == 64
        && p->strength == .25f && p->sample_params.sample_steps == 8 && p->sample_params.guidance.txt_cfg == 6.5f,
        "Detailer lost masked crop or sampling settings");
    require(!p->hires.enabled && !p->ref_images_count && !p->control_image.data && p->batch_count == 1,
        "Full-canvas conditioning leaked into cropped Detailer generation");
    *output = nullptr; *count = 0;
    if (failInpaint) return false;
    *output = static_cast<sd_image_t*>(calloc(1, sizeof(sd_image_t)));
    require(*output != nullptr, "Cannot allocate sampled crop container");
    **output = {64, 64, 3, static_cast<uint8_t*>(malloc(64 * 64 * 3))};
    require(*output && (*output)->data, "Cannot allocate sampled crop fixture");
    std::fill_n((*output)->data, 64 * 64 * 3, uint8_t(200));
    *count = 1;
    if (cancelInpaint) abortRequested = true;
    return true;
}
int main(int argc, char** argv) {
    try {
        require(argc == 2, "Expected build-local fixture directory");
        const std::filesystem::path directory(argv[1]); std::filesystem::create_directories(directory);
        sd_set_log_callback([](sd_log_level_t level, const char* text, void*) {
            if (level >= SD_LOG_WARN && text) std::cerr << text;
        }, nullptr);
        sd_set_abort_callback([](void*) { return abortRequested; }, nullptr);
        std::vector<std::string> backends{"cpu"};
#if defined(__APPLE__)
        backends.push_back("metal");
#endif
        for (const auto& backend : backends) for (float logit : {10.f, -10.f}) {
            const auto path = directory / (backend + (logit > 0 ? "-detected.safetensors" : "-empty.safetensors"));
            writeDetector(path, logit);
            auto context = std::unique_ptr<adetailer_ctx_t, decltype(&free_adetailer_ctx)>(
                new_resident_adetailer_ctx(path.string().c_str(), 3, backend.c_str()), free_adetailer_ctx);
            require(bool(context), "Resident YOLO detector loading failed");
#if defined(__APPLE__)
            require(backend != "metal" || ggml_backend_is_metal(context->detailer->backend_manager.runtime_backend(SDBackendModule::DETECTOR)),
                "Detector silently replaced Metal");
#endif
            std::filesystem::remove(path); // Only this test-created source; inference now must use memory.
            std::vector<uint8_t> pixels(64 * 64 * 3, 25);
            sd_image_t input{64, 64, 3, pixels.data()};
            sd_img_gen_params_t generation{};
            generation.width = generation.height = 64; generation.strength = .25f;
            generation.prompt = "detail test"; generation.negative_prompt = "noise";
            generation.sample_params.sample_steps = 8; generation.sample_params.guidance.txt_cfg = 6.5f;
            generation.hires.enabled = true; generation.ref_images_count = 1; generation.control_image = input;
            const sd_adetailer_params_t policy{nullptr, nullptr,
                "input_size=32,max_detections=1,mask_max_ratio=0.2,dilate_erode=0,mask_blur=2,inpaint_padding=4"};
            // The fake sampler never dereferences the opaque diffusion context.
            auto* diffusion = reinterpret_cast<sd_ctx_t*>(context.get());
            for (int repeat = 0; repeat < 2; ++repeat) {
                const auto before = inpaintCalls;
                sd_image_t* output = nullptr; int count = 0;
                require(adetail_image(context.get(), diffusion, input, &policy, &generation, &output, &count),
                    "Detailer pipeline failed");
                require(output && count == 1 && output->width == 64 && output->height == 64 && output->channel == 3,
                    "Detailer changed output contract");
                int changed = 0, untouched = 0, feathered = 0;
                for (size_t i = 0; i < pixels.size(); ++i) {
                    changed += output->data[i] != 25; untouched += output->data[i] == 25;
                    feathered += output->data[i] > 25 && output->data[i] < 200;
                }
                free_sd_images(output, count);
                require(std::all_of(pixels.begin(), pixels.end(), [](auto p) { return p == 25; }), "Caller input was mutated");
                require(logit > 0 ? (inpaintCalls == before + 1 && changed && untouched && feathered)
                                  : (inpaintCalls == before && !changed),
                    "Detection, masked refinement, no-detection or feathering contract failed");
            }
            // The graph still emits its original class count. A mismatched
            // output contract must fail, not be confused with no detections.
            ++context->detailer->detector->config.num_classes;
            sd_image_t* malformedOutput = nullptr; int malformedCount = 0;
            require(!adetail_image(context.get(), diffusion, input, &policy, &generation, &malformedOutput, &malformedCount)
                && !malformedOutput && !malformedCount, "Malformed detector output was accepted");
            --context->detailer->detector->config.num_classes;
            if (logit > 0) for (int failure = 0; failure < 3; ++failure) {
                failInpaint = failure == 0; abortRequested = failure == 1; cancelInpaint = failure == 2;
                sd_image_t* output = nullptr; int count = 0;
                const bool ok = adetail_image(context.get(), diffusion, input, &policy, &generation, &output, &count);
                require(!ok && !output && !count, "Failure/cancellation returned a successful base image");
                failInpaint = abortRequested = cancelInpaint = false;
            }
            std::cout << "Resident Detailer backend=" << backend << " confidence_logit=" << logit << " passed\n";
        }
        sd_set_abort_callback(nullptr, nullptr); sd_release_resident_model_memory();
    } catch (const std::exception& e) { sd_set_abort_callback(nullptr, nullptr); std::cerr << e.what() << '\n'; return 1; }
}
