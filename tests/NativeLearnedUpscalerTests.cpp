#include "upscaler.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#if defined(__APPLE__)
#include <ggml-metal.h>
#endif

namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
// A real one-block RRDB graph with controlled output biases. No downloaded
// model or quality claim: known weights prove learned inference and residency.
void writeModel(const std::filesystem::path& path, float bias) {
    std::ostringstream header;
    std::vector<float> weights;
    header << '{';
    bool first = true;
    const auto tensor = [&](const std::string& name, const std::vector<int>& shape, float value) {
        if (!first) header << ',';
        first = false;
        header << '"' << name << "\":{\"dtype\":\"F32\",\"shape\":[";
        size_t count = 1;
        for (size_t i = 0; i < shape.size(); ++i) { if (i) header << ','; header << shape[i]; count *= shape[i]; }
        const auto start = weights.size() * sizeof(float);
        weights.insert(weights.end(), count, value);
        header << "],\"data_offsets\":[" << start << ',' << weights.size() * sizeof(float) << "]}";
    };
    const auto conv = [&](const std::string& name, int input, int output) {
        tensor(name + ".weight", {output, input, 3, 3}, 0);
        tensor(name + ".bias", {output}, name == "conv_last" ? bias : 0);
    };
    conv("conv_first", 3, 64);
    for (int block = 1; block <= 3; ++block) for (int layer = 1; layer <= 5; ++layer)
        conv("body.0.rdb" + std::to_string(block) + ".conv" + std::to_string(layer),
             64 + (layer - 1) * 32, layer == 5 ? 64 : 32);
    for (const auto* name : {"conv_body", "conv_up1", "conv_up2", "conv_hr"}) conv(name, 64, 64);
    conv("conv_last", 64, 3);
    header << '}';
    auto json = header.str();
    while (json.size() % 8) json += ' ';
    const uint64_t size = json.size();
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
    stream.write(json.data(), json.size());
    stream.write(reinterpret_cast<const char*>(weights.data()), weights.size() * sizeof(float));
    check(bool(stream), "Cannot write learned-upscaler fixture");
}
}
int main(int argc, char** argv) {
    try {
        check(argc == 2, "Expected build-local fixture directory");
        const std::filesystem::path directory(argv[1]);
        std::filesystem::create_directories(directory);
        const sd::Tensor<float> input({2, 3, 3, 1}, std::vector<float>(18, .9f));
        std::vector<std::string> backends{"cpu"};
#if defined(__APPLE__)
        backends.push_back("metal");
#endif
        for (const auto& backend : backends) for (float bias : {.25f, .75f}) {
            const auto path = directory / (backend + (bias < .5f ? "-first.safetensors" : "-second.safetensors"));
            writeModel(path, bias);
            UpscalerGGML model(3, false, 128, backend, "cpu");
            check(model.load_from_file(path.string(), 3, true), "Resident ESRGAN loading failed");
            const auto actualBackend = std::string(ggml_backend_name(model.backend_manager.runtime_backend(SDBackendModule::UPSCALER)));
#if defined(__APPLE__)
            check(backend != "metal" || ggml_backend_is_metal(model.backend_manager.runtime_backend(SDBackendModule::UPSCALER)),
                  "Requested Metal backend was silently substituted");
#endif
            check(model.esrgan_upscaler->config.scale == 4, "Model is not 4x");
            // Only this test-created file is removed. Subsequent inference must
            // use anonymous source storage, even if backend parameters evict.
            std::filesystem::remove(path);
            for (int repeat = 0; repeat < 2; ++repeat) {
                const auto output = model.upscale_tensor(input);
                check(output.shape() == std::vector<int64_t>({8, 12, 3, 1}), "Learned output dimensions differ");
                for (float pixel : output.values()) check(std::abs(pixel - bias) < 1e-5f, "Output did not use selected weights");
            }
            std::cout << "Learned upscaler backend=" << actualBackend << " bias=" << bias << " passed\n";
        }
        sd_release_resident_model_memory();
        std::cout << "Resident learned 4x ESRGAN inference passed after source removal\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
