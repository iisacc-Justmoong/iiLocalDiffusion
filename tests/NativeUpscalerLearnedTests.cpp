#include "Generation/NativeUpscaler.hpp"
#include "Generation/NativeUpscalerBridge.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>
namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
// A real one-block RRDB graph with controlled output biases. No downloaded
// model or quality claim: known weights prove learned inference and residency.
void writeModel(const std::filesystem::path& path, float bias, int scale = 4) {
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
    for (const auto* name : {"conv_body", "conv_up1", "conv_hr"}) conv(name, 64, 64);
    if (scale == 4) conv("conv_up2", 64, 64);
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
int main(int argc, char **argv) {
    using namespace iiLocalDiffusion;
    try {
        check(argc == 2, "Expected resource workspace");
        const std::filesystem::path dir(argv[1]);
        std::filesystem::create_directories(dir);
        UpscaleOptions settings; settings.passes = 2;
        for (float bias : {.25f, .75f}) {
            const auto path = dir / (bias < .5f ? "first.safetensors" : "second.safetensors");
            writeModel(path,bias);
            auto provider = makeEsrganUpscaler({path,UpscaleDevice::Cpu,0,1});
            UpscaleImage input{2,2,3,std::vector<std::uint8_t>(12,200)};
            const auto result = upscaleImage(input,*provider,settings);
            check(result.image.width == 32 && result.image.height == 32 && result.passes.size() == 2
                  && result.appliedScale == 4, "Wrong native ESRGAN dimensions or model default");
            check(result.details.at("model_scale") == "4" && result.details.at("requested_device") == "cpu"
                  && result.details.at("threads") == "1" && result.details.at("tile_size") == "0", "Lost ESRGAN options");
            for (auto pixel : result.image.pixels) check(std::abs(int(pixel)-int(bias*255)) <= 1, "Selected weights were not used");
            settings.scale=2; settings.passes=1;
            bool rejected = false;
            try { (void)upscaleImage(input,*provider,settings); } catch (const std::invalid_argument &) { rejected=true; }
            check(rejected,"Incompatible model scale was silently changed");
            settings.scale=0; settings.passes=2;
        }
        const auto twoPath=dir / "two.safetensors";
        writeModel(twoPath,.5f,2);
        auto two=makeEsrganUpscaler({twoPath,UpscaleDevice::Cpu,0,1});
        settings.scale=0; settings.passes=1;
        const auto doubled=upscaleImage({2,2,3,std::vector<std::uint8_t>(12,100)},*two,settings);
        check(doubled.image.width==4 && doubled.image.height==4 && doubled.details.at("model_scale")=="2"
              && doubled.appliedScale == 2, "2x model default detection failed");
        const auto pathString=twoPath.string();
        const std::vector<std::uint8_t> rgb(12,100);
        iild_upscale_request_v1 request{};
        request.size=sizeof(request); request.pixels=rgb.data(); request.pixel_bytes=rgb.size();
        request.width=2; request.height=2; request.channels=3; request.kind=4;
        request.model=pathString.c_str(); request.cpu=1; request.tile_size=0; request.threads=1;
        request.scale=0; request.passes=1; request.max_output_bytes=4096;
        const std::unique_ptr<iild_upscale_result_v1,decltype(&iild_upscale_free_v1)> result(
            iild_upscale_v1(&request,nullptr,nullptr),iild_upscale_free_v1);
        check(bool(result), "C bridge failed to allocate a result");
        const auto *metadata=iild_upscale_metadata_v1(result.get());
        check(metadata && std::string(metadata).find("\"model_scale\":\"2\"") != std::string::npos,
              "C bridge did not select the supplied 2x model");
        std::size_t bytes=0;
        check(iild_upscale_pixels_v1(result.get(),&bytes) && bytes==4*4*3, "C bridge returned invalid learned output");
        iild_upscale_request_v2 v2{};
        v2.size=sizeof(v2); v2.image=request;
        v2.max_weight_bytes=1024ULL*1024*1024; v2.max_working_bytes=512ULL*1024*1024;
        const std::unique_ptr<iild_upscale_result_v1,decltype(&iild_upscale_free_v1)> resultV2(
            iild_upscale_v2(&v2,nullptr,nullptr),iild_upscale_free_v1);
        check(resultV2 && std::string(iild_upscale_metadata_v1(resultV2.get())).find("\"model_scale\":\"2\"") != std::string::npos,
              "V2 C bridge no longer preserves the original learned provider");
        std::cout << "Synthetic native ESRGAN adapter passed\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
