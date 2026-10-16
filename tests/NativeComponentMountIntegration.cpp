#include "Generation/NativeDiffusion.hpp"
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>

// Opt-in real-model check: exact source checkpoint, catalog-mounted components,
// resident preparation and one complete 128px inference. No model conversion.
int main(int argc, char **argv) try {
    using namespace iiLocalDiffusion;
    if (argc != 4 || !nativeDiffusionAvailable()) return 2;
    const auto output = std::filesystem::absolute(argv[3]);
    std::filesystem::create_directories(output);
    NativeGenerationRequest request;
    request.modelPath = std::filesystem::canonical(argv[1]);
    request.prompt = "a red apple on a wooden table";
    request.width = request.height = 128; request.steps = 1; request.seed = 42;
    request.timeoutMilliseconds = 3600000;
    NativeGenerationOptions options;
    options.resourceDirectory = std::filesystem::canonical(argv[2]); options.defaultModifiers = false;
    NativeModelComponents components;
    components.guidanceScale = 1; components.hires = false;
    NativeSamplingControls sampling; sampling.sampler = NativeSampler::Euler;
    std::atomic_bool cancelled{false};
    const auto progress = [](const NativeGenerationProgress &event) {
        std::cout << "stage=" << static_cast<int>(event.stage) << " step=" << event.step
                  << " total=" << event.total << std::endl;
    };
    const auto mounted = generateNativeImageWithResidentWeights(request, options, components, sampling,
        NativeComputeBackend::Automatic, cancelled, true, progress);
    if (mounted.cancelled || !mounted.error.empty() || !mounted.rgb.empty())
        throw std::runtime_error("Model preparation failed: " + mounted.error);
    std::cout << "resident model preparation complete; bytes=" << mounted.modelBytes
              << " ms=" << mounted.modelLoadMilliseconds << std::endl;
    { std::ofstream report(output / "mounted.json");
      report << "{\"prepared\":true,\"resident\":true,\"modelBytes\":" << mounted.modelBytes
             << ",\"modelLoadMilliseconds\":" << mounted.modelLoadMilliseconds << "}\n"; }
    const auto result = generateNativeImageWithResidentWeights(request, options, components, sampling,
        NativeComputeBackend::Automatic, cancelled, false, progress);
    if (result.cancelled || !result.error.empty() || result.width != 128 || result.height != 128
        || result.rgb.size() != 128 * 128 * 3 || !result.modelCacheHit)
        throw std::runtime_error("Prepared model inference failed: " + result.error);
    std::set<std::uint32_t> colors;
    for (std::size_t i = 0; i < result.rgb.size(); i += 3)
        colors.insert((std::uint32_t(result.rgb[i]) << 16) | (std::uint32_t(result.rgb[i+1]) << 8) | result.rgb[i+2]);
    if (colors.size() < 256) throw std::runtime_error("Real model returned a blank or uniform image");
    std::ofstream pixels(output / "image.ppm", std::ios::binary);
    pixels << "P6\n128 128\n255\n";
    pixels.write(reinterpret_cast<const char*>(result.rgb.data()), result.rgb.size());
    pixels.close();
    if (!pixels) throw std::runtime_error("Cannot save image");
    std::ofstream report(output / "result.json");
    report << "{\"complete\":true,\"modelCacheHit\":true,\"width\":128,\"height\":128,\"uniqueColors\":"
           << colors.size() << ",\"generationMilliseconds\":" << result.generationMilliseconds << "}\n";
    releaseNativeDiffusionCache();
    std::cout << "Real checkpoint produced a complete nonuniform RGB image from the retained context\n";
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n'; return 1;
}
