// Opt-in real-checkpoint smoke. Input/output are interleaved RGB files so this
// probe tests the SDK without introducing an image codec or GUI dependency.
#include <Generation/NativeDiffusion.hpp>
#include <fstream>
#include <iostream>
#include <chrono>

int main(int argc, char **argv) {
    if (argc != 11 && argc != 12) {
        std::cerr << "model input.rgb input-width input-height width height steps cfg strength output.rgb [prompt.txt]\n";
        return 1;
    }
    using namespace iiLocalDiffusion;
    NativeGenerationRequest request;
    request.modelPath = std::filesystem::canonical(argv[1]);
    request.prompt = "A young woman with short black hair, a neutral expression, a white blouse with a blue ribbon and a warm brown checked jacket.";
    if (argc == 12) {
        std::ifstream prompt(argv[11], std::ios::binary);
        if (!prompt) return 2;
        request.prompt.assign(std::istreambuf_iterator<char>(prompt), std::istreambuf_iterator<char>());
        if (request.prompt.empty()) return 2;
    }
    request.width = std::stoi(argv[5]); request.height = std::stoi(argv[6]);
    request.steps = std::stoi(argv[7]); request.seed = 1396834166;
    request.timeoutMilliseconds = 7200000;
    NativeAdvancedControls controls;
    NativeReferenceImage image;
    image.width = std::stoi(argv[3]); image.height = std::stoi(argv[4]);
    image.rgb.resize(std::size_t(image.width) * image.height * 3);
    std::ifstream input(argv[2], std::ios::binary);
    input.read(reinterpret_cast<char *>(image.rgb.data()), image.rgb.size());
    if (!input) return 2;
    controls.references.push_back(std::move(image)); controls.imageStrength = std::stof(argv[9]);
    NativeModelComponents components; components.guidanceScale = std::stof(argv[8]);
    NativeGenerationOptions options; options.defaultModifiers = false;
    std::atomic_bool cancelled{false};
    const auto start = std::chrono::steady_clock::now();
    auto result = generateNativeAdvancedImage(request, options, components, controls,
        NativeComputeBackend::Automatic, cancelled, [&](const NativeGenerationProgress &progress) {
            if (progress.stage == NativeGenerationStage::Computing) return;
            std::cout << "stage=" << int(progress.stage) << " step=" << progress.step << '/' << progress.total
                << " seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count() << std::endl;
        });
    if (!result.error.empty() || result.cancelled || result.width != request.width || result.height != request.height) {
        std::cerr << result.error << '\n'; return 3;
    }
    std::ofstream output(argv[10], std::ios::binary);
    output.write(reinterpret_cast<const char *>(result.rgb.data()), result.rgb.size());
    std::cout << "complete width=" << result.width << " height=" << result.height
        << " generation_ms=" << result.generationMilliseconds << '\n';
    return output ? 0 : 4;
}
