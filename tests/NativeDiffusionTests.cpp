#include "Generation/NativeDiffusion.hpp"
#include <atomic>
#include <fstream>
#include <iostream>
#include <chrono>
#include <cstdlib>

int main(int argc, char **argv) {
    using namespace iiLocalDiffusion;
    if (argc != 2) return 1;
    const auto directory = std::filesystem::path(argv[1]);
    std::filesystem::create_directories(directory);
    std::atomic_bool cancelled{true};
    NativeGenerationRequest request;
    auto result = generateNativeImage(request, cancelled);
    if (!result.cancelled || !result.rgb.empty()) return 2;
    cancelled = false;
    request.modelPath = directory / "missing.safetensors";
    request.prompt = "a forest";
    result = generateNativeImage(request, cancelled);
    if (result.error.empty() || !result.rgb.empty()) return 3;
    request.modelPath = directory / "invalid.safetensors";
    { std::ofstream file(request.modelPath); file << "not a checkpoint"; }
    request.width = 63;
    result = generateNativeImage(request, cancelled);
    if (result.error != "Invalid native image generation parameters.") return 4;
    request.width = 64;
    request.timeoutMilliseconds = 0;
    result = generateNativeImageWithProgress(request, cancelled, {});
    if (result.error != "Invalid native image generation parameters.") return 6;
    request.timeoutMilliseconds = 900000;
    bool falseStep = false;
    result = generateNativeImage(request, cancelled, [&](int, int) { falseStep = true; });
    if (falseStep) return 7; // Loading a rejected file cannot emit denoising steps.
    result = generateNativeImage(request, cancelled);
    if (result.error.empty() || !result.rgb.empty()) return 5;
    std::filesystem::remove(request.modelPath);
    if (const auto *model = std::getenv("IILD_NATIVE_TEST_MODEL")) {
        request.modelPath = std::filesystem::canonical(model);
        request.width = request.height = 64;
        request.steps = 1;
        for (const auto stage : {NativeGenerationStage::Loading, NativeGenerationStage::Denoising}) {
            cancelled = false;
            bool requested = false;
            const auto started = std::chrono::steady_clock::now();
            result = generateNativeImageWithProgress(request, cancelled, [&](const auto &event) {
                if (event.stage == stage && (stage != NativeGenerationStage::Loading || event.step > 0))
                    requested = cancelled = true;
            });
            const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            std::cout << "real_model_cancel_stage=" << static_cast<int>(stage) << "; seconds=" << seconds << '\n';
            if (!requested || !result.cancelled || !result.rgb.empty() || seconds > 30) return 8;
        }
        cancelled = false;
        request.timeoutMilliseconds = 100;
        result = generateNativeImageWithProgress(request, cancelled, {});
        if (result.cancelled || !result.rgb.empty() || result.error.find("time limit") == std::string::npos) return 9;
        std::cout << "real_model_deadline_passed\n";
        request.timeoutMilliseconds = 900000;
        request.q8CacheDirectory = directory / "q8-cancel";
        std::filesystem::remove_all(request.q8CacheDirectory);
        bool preparationProgress = false;
        const auto started = std::chrono::steady_clock::now();
        result = generateNativeImageWithProgress(request, cancelled, [&](const auto &event) {
            if (event.stage == NativeGenerationStage::Preparing && event.step > 0)
                preparationProgress = cancelled = true;
        });
        if (!preparationProgress || !result.cancelled || !result.rgb.empty()
            || !std::filesystem::is_empty(request.q8CacheDirectory)
            || std::chrono::steady_clock::now() - started > std::chrono::seconds(30)) return 10;
        std::cout << "real_model_q8_cancel_passed\n";
    }
    std::cout << "native_available=" << nativeDiffusionAvailable() << "; cancellation, invalid inputs, model rejection passed\n";
}
