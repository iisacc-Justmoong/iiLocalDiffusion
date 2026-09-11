#include "Generation/NativeDiffusion.hpp"
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <iterator>
#include <algorithm>

// Opt-in real-model integration: deterministic warm reuse and pressure eviction.
// Keep this separate from the fast, model-free contract suite.
int main(int argc, char **argv) {
    using namespace iiLocalDiffusion;
    if (argc != 2 && argc != 3) return 1;
    NativeGenerationRequest request;
    request.modelPath = std::filesystem::canonical(argv[1]);
    request.prompt = "A white cockatoo on a flowering branch, botanical illustration";
    request.width = request.height = 512;
    request.steps = 1;
    request.seed = 1012192758;
    if (const auto *directory = std::getenv("IILD_NATIVE_Q8_CACHE")) request.q8CacheDirectory = directory;
    std::atomic_bool cancelled{false};
    releaseNativeDiffusionCache();
    std::vector<std::uint8_t> first;
    for (int run = 0; run < 3; ++run) {
        bool releasedInCallback = false;
        auto result = generateNativeImageWithProgress(request, cancelled, [&](const auto &event) {
            if (run == 1 && !releasedInCallback && event.stage == NativeGenerationStage::Denoising) {
                releasedInCallback = true;
                releaseNativeDiffusionCache();
            }
        });
        std::cout << "run=" << run << " hit=" << result.modelCacheHit << " budget=" << result.memoryBudgetBytes
                  << " load_ms=" << result.modelLoadMilliseconds << " generate_ms=" << result.generationMilliseconds
                  << " q8=" << result.q8CacheUsed << " disk_hit=" << result.diskCacheHit
                  << " prepare_ms=" << result.preparationMilliseconds << " bytes=" << result.modelBytes
                  << " error=" << result.error << std::endl;
        if (!result.error.empty() || result.rgb.empty() || result.modelCacheHit != (run == 1)) return 2;
        if (run == 1 && !releasedInCallback) return 4;
        if (!request.q8CacheDirectory.empty()
            && (!result.q8CacheUsed || (run > 0 && !result.diskCacheHit))) return 9;
        if (argc == 3) {
            std::ofstream output(std::string(argv[2]) + ".run" + std::to_string(run), std::ios::binary);
            output.write(reinterpret_cast<const char *>(result.rgb.data()), result.rgb.size());
            if (!output) return 7;
        }
        if (run == 0) {
            first = result.rgb;
            const auto [minimum, maximum] = std::minmax_element(first.begin(), first.end());
            if (*minimum == *maximum) return 8; // A flat image is not a useful model fixture.
            if (argc == 3) {
                if (std::filesystem::exists(argv[2])) {
                    std::ifstream input(argv[2], std::ios::binary);
                    const std::vector<std::uint8_t> expected{std::istreambuf_iterator<char>(input), {}};
                    if (expected != first) return 6;
                } else {
                    std::ofstream output(argv[2], std::ios::binary);
                    output.write(reinterpret_cast<const char*>(first.data()), first.size());
                    if (!output) return 7;
                }
            }
        }
        else if (first != result.rgb) return 3;
    }
    // Leave the final successful context warm: process-exit cleanup must run
    // before the Metal backend registry is destroyed.
}
