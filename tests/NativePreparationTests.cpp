#include "Generation/NativeDiffusion.hpp"
#include "Generation/NativeCachePolicy.hpp"
#include <fstream>
#include <iostream>
#include <sstream>

// A small valid checkpoint exercises the real streaming quantizer and worker
// cancellation without a trained model, inference, downloads or a large fixture.
int main(int argc, char **argv) {
    using namespace iiLocalDiffusion;
    namespace fs = std::filesystem;
    if (argc != 2 || !nativeDiffusionAvailable()) return 1;
    const fs::path root = argv[1];
    fs::remove_all(root);
    fs::create_directories(root);
    const auto source = root / "fixture.safetensors";
    constexpr std::uint64_t bytes = 128 * 128 * 2;
    std::ostringstream header;
    header << '{';
    for (int index = 0; index < 32; ++index) {
        if (index) header << ',';
        header << "\"model.diffusion_model.input_blocks." << index << ".0.weight\":{\"dtype\":\"F16\","
            "\"shape\":[128,128],\"data_offsets\":[" << index * bytes << ',' << (index + 1) * bytes << "]}";
    }
    header << '}';
    auto metadata = header.str();
    metadata.append((8 - metadata.size() % 8) % 8, ' ');
    {
        std::ofstream file(source, std::ios::binary);
        const auto length = static_cast<std::uint64_t>(metadata.size());
        for (int byte = 0; byte < 8; ++byte) file.put(static_cast<char>(length >> (byte * 8)));
        file << metadata;
        for (std::uint64_t element = 0; element < bytes * 32 / 2; ++element) { file.put('\0'); file.put('\x3c'); }
        if (!file) return 2;
    }
    const auto identity = native_detail::modelIdentity(source);
    for (const int stopAfter : {1, 3, 5}) {
        NativeGenerationRequest request;
        request.modelPath = source;
        request.q8CacheDirectory = root / ("cache-" + std::to_string(stopAfter));
        request.prompt = "fixture";
        request.width = request.height = 64;
        std::atomic_bool cancelled{false};
        bool preparing = false;
        auto result = generateNativeImageWithProgress(request, cancelled, [&](const auto &event) {
            if (event.stage == NativeGenerationStage::Preparing && event.step >= stopAfter)
                preparing = cancelled = true;
        });
        if (!preparing || !result.cancelled || !result.rgb.empty() || !result.error.empty()
            || result.preparationMilliseconds <= 0 || !fs::is_empty(request.q8CacheDirectory)
            || native_detail::modelIdentity(source) != identity) {
            std::cerr << "Q8 cancellation failed: " << result.error << '\n';
            return 3;
        }
    }
    std::cout << "real quantizer: three cancellations joined workers, removed partial files and preserved source\n";
}
