#pragma once
#include <filesystem>
#include <string>
#include <vector>

namespace iiLocalDiffusion::native_detail {
struct DefaultEmbedding {
    std::filesystem::path path;
    std::string token;
    bool sd15 = false;
};
struct DefaultLora {
    std::filesystem::path path;
    float scale = 1;
    std::vector<std::string> families;
};
struct GenerationDefaults {
    std::vector<DefaultLora> loras;
    std::vector<DefaultEmbedding> embeddings;
    std::string identity;
};
struct DefaultVae {
    std::filesystem::path path;
    std::string identity;
};
DefaultVae loadFallbackVae(const std::filesystem::path &directory, const std::string &family);
std::string canonicalLoraFamily(const std::string &family);
GenerationDefaults loadGenerationDefaults(const std::filesystem::path &directory);
std::filesystem::path generationResourceDirectory();
std::string appendNegativeTokens(std::string prompt, const std::vector<std::string> &tokens);
}
