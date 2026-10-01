#include "core/util.h"
#include "conditioning/conditioner.hpp"
#include <ggml-cpu.h>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <iostream>
#include <stdexcept>

void check(bool condition) { if (!condition) throw std::runtime_error("Text conditioning contract failed"); }
void writeEmbedding(const std::filesystem::path &path, int width) {
    const auto header = std::string("{\"emb_params\":{\"dtype\":\"F32\",\"shape\":[1,")
        + std::to_string(width) + "],\"data_offsets\":[0," + std::to_string(width * 4) + "]}}";
    const uint64_t length = header.size();
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char *>(&length), sizeof(length));
    stream.write(header.data(), static_cast<std::streamsize>(header.size()));
    const std::vector<float> values(width, 0.25f);
    stream.write(reinterpret_cast<const char *>(values.data()), width * 4);
    check(bool(stream));
}
int main(int argc, char **argv) {
    try {
        for (const auto &text : std::vector<std::string>{"(house:1.5) [forest] BREAK coast", "", "literal \\(한글\\) [sky]", "((unbalanced"}) {
            const auto weighted = parse_prompt_attention(text);
            const auto literal = parse_prompt_attention(text, false);
            check(literal == std::vector<std::pair<std::string, float>>{{text, 1.f}});
            check(parse_prompt_attention(text, true) == weighted); // Policy never leaks between calls.
        }
        const auto weighted = parse_prompt_attention("(house:1.5)", true);
        check(weighted.size() == 1 && weighted[0].first == "house" && weighted[0].second == 1.5f);
        check(argc == 2);
        const std::filesystem::path directory(argv[1]);
        std::filesystem::create_directories(directory);
        const auto valid = directory / "valid.safetensors", invalid = directory / "invalid.safetensors";
        writeEmbedding(valid, 768); writeEmbedding(invalid, 3);
        auto backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(ggml_backend_cpu_init(), &ggml_backend_free);
        check(bool(backend));
        {
            // Only CLIP tensor metadata is created; no diffusion or text-model weights are allocated/inferred.
            FrozenCLIPEmbedderWithCustomWords conditioner(backend.get(), {},
                {{"user_style", valid.string()}, {"invalid", invalid.string()}}, VERSION_SD1);
            conditioner.memory_resident_embeddings = true;
            check(conditioner.preload_embedding("user_style"));
            check(conditioner.num_custom_embeddings == 1 && conditioner.token_embed_custom.size() == 768 * sizeof(float));
            float first = 0; std::memcpy(&first, conditioner.token_embed_custom.data(), sizeof(first));
            check(first == 0.25f);
            const auto moved = directory / "temporarily-moved.safetensors";
            std::filesystem::rename(valid, moved);
            const bool warm = conditioner.preload_embedding("user_style");
            std::filesystem::rename(moved, valid);
            check(warm && conditioner.num_custom_embeddings == 1); // Warm use must not reopen the source.
            check(!conditioner.preload_embedding("invalid"));
            auto tokens = conditioner.tokenize("(user_style:1.5)", 77, 77, true);
            check(std::find(tokens.second.begin(), tokens.second.end(), 1.5f) != tokens.second.end());
            conditioner.prompt_weighting = false;
            tokens = conditioner.tokenize("(user_style:1.5)", 77, 77, true);
            check(std::all_of(tokens.second.begin(), tokens.second.end(), [](float weight) { return weight == 1; }));
        }
        sd_release_resident_model_memory();
        std::cout << "Native prompt weighting and literal parsing passed\n";
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
