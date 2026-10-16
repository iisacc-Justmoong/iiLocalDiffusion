#include "Generation/GenerationDefaults.hpp"
#include <json-c/json.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

int main(int argc, char **argv) try {
    if (argc != 2) return 2;
    const auto root = std::filesystem::absolute(argv[1]);
    std::filesystem::create_directories(root);
    const auto weights = root / "qwen.safetensors";
    { std::ofstream output(weights, std::ios::binary); const std::uint64_t length = 2;
      output.write(reinterpret_cast<const char *>(&length), sizeof(length)); output << "{}"; }
    const auto write = [&](const std::string &entries) {
        std::ofstream(root / "generation-defaults.json") << "{\"version\":1,\"fallback_llms\":" << entries << '}';
    };
    const std::string entry = "{\"file\":\"qwen.safetensors\",\"size\":10,\"class_name\":\"Qwen3VLForConditionalGeneration\",\"families\":[\"krea2\"]}";
    write('[' + entry + ']');
    const auto first = iiLocalDiffusion::native_detail::loadFallbackLlm(root, "krea2");
    if (first.path != weights || first.identity.empty()) throw std::runtime_error("Verified catalog entry was not mounted");
    const auto reject = [&](const std::string &entries, const std::string &family = "krea2") {
        write(entries);
        try { iiLocalDiffusion::native_detail::loadFallbackLlm(root, family); }
        catch (const std::runtime_error &) { return; }
        throw std::runtime_error("Incompatible or missing text encoder was accepted");
    };
    reject("[]"); reject('[' + entry + ',' + entry + ']'); reject('[' + entry + ']', "flux2");
    for (const auto &change : {std::pair{"Qwen3VLForConditionalGeneration", "Qwen3ForCausalLM"},
                               std::pair{"qwen.safetensors", "../qwen.safetensors"},
                               std::pair{"\"size\":10", "\"size\":11"},
                               std::pair{"krea2", "z-image"}}) {
        auto wrong = entry; wrong.replace(wrong.find(change.first), std::string(change.first).size(), change.second);
        reject('[' + wrong + ']');
    }
    write('[' + entry + ']');
    std::filesystem::remove(weights);
    reject('[' + entry + ']');
    std::cout << "Missing, wrong, duplicate and redirected Qwen3-VL resources rejected; family-scoped mount preserved\n";
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n'; return 1;
}
