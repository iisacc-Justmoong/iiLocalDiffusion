#include "Generation/GenerationDefaults.hpp"
#include <json-c/json.h>
#include <cstdlib>
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
    const std::string animaEntry = "{\"file\":\"qwen.safetensors\",\"size\":10,\"class_name\":\"Qwen3ForCausalLM\",\"families\":[\"anima\"]}";
    write('[' + entry + ']');
    const auto first = iiLocalDiffusion::native_detail::loadFallbackLlm(root, "krea2");
    if (first.path != weights || first.identity.empty()) throw std::runtime_error("Verified catalog entry was not mounted");
    write('[' + entry + ',' + animaEntry + ']');
    const auto anima = iiLocalDiffusion::native_detail::loadFallbackLlm(root, "anima");
    if (anima.path != weights || anima.identity.empty()) throw std::runtime_error("Anima text encoder entry was not mounted");
    const auto reject = [&](const std::string &entries, const std::string &family = "krea2") {
        write(entries);
        try { iiLocalDiffusion::native_detail::loadFallbackLlm(root, family); }
        catch (const std::runtime_error &) { return; }
        throw std::runtime_error("Incompatible or missing text encoder was accepted");
    };
    reject("[]"); reject('[' + entry + ',' + entry + ']'); reject('[' + entry + ']', "flux2");
    reject('[' + entry + ']', "anima"); reject('[' + animaEntry + ',' + animaEntry + ']', "anima");
    auto wrongAnima = animaEntry;
    wrongAnima.replace(wrongAnima.find("Qwen3ForCausalLM"), std::string("Qwen3ForCausalLM").size(), "Qwen3VLForConditionalGeneration");
    reject('[' + wrongAnima + ']', "anima");
    const auto gguf = root / "qwen-unusual-extension.bin";
    { std::ofstream output(gguf, std::ios::binary); output.write("GGUF", 4);
      const char version[4] = {3, 0, 0, 0}; output.write(version, 4); output << std::string(16, '\0'); }
    const std::string ggufEntry = "{\"file\":\"qwen-unusual-extension.bin\",\"size\":24,\"class_name\":\"Qwen3ForCausalLM\",\"families\":[\"anima\"]}";
    write('[' + entry + ',' + ggufEntry + ']');
    if (iiLocalDiffusion::native_detail::loadFallbackLlm(root, "anima").path != gguf)
        throw std::runtime_error("GGUF text encoder was not identified by its content");
    { std::ofstream output(gguf, std::ios::binary); output << "GGUF"; }
    reject('[' + ggufEntry + ']', "anima");
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
    if (const char *resources = std::getenv("IILD_TEST_GENERATION_RESOURCES")) {
        const auto actualRoot = std::filesystem::canonical(resources);
        const auto actual = iiLocalDiffusion::native_detail::loadFallbackLlm(actualRoot, "anima");
        if (actual.path != actualRoot / "text_encoders/anima_qwen3_0_6b_base.safetensors")
            throw std::runtime_error("Provisioned Anima encoder was not selected from the actual catalog");
        std::cout << "Provisioned Anima encoder selected from catalog\n";
    }
    if (const char *resources = std::getenv("IILD_TEST_SOCIETY_RESOURCES")) {
        const auto actualRoot = std::filesystem::canonical(resources);
        const auto actual = iiLocalDiffusion::native_detail::loadFallbackVae(actualRoot, "qwen-image");
        if (!std::filesystem::is_regular_file(actual.path) || actual.identity.empty())
            throw std::runtime_error("Society Qwen Image VAE was not selected from its catalog");
        std::cout << "Society Qwen Image VAE selected from catalog\n";
    }
    std::cout << "Missing, wrong, duplicate and redirected Qwen3-VL resources rejected; family-scoped mount preserved\n";
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n'; return 1;
}
