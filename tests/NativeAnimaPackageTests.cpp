#include "Generation/NativeDiffusion.hpp"
#include "Generation/ImageModelRouting.hpp"
#include "ModelPackaging/ModelPackaging.hpp"
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace iiLocalDiffusion;

static void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

static std::string read(const fs::path &path) {
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

static void tensorFile(const fs::path &path, const std::vector<std::string> &keys) {
    fs::create_directories(path.parent_path());
    std::string header = "{";
    for (std::size_t i = 0; i < keys.size(); ++i) {
        if (i) header += ',';
        header += '"' + keys[i] + "\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":["
            + std::to_string(i * 4) + ',' + std::to_string((i + 1) * 4) + "]}";
    }
    header += '}';
    while (header.size() % 8) header += ' ';
    std::ofstream file(path, std::ios::binary);
    const auto length = static_cast<std::uint64_t>(header.size());
    for (unsigned shift = 0; shift < 64; shift += 8) file.put(static_cast<char>((length >> shift) & 0xff));
    file << header << std::string(keys.size() * 4, '\0');
    require(static_cast<bool>(file), "fixture write failed");
}

int main(int argc, char **argv) {
    try {
        require(argc == 2, "expected workspace");
        fs::create_directories(argv[1]);
        const auto root = fs::canonical(argv[1]);
        const auto source = root / "sources";
        const auto model = source / "Checkpoint/anima.safetensors";
        const auto encoder = source / "LLM/text.safetensors";
        const auto vae = source / "Other/qwen.safetensors";
        tensorFile(model, {"model.diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj.weight",
            "model.diffusion_model.blocks.0.self_attn.q_proj.weight"});
        tensorFile(encoder, {"model.embed_tokens.weight", "model.norm.weight", "model.layers.0.self_attn.q_proj.weight"});
        tensorFile(vae, {"encoder.conv1.weight", "encoder.head.2.weight", "decoder.conv1.weight", "decoder.head.2.weight"});
        const std::vector<iild::ModelPackagingAssignment> assignments{
            {"Checkpoint/anima.safetensors", "model", "main"},
            {"LLM/text.safetensors", "text_encoder", "component"},
            {"Other/qwen.safetensors", "video_vae", "component"}};
        const auto package = root / "anima-pack.safetensors";
        (void)iild::createModelPackageWithAssignments(source, package, assignments);
        const auto original = read(package);
        NativeGenerationRequest request; request.modelPath = package; request.prompt = "fixture";
        std::atomic_bool cancelled{false};
        bool preparedProgress = false;
        const auto prepared = prepareNativeImagePackage(request, root / "cache", {}, cancelled,
            [&](const auto &event) { preparedProgress |= event.stage == NativeGenerationStage::Preparing; });
        require(prepared.packaged, "Anima package was not recognized");
        require(prepared.request.modelPath == package, "root transformer was unnecessarily extracted");
        require(read(prepared.components.llm) == read(encoder), "original text encoder was not recovered");
        require(read(prepared.components.vae) == read(vae), "original Qwen VAE was not recovered");
        require(preparedProgress, "component preparation did not report progress");
        require(read(package) == original, "source package changed");
        NativeModelComponents overrides; overrides.llm = encoder; overrides.vae = vae;
        const auto explicitComponents = prepareNativeImagePackage(request, root / "unused", overrides, cancelled, {});
        require(explicitComponents.components.llm == encoder && explicitComponents.components.vae == vae,
            "explicit components were overridden");
        require(!fs::exists(root / "unused"), "explicit components caused unnecessary extraction");
        NativeModelComponents onlyVae; onlyVae.vae = vae;
        const auto mixed = prepareNativeImagePackage(request, root / "mixed", onlyVae, cancelled, {});
        require(mixed.components.vae == vae && read(mixed.components.llm) == read(encoder),
            "explicit VAE and embedded encoder were not combined");
        const auto nestedPackage = root / "nested-anima.safetensors";
        const std::vector<iild::ModelPackagingAssignment> nestedAssignments{
            {"Checkpoint/anima.safetensors", "model", "component"},
            {"LLM/text.safetensors", "model", "main"},
            {"Other/qwen.safetensors", "video_vae", "component"}};
        (void)iild::createModelPackageWithAssignments(source, nestedPackage, nestedAssignments);
        auto nestedRequest = request; nestedRequest.modelPath = nestedPackage;
        const auto nested = prepareNativeImagePackage(nestedRequest, root / "nested", {}, cancelled, {});
        require(nested.packaged && nested.request.modelPath != nestedPackage
            && read(nested.request.modelPath) == read(model) && read(nested.components.llm) == read(encoder)
            && read(nested.components.vae) == read(vae), "nested transformer was not reconstructed");
        bool stoppedDuringCopy = false;
        try {
            (void)prepareNativeImagePackage(request, root / "cancel-midway", {}, cancelled,
                [&](const auto &) { cancelled = true; });
        } catch (const std::exception &) { stoppedDuringCopy = true; }
        require(stoppedDuringCopy, "preparation did not stop when cancellation arrived");
        cancelled = false;
        const auto ordinary = prepareNativeImagePackage(NativeGenerationRequest{model}, root / "ordinary", {}, cancelled, {});
        require(!ordinary.packaged && ordinary.request.modelPath == model && !fs::exists(root / "ordinary"),
            "a standalone checkpoint was incorrectly treated as an embedded package");
        auto control = std::make_shared<NativeExecutionControl>(); control->setPaused(true);
        auto paused = std::async(std::launch::async, [&] {
            try { (void)prepareNativeImagePackage(request, root / "paused", {}, cancelled, {}, control); return false; }
            catch (const std::exception &) { return true; }
        });
        for (int i = 0; i < 100 && !control->isWaiting(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        const auto wasWaiting = control->isWaiting(); cancelled = true;
        require(paused.get() && wasWaiting && !fs::exists(root / "paused"), "paused preparation did not cancel safely");
        bool interrupted = false;
        try { (void)prepareNativeImagePackage(request, root / "cancelled", {}, cancelled, {}); }
        catch (const std::exception &) { interrupted = true; }
        require(interrupted && !fs::exists(root / "cancelled"), "cancellation was not respected");
        if (const auto *actual = std::getenv("IILD_TEST_ANIMA_PACKAGE")) {
            const auto actualPath = fs::canonical(actual);
            const auto inspection = inspectImageModel(actualPath);
            const auto assets = inspectEmbeddedImageComponents(actualPath);
            require(inspection.architecture == ImageModelArchitecture::Anima && assets.packaged
                && !assets.transformer.empty() && !assets.textEncoder.empty() && !assets.vae.empty(),
                "actual package metadata did not identify all Anima components");
        }
        std::cout << "Anima package component binding verified\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
