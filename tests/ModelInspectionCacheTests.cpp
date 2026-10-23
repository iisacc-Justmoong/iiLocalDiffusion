#include <ModelPackaging/ModelInspection.hpp>
#include <iiFileProviderNative.h>
#include <iiFileProviderPresentation.h>
#include <filesystem>
#include <iostream>
#include <stdexcept>

void require(bool condition, const char *message) { if (!condition) throw std::runtime_error(message); }
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    namespace fs = std::filesystem;
    using iiFileProvider::native::File;
    try {
        const fs::path root = fs::absolute(argv[1]);
        File::createDirectories(root);
        const auto model = root / "model.safetensors";
        const std::string header = R"({"__metadata__":{"config":"{\"_class_name\":\"LTX2Pipeline\"}"}})";
        std::string bytes(8, '\0');
        for (unsigned i = 0; i < 8; ++i) bytes[i] = static_cast<char>((header.size() >> (i * 8)) & 255);
        File::write(model, bytes + header);
        require(iiFileProvider::native::Presentation::waitForIdle(std::chrono::seconds(5)), "Fixture publication must finish");
        auto first = iild::inspectModel(model);
        require(first.recognized && !first.cacheHit && first.metadataBytesRead > 0, "First inspection must read architecture metadata");
        auto cached = iild::inspectModel(model);
        require(cached.cacheHit && cached.metadataBytesRead == 0, "Repeated polling must not reread a large header");
        iild::setModelModalities(model, {"image", "video"});
        auto manual = iild::inspectModel(model);
        require(!manual.cacheHit && manual.userOverride && manual.modalities == std::vector<std::string>{"image", "video"}, "Sidecar edits must invalidate discovery");
        iild::setModelModalities(model, {});
        auto automatic = iild::inspectModel(model);
        require(!automatic.userOverride && automatic.modalities == first.modalities, "Automatic inspection must restore actual output modalities");
        const auto index = root / "model_index.json";
        File::write(index, R"({"_class_name":"LTXPipeline"})");
        require(iild::inspectModel(root).modalities == std::vector<std::string>{"video"}, "Legacy pipeline discovery");
        File::write(index, R"({"_class_name":"StableDiffusionPipeline"})");
        require(iild::inspectModel(root).modalities == std::vector<std::string>{"image"}, "Directory configuration edits invalidate cached classification");
        std::cout << "Model inspection cache and metadata invalidation passed\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
