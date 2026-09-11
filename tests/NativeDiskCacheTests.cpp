#include "Generation/NativeDiskCache.hpp"
#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
    using namespace iiLocalDiffusion::native_detail;
    namespace fs = std::filesystem;
    if (argc != 2) return 1;
    const fs::path root = argv[1];
    fs::remove_all(root);
    fs::create_directories(root);
    const auto source = root / "original.safetensors";
    { std::ofstream out(source); out << "original weights"; }
    const auto original = modelIdentity(source);
    int conversions = 0;
    const auto convert = [&](const fs::path &out) {
        ++conversions;
        std::ofstream file(out, std::ios::binary);
        file << "GGUF" << std::string(100, '\0');
        return bool(file);
    };
    const auto check = [] {};
    auto first = prepareQ8Cache(source, root / "cache", original, convert, check);
    auto warm = prepareQ8Cache(source, root / "cache", original, convert, check);
    if (first.hit || !warm.hit || conversions != 1 || first.path != warm.path || first.bytes != 104) return 2;
    if (modelIdentity(source) != original) return 3;
    // A truncated/replaced derivative must be rebuilt, never accepted by name.
    { std::ofstream corrupt(first.path); corrupt << "GGUF"; }
    if (prepareQ8Cache(source, root / "cache", original, convert, check).hit || conversions != 2) return 4;
    const auto failureCache = root / "failure";
    try {
        prepareQ8Cache(source, failureCache, original, [&](const auto &path) {
            convert(path);
            throw std::runtime_error("cancelled");
            return false;
        }, check);
        return 5;
    } catch (const std::runtime_error &) {}
    if (!fs::is_empty(failureCache)) return 6;
    // Source replacement during conversion cannot publish a valid manifest.
    try {
        prepareQ8Cache(source, failureCache, original, [&](const auto &path) {
            convert(path);
            { std::ofstream changed(source); changed << "new model weights"; }
            return true;
        }, check);
        return 7;
    } catch (const std::runtime_error &) {}
    if (!fs::is_empty(failureCache)) return 8;
    const auto changed = modelIdentity(source);
    auto replacement = prepareQ8Cache(source, root / "cache", changed, convert, check);
    if (replacement.hit || replacement.path == first.path) return 9;
    // Failed conversion may create a plausible header, but must not commit it.
    try {
        prepareQ8Cache(source, failureCache, changed, [&](const auto &path) { convert(path); return false; }, check);
        return 10;
    } catch (const std::runtime_error &) {}
    if (!fs::is_empty(failureCache)) return 11;
    std::cout << "Q8 cache: reuse, source identity, corruption, cancellation and atomic publication passed\n";
}
