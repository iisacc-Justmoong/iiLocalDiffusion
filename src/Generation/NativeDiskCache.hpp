#pragma once
#include "NativeCachePolicy.hpp"
#include <atomic>
#include <fstream>
#include <iomanip>

namespace iiLocalDiffusion::native_detail {
// A format revision is part of every key; changing converter/rules cannot reuse
// an older derivative. The full source identity is also checked in the manifest.
inline constexpr auto q8Revision = "sdcpp-d04e895-q8_0-vae-f16-v1";
struct PreparedModel {
    std::filesystem::path path;
    bool hit = false;
    std::uint64_t bytes = 0;
};
inline std::string cacheKey(std::string_view identity) {
    // This is a filename, not an integrity/authentication hash. Full identities
    // below prevent a hash collision from selecting another source's weights.
    std::uint64_t value = 14695981039346656037ull;
    for (const unsigned char byte : identity) { value ^= byte; value *= 1099511628211ull; }
    std::ostringstream out;
    out << std::hex << value;
    return out.str();
}
inline bool ggufHeader(const std::filesystem::path &path) {
    if (!std::filesystem::is_regular_file(path) || std::filesystem::file_size(path) < 24) return false;
    std::ifstream file(path, std::ios::binary);
    char magic[4]{};
    file.read(magic, 4);
    return file && std::string_view(magic, 4) == "GGUF";
}
inline bool validQ8Cache(const std::filesystem::path &path, const std::filesystem::path &manifest,
                         const std::string &sourceIdentity) {
    try {
        if (!std::filesystem::is_regular_file(manifest) || std::filesystem::file_size(manifest) > 32768
            || !ggufHeader(path)) return false;
        std::ifstream file(manifest);
        std::string revision, source, output;
        file >> std::quoted(revision) >> std::quoted(source) >> std::quoted(output);
        return file && revision == q8Revision && source == sourceIdentity && output == modelIdentity(path);
    } catch (const std::filesystem::filesystem_error &) { return false; }
}
template<class Convert, class Check>
PreparedModel prepareQ8Cache(const std::filesystem::path &source, const std::filesystem::path &directory,
                            const std::string &sourceIdentity, Convert convert, Check check) {
    namespace fs = std::filesystem;
    check();
    if (!directory.is_absolute()) throw std::runtime_error("Use an absolute app cache directory for Q8 models.");
    fs::create_directories(directory);
    const auto cache = fs::canonical(directory);
    const auto key = std::string(q8Revision) + '-' + cacheKey(sourceIdentity);
    const auto output = cache / (key + ".gguf");
    const auto manifest = cache / (key + ".cache");
    if (validQ8Cache(output, manifest, sourceIdentity)) return {output, true, fs::file_size(output)};
    // Each attempt owns a unique staging directory. Crashes never turn a
    // partially written checkpoint into a cache hit; concurrent attempts do not
    // truncate each other's output. No source file is opened for writing.
    static std::atomic_uint64_t sequence{0};
    fs::path staging;
    do {
        staging = cache / (".prepare-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())
                           + '-' + std::to_string(sequence++));
    } while (!fs::create_directory(staging));
    struct Cleanup {
        fs::path path;
        ~Cleanup() { std::error_code ignored; fs::remove_all(path, ignored); }
    } cleanup{staging};
    const auto temporary = staging / "model.gguf";
    if (!convert(temporary) || !ggufHeader(temporary))
        throw std::runtime_error("Could not prepare the Q8 model cache. Check available storage and try again.");
    check();
    if (modelIdentity(source) != sourceIdentity)
        throw std::runtime_error("The local model changed while preparing its Q8 cache. Try again.");
    // Another process may have completed the same deterministic conversion.
    if (validQ8Cache(output, manifest, sourceIdentity)) return {output, true, fs::file_size(output)};
    fs::rename(temporary, output);
    const auto metadata = staging / "model.cache";
    {
        std::ofstream file(metadata, std::ios::trunc);
        file << std::quoted(q8Revision) << '\n' << std::quoted(sourceIdentity) << '\n'
             << std::quoted(modelIdentity(output)) << '\n';
        file.close();
        if (!file) throw std::runtime_error("Could not save the Q8 model cache metadata.");
    }
    fs::rename(metadata, manifest);
    return {output, false, fs::file_size(output)};
}
}
