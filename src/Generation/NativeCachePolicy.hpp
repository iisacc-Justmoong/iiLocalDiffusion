#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <chrono>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#if defined(__unix__) || defined(__APPLE__)
#include <sys/stat.h>
#endif

namespace iiLocalDiffusion::native_detail {
struct ResourceLimits {
    std::uint64_t physical = 0;
    std::uint64_t recommended = 0;
    std::uint64_t available = 0;
    bool availableKnown = false;
};
inline std::uint64_t memoryBudget(ResourceLimits limits, bool mobileCpuVae = false) {
    constexpr std::uint64_t MiB = 1024 * 1024;
    if (!limits.recommended) limits.recommended = limits.physical ? limits.physical / 2 : 2 * 1024 * MiB;
    // The engine reserves graph workspace inside this budget. Desktop leaves
    // a fixed host margin instead of withholding another sixth of free memory.
    // iOS still needs proportional headroom for its CPU VAE and GPU driver.
    auto budget = limits.recommended > 256 * MiB ? limits.recommended - 256 * MiB : 0;
    if (mobileCpuVae) budget = std::min(budget, limits.recommended / 5 * 3);
    if (limits.available || limits.availableKnown) {
        const auto reserve = mobileCpuVae
            ? std::max<std::uint64_t>(1536 * MiB, limits.available / 5 * 2)
            : 1024 * MiB;
        budget = std::min(budget, limits.available > reserve ? limits.available - reserve : 0);
    }
    return budget / (256 * MiB) * (256 * MiB);
}
inline bool mayAttemptModelLoad(std::uint64_t memoryBudgetBytes, bool memoryResidentModel) {
    constexpr std::uint64_t minimumManagedBudget = 512ull * 1024 * 1024;
    // Anonymous resident pages may be compressed or swapped by the OS. Do not
    // reject these loads using the GPU-placement budget; let allocation/loading
    // report a real failure instead.
    return memoryResidentModel || memoryBudgetBytes >= minimumManagedBudget;
}
// Diagnostic experiments may lower the managed GPU budget, never raise it
// beyond the live resource policy. Invalid input leaves that policy intact.
inline std::uint64_t diagnosticMemoryCeiling(std::uint64_t budget, std::string_view text) {
    std::uint64_t mib = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), mib);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
        mib < 512 || mib > UINT64_MAX / (1024 * 1024)) return budget;
    return std::min(budget, mib * 1024 * 1024);
}
inline std::string modelMetadataIdentity(const std::filesystem::path &path) {
    const auto name = path.string();
    const auto bytes = std::filesystem::file_size(path);
    std::ostringstream key;
    key.imbue(std::locale::classic());
    key << name << ':' << bytes << ':'
        << std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::filesystem::last_write_time(path).time_since_epoch()).count();
#if defined(__unix__) || defined(__APPLE__)
    struct stat info{};
    if (::stat(path.c_str(), &info) != 0) throw std::runtime_error("The local model is no longer available.");
    key << ':' << info.st_dev << ':' << info.st_ino;
#if defined(__APPLE__)
    key << ':' << info.st_ctimespec.tv_sec << ':' << info.st_ctimespec.tv_nsec;
#else
    key << ':' << info.st_ctim.tv_sec << ':' << info.st_ctim.tv_nsec;
#endif
#endif
    return key.str();
}
inline std::string modelIdentity(const std::filesystem::path &path) {
    const auto name = path.string();
    const auto bytes = std::filesystem::file_size(path);
    const auto metadata = modelMetadataIdentity(path);
    // Metadata tells us when to re-read, not whether model bytes changed.
    // iOS data-protection attributes can change ctime during inference.
    struct CachedIdentity { std::string metadata; std::string content; };
    static std::mutex mutex;
    static std::unordered_map<std::string, CachedIdentity> cache;
    // Generation references files by metadata by default; do not scan a large
    // checkpoint before loading its tensors. Offline integrity tools can request
    // content identity explicitly. Unknown policies retain strict validation.
    if (const auto *policy = std::getenv("IILD_MODEL_VALIDATION");
        !policy || std::string_view(policy) == "metadata") return metadata;
    {
        const std::lock_guard lock(mutex);
        const auto found = cache.find(name);
        if (found != cache.end() && found->second.metadata == metadata) return found->second.content;
    }
    // FNV-1a is already used for the cache. Hash every byte so same-size writes
    // with restored mtime are detected; this is not an authentication signature.
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("The local model is no longer available.");
    std::array<char, 64 * 1024> buffer{};
    std::uint64_t content = 14695981039346656037ull;
    while (input.read(buffer.data(), buffer.size()) || input.gcount()) {
        for (std::streamsize i = 0; i < input.gcount(); ++i) {
            content ^= static_cast<unsigned char>(buffer[static_cast<std::size_t>(i)]);
            content *= 1099511628211ull;
        }
    }
    if (!input.eof()) throw std::runtime_error("Cannot read the local model identity.");
    std::ostringstream digest;
    digest.imbue(std::locale::classic());
    digest << name << ':' << bytes << ":data=" << std::hex << content;
    const auto identity = digest.str();
    {
        const std::lock_guard lock(mutex);
        if (cache.size() >= 128 && !cache.contains(name)) cache.clear();
        cache[name] = {metadata, identity};
    }
    if (std::getenv("IILD_NATIVE_DIAGNOSTICS"))
        std::fprintf(stderr, "iiLocalDiffusion identity: metadata=%s content=%s\n", metadata.c_str(), identity.c_str());
    return identity;
}
#if defined(__APPLE__)
ResourceLimits resourceLimits();
#endif
}
