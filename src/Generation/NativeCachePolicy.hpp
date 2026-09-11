#pragma once
#include <algorithm>
#include <cstdint>
#include <chrono>
#include <charconv>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string_view>
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
inline std::uint64_t memoryBudget(ResourceLimits limits) {
    constexpr std::uint64_t MiB = 1024 * 1024;
    if (!limits.recommended) limits.recommended = limits.physical ? limits.physical / 2 : 2 * 1024 * MiB;
    // Keep a small Metal reserve, and independently reserve app/CPU memory
    // against the process limit. Percentage reserves can unnecessarily evict
    // an entire UNet that is only a few hundred MiB below the working-set limit.
    auto budget = limits.recommended > 256 * MiB ? limits.recommended - 256 * MiB : 0;
    if (limits.available || limits.availableKnown) {
        const auto reserve = std::max<std::uint64_t>(768 * MiB, limits.available / 6);
        budget = std::min(budget, limits.available > reserve ? limits.available - reserve : 0);
    }
    return budget / (256 * MiB) * (256 * MiB);
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
inline std::string modelIdentity(const std::filesystem::path &path) {
    std::ostringstream key;
    key << path.string() << ':' << std::filesystem::file_size(path) << ':'
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
#if defined(__APPLE__)
ResourceLimits resourceLimits();
#endif
}
