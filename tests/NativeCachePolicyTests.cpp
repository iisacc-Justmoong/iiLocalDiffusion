#include "Generation/NativeCachePolicy.hpp"
#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
    using namespace iiLocalDiffusion::native_detail;
    if (argc != 2) return 1;
    constexpr auto GiB = 1024ull * 1024 * 1024;
    // Preserve the desktop working-set policy. Low-memory budgets must never
    // be rounded upward; the iOS shared-memory policy is tested separately.
    if (memoryBudget({8 * GiB, 5 * GiB, 6 * GiB}) != 4864ull * 1024 * 1024) return 2;
    if (memoryBudget({4 * GiB, 3 * GiB, 1536ull * 1024 * 1024}) > GiB) return 3;
    if (memoryBudget({8 * GiB, 5 * GiB, 256ull * 1024 * 1024}) != 0) return 4;
    if (memoryBudget({8 * GiB, 5 * GiB, 0, true}) != 0) return 8;
    // Observed iPhone 15 Pro Max limits: leave room for CPU VAE, mapped files,
    // UIKit and the GPU driver instead of filling the shared working set.
    if (memoryBudget({8027406336, 5726633984, 5684492088, true}, true) != 3 * GiB) return 16;
    if (memoryBudget({8 * GiB, 5 * GiB, 0, true}, true) != 0) return 17;
    if (memoryBudget({4 * GiB, 3 * GiB, 1536ull * 1024 * 1024, true}, true) != 0) return 18;
    if (memoryBudget({8 * GiB, 5 * GiB, 2 * GiB, true}, true) != GiB / 2) return 19;
    if (memoryBudget({16 * GiB, 11 * GiB, 10 * GiB, true}, true) != 6 * GiB) return 20;
    if (diagnosticMemoryCeiling(4 * GiB, "2048") != 2 * GiB) return 9;
    if (diagnosticMemoryCeiling(4 * GiB, "8192") != 4 * GiB) return 10;
    for (const auto invalid : {"", "0", "511", "oops", "2048oops", "18446744073709551615"})
        if (diagnosticMemoryCeiling(4 * GiB, invalid) != 4 * GiB) return 11;
    const auto dir = std::filesystem::path(argv[1]);
    std::filesystem::create_directories(dir);
    const auto model = dir / "model.safetensors";
    { std::ofstream out(model); out << "original"; }
    const auto first = modelIdentity(model);
    if (first != modelIdentity(model)) return 5;
    const auto time = std::filesystem::last_write_time(model);
    std::filesystem::permissions(model, std::filesystem::perms::owner_read | std::filesystem::perms::owner_write);
    if (first != modelIdentity(model)) {
        std::cerr << "Metadata-only permission changes invalidated unchanged model data\n";
        return 12;
    }
    std::filesystem::last_write_time(model, time + std::chrono::seconds(1));
    if (first != modelIdentity(model)) return 14; // Timestamps do not change model bytes.
    { std::fstream file(model, std::ios::in | std::ios::out); file << "modified"; }
    std::filesystem::last_write_time(model, time);
    if (first == modelIdentity(model)) return 13; // Same-size content writes with restored mtime are detected.
    const auto written = modelIdentity(model);
    const auto replacement = dir / "replacement";
    { std::ofstream out(replacement); out << "modified"; }
    std::filesystem::last_write_time(replacement, time);
    std::filesystem::rename(replacement, model);
    if (written != modelIdentity(model)) return 15; // Identical replacement keeps the content identity.
    { std::ofstream file(replacement); file << "replaced"; }
    std::filesystem::last_write_time(replacement, time);
    std::filesystem::rename(replacement, model);
    if (written == modelIdentity(model)) return 6; // Different bytes, same size/mtime.
    std::filesystem::remove(model);
    bool missingRejected = false;
    try { modelIdentity(model); } catch (...) { missingRejected = true; }
    if (!missingRejected) return 7;
    std::cout << "resource headroom and source identity invalidation passed\n";
}
