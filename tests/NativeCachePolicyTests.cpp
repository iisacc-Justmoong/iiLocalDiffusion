#include "Generation/NativeCachePolicy.hpp"
#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
    using namespace iiLocalDiffusion::native_detail;
    if (argc != 2) return 1;
    constexpr auto GiB = 1024ull * 1024 * 1024;
    // An 8 GB phone must use more than the old fixed 2 GiB without spending
    // UIKit's headroom; a low-memory device must never be rounded upward.
    if (memoryBudget({8 * GiB, 5 * GiB, 6 * GiB}) != 4864ull * 1024 * 1024) return 2;
    if (memoryBudget({4 * GiB, 3 * GiB, 1536ull * 1024 * 1024}) > GiB) return 3;
    if (memoryBudget({8 * GiB, 5 * GiB, 256ull * 1024 * 1024}) != 0) return 4;
    if (memoryBudget({8 * GiB, 5 * GiB, 0, true}) != 0) return 8;
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
    const auto replacement = dir / "replacement";
    { std::ofstream out(replacement); out << "replaced"; }
    std::filesystem::last_write_time(replacement, time);
    std::filesystem::rename(replacement, model);
    if (first == modelIdentity(model)) return 6; // Same size/mtime, different file.
    std::filesystem::remove(model);
    bool missingRejected = false;
    try { modelIdentity(model); } catch (...) { missingRejected = true; }
    if (!missingRejected) return 7;
    std::cout << "resource headroom and source identity invalidation passed\n";
}
