#pragma once
#include <algorithm>
#include <cstdint>
#include <string_view>

namespace iiLocalDiffusion::native_detail {
struct VaeDecodePolicy {
    int tile = 32;
    bool tiled = true;
    float overlap = 0.5f;
};

inline VaeDecodePolicy vaeDecodePolicy(std::string_view family, int width, int height,
                                     std::uint64_t memoryBudget) {
    VaeDecodePolicy policy;
    // Only SDXL has a measured CPU workspace/quality baseline. Other latent
    // architectures retain their own decoder's conservative existing policy.
    if (family != "sdxl-base" && family != "sdxl-refiner") return policy;
    constexpr std::uint64_t mib = 1024 * 1024;
    // Measured 32x32 latent tiles use ~480 MiB. Round to 512 MiB and reserve
    // two thirds of the budget for resident weights, outputs and the app.
    const auto workspace = std::min(memoryBudget / 3, 2048 * mib);
    policy.tile = 16;
    for (const int candidate : {24, 32, 40, 48, 56, 64}) {
        const auto area = static_cast<std::uint64_t>(candidate * candidate);
        if (512 * mib * area / (32 * 32) > workspace) break;
        policy.tile = candidate;
    }
    // A complete canvas avoids tile-local normalization/attention differences.
    policy.tiled = width > policy.tile * 8 || height > policy.tile * 8;
    return policy;
}
}
