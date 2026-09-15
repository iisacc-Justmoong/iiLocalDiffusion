#include "Generation/NativeVaePolicy.hpp"
#include <iostream>
#include <stdexcept>

int main() {
    using iiLocalDiffusion::native_detail::vaeDecodePolicy;
    constexpr std::uint64_t mib = 1024 * 1024;
    auto require = [](bool ok) { if (!ok) throw std::runtime_error("VAE workspace policy regression"); };
    try {
        // A larger tile improves context without borrowing the denoiser's
        // entire memory budget. Low-memory requests remain bounded.
        require(vaeDecodePolicy("sdxl-base", 1024, 1856, 3072 * mib).tile == 40);
        require(vaeDecodePolicy("sdxl-base", 1024, 1856, 2304 * mib).tile == 32);
        require(vaeDecodePolicy("sdxl-base", 1024, 1856, 1024 * mib).tile == 24);
        require(vaeDecodePolicy("sdxl-base", 1024, 1856, 512 * mib).tile == 16);
        require(vaeDecodePolicy("sdxl-base", 2048, 2048, 32768 * mib).tile == 64);
        const auto small = vaeDecodePolicy("sdxl-base", 256, 256, 2304 * mib);
        require(!small.tiled && small.overlap == 0.5f);
        require(vaeDecodePolicy("sdxl-base", 256, 320, 2304 * mib).tiled);
        for (const auto family : {"qwen-image", "flux1", "flux2", "sd15", "unknown"}) {
            const auto other = vaeDecodePolicy(family, 1024, 1024, 32768 * mib);
            require(other.tile == 32 && other.tiled);
        }
        std::cout << "SDXL memory bounds, small canvases and family isolation passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
