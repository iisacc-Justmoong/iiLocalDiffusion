#include "runtime/control_region_mask.hpp"
#include <iostream>
#include <stdexcept>

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
int main() {
    try {
        ControlRegionMask mask;
        std::vector<uint8_t> pixels{0, 255};
        require(mask.set({2, 1, 1, pixels.data()}), "Valid mask rejected");
        pixels = {255, 0}; // Setter must own its source pixels.
        std::vector<sd::Tensor<float>> tensors{
            sd::Tensor<float>({4, 1, 3, 2}, std::vector<float>(24, 4.f)),
            sd::Tensor<float>({1, 2, 1, 1}, {8.f, -8.f})};
        require(mask.apply(tensors), "Mask application failed");
        const float oracle[]{0, 1, 3, 4};
        for (size_t i = 0; i < 24; ++i)
            require(std::abs(tensors[0].values()[i] - oracle[i % 4]) < 1e-6f,
                "Pixel-center interpolation or feature/batch broadcast is incorrect");
        require(tensors[1].values() == std::vector<float>({4, -4}), "Downsampled coverage is incorrect");
        std::vector<sd::Tensor<float>> repeat{sd::Tensor<float>({4, 1, 1, 1}, {4, 4, 4, 4})};
        require(mask.apply(repeat) && repeat[0].values() == std::vector<float>({0, 1, 3, 4}), "Cached scale changed values");
        const auto* attention = mask.coverage(4, 1);
        require(attention && *attention == std::vector<float>({0, .25f, .75f, 1}),
            "Attention coverage differs from ControlNet residual coverage");
        require(mask.coverage(3, 2) && mask.coverage(4, 1) == attention,
            "Adding a different spatial scale invalidated borrowed attention coverage");
        require(!mask.coverage(0, 1) && !mask.coverage(1, 4097), "Invalid coverage dimensions accepted");
        std::vector<uint8_t> rgba{255, 255, 255, 0, 255, 255, 255, 128};
        require(mask.set({2, 1, 4, rgba.data()}), "RGBA mask rejected");
        repeat = {sd::Tensor<float>({2, 1, 1, 1}, {255, 255})};
        require(mask.apply(repeat) && repeat[0].values() == std::vector<float>({0, 128}), "Alpha was ignored or scale cache was stale");
        std::vector<uint8_t> rgb{255, 0, 0};
        require(mask.set({1, 1, 3, rgb.data()}), "RGB mask rejected");
        repeat = {sd::Tensor<float>({1, 1, 1, 1}, {256})};
        require(mask.apply(repeat) && repeat[0].values()[0] == 77, "Luminance coefficient is incorrect");
        for (uint8_t coverage : {uint8_t(0), uint8_t(255)}) {
            require(mask.set({1, 1, 1, &coverage}), "Constant mask rejected");
            repeat = {sd::Tensor<float>({3, 2, 1, 1}, std::vector<float>(6, -2.f))};
            require(mask.apply(repeat), "Constant mask failed");
            for (auto value : repeat[0].values())
                require(value == (coverage == 0 ? 0.f : -2.f), "Black/white endpoints changed");
        }
        require(!mask.set({1, 1, 2, rgb.data()}) && !mask.set({2049, 1, 3, rgb.data()}), "Malformed mask accepted");
        repeat = {sd::Tensor<float>({2, 1}, {4, 8}), sd::Tensor<float>({2}, {1, 2})};
        require(!mask.apply(repeat) && repeat[0].values() == std::vector<float>({4, 8}),
            "Malformed residual caused partial mutation");
        require(mask.set({}), "Clear rejected");
        require(!mask.coverage(4, 1), "Cleared mask still exposed stale attention coverage");
        require(mask.apply(repeat) && repeat[0].values() == std::vector<float>({4, 8}), "Cleared mask still applied");
        std::cout << "Native regional ControlNet mask tests passed\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
