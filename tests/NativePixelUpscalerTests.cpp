#include "core/tensor.hpp"
#include "stable-diffusion.h"
#include <iostream>
#include <stdexcept>

void check(bool condition) {
    if (!condition) throw std::runtime_error("Native pixel upscaler contract failed");
}
int main() {
    try {
        using Mode = sd::ops::InterpolateMode;
        check(std::string(sd_hires_upscaler_name(SD_HIRES_UPSCALER_BILINEAR)) == "Bilinear");
        check(std::string(sd_hires_upscaler_name(SD_HIRES_UPSCALER_BICUBIC)) == "Bicubic");
        const sd::Tensor<float> checker({2, 2, 1, 1}, {0, 1, 1, 0});
        const auto nearest = sd::ops::interpolate(checker, {4, 4, 1, 1}, Mode::Nearest);
        const auto linear = sd::ops::interpolate(checker, {4, 4, 1, 1}, Mode::Bilinear);
        const auto cubic = sd::ops::interpolate(checker, {4, 4, 1, 1}, Mode::Bicubic);
        check(nearest[5] == 0);
        check(std::abs(linear[5] - 0.375f) < 1e-6f);
        check(std::abs(cubic[5] - linear[5]) > 1e-3f);
        for (const auto mode : {Mode::Nearest, Mode::Bilinear, Mode::Bicubic, Mode::Lanczos}) {
            const sd::Tensor<float> constant({2, 3, 3, 1}, std::vector<float>(18, 0.37f));
            const auto enlarged = sd::ops::interpolate(constant, {4, 6, 3, 1}, mode);
            check(enlarged.shape() == std::vector<int64_t>({4, 6, 3, 1}));
            for (const float pixel : enlarged.values()) check(std::abs(pixel - 0.37f) < 1e-5f);
            const auto bounded = sd::ops::clamp(sd::ops::interpolate(checker, {4, 4, 1, 1}, mode), 0.f, 1.f);
            for (const float pixel : bounded.values()) check(std::isfinite(pixel) && pixel >= 0 && pixel <= 1);
        }
        std::cout << "Native pixel interpolation contracts passed\n";
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
