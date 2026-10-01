#include <stable-diffusion.h>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>
// Exercise private numerical stages as well as the linked production C API.
// Rename only this translation unit's header implementation to avoid an ODR clash.
#define preprocess_canny test_header_preprocess_canny
#include "runtime/preprocessing.hpp"
#undef preprocess_canny

void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
int main()
{
    try {
        for (const auto &[angle, dx, dy] : std::vector<std::tuple<float, int, int>>{
                {10, 0, 1}, {40, 1, -1}, {80, 1, 0}, {130, 1, 1}}) {
            sd::Tensor<float> magnitude({3, 3, 1, 1}, std::vector<float>(9, 0.f));
            sd::Tensor<float> direction({3, 3, 1, 1}, std::vector<float>(9, angle * M_PI_ / 180.f));
            preprocessing_set_4d(magnitude, .5f, 1, 1);
            preprocessing_set_4d(magnitude, .8f, 1 + dx, 1 + dy);
            require(non_max_supression(magnitude, direction)[4] == 0, "Canny compared the wrong gradient direction");
            preprocessing_set_4d(magnitude, .2f, 1 + dx, 1 + dy);
            require(non_max_supression(magnitude, direction)[4] == .5f, "Canny suppressed a local gradient maximum");
        }
        sd::Tensor<float> chain({16, 16, 1, 1}, std::vector<float>(256, 0.f));
        preprocessing_set_4d(chain, 1.f, 10, 10);
        for (int i = 7; i <= 9; ++i) preprocessing_set_4d(chain, .5f, i, i);
        preprocessing_set_4d(chain, .5f, 3, 3);
        threshold_hystersis(&chain, .8f, .5f, .8f, 1.f);
        for (int i = 7; i <= 10; ++i)
            require(preprocessing_get_4d(chain, i, i) == 1.f, "Hysteresis lost a reverse-diagonal connected weak edge");
        require(preprocessing_get_4d(chain, 3, 3) == 0, "Hysteresis retained a disconnected weak edge");
        constexpr unsigned width = 64, height = 48;
        std::vector<uint8_t> rgb(width * height * 3, 0);
        sd_image_t image{width, height, 3, rgb.data()};
        require(preprocess_canny(image, .08f, .08f, .8f, 1.f, false), "Black image preprocessing failed");
        require(std::all_of(rgb.begin(), rgb.end(), [](auto v) { return v == 0; }), "Black image invented edges");
        for (unsigned y = 12; y < 36; ++y)
            for (unsigned x = 16; x < 48; ++x)
                std::fill_n(rgb.data() + (y * width + x) * 3, 3, uint8_t(255));
        const auto original = rgb;
        require(preprocess_canny(image, .08f, .08f, .8f, 1.f, false), "Rectangle preprocessing failed");
        require(rgb != original, "Canny returned the original RGB instead of edges");
        const auto edges = std::count(rgb.begin(), rgb.end(), uint8_t(255));
        require(edges > 0 && edges < 32 * 24 * 3, "Canny did not produce sparse rectangle edges");
        for (std::size_t i = 0; i < rgb.size(); i += 3)
            require((rgb[i] == 0 || rgb[i] == 255) && rgb[i] == rgb[i + 1] && rgb[i] == rgb[i + 2],
                "Canny hint is not replicated grayscale binary RGB");
        auto repeat = original;
        require(preprocess_canny({width, height, 3, repeat.data()}, .08f, .08f, .8f, 1.f, false)
            && repeat == rgb, "Canny preprocessing is not deterministic");
        std::cout << "Native ControlNet Canny preprocessing passed\n";
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
