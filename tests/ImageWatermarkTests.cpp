#include "Generation/ImageWatermark.hpp"
#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace iiLocalDiffusion;
namespace {
void require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
}
int main()
{
    try {
        std::string error;
        std::vector<uint8_t> pixels(4 * 4 * 4, 0);
        for (size_t i = 3; i < pixels.size(); i += 4) pixels[i] = 255;
        const std::array<uint8_t, 4> white{255, 255, 255, 255};
        RgbaImageView target{pixels, 4, 4, 16};
        const RgbaImageSource mark{white, 1, 1, 4};
        require(compositeImageWatermark(target, mark, {2, 2, 1, 1, 0.5}, &error), "valid composition rejected");
        require(pixels[40] == 128 && pixels[43] == 255, "half-opacity source-over incorrect");
        require(pixels[0] == 0 && pixels[60] == 0, "pixels outside mark changed");
        const auto before = pixels;
        for (const auto region : {WatermarkPlacement{-1, 0, 1, 1, 1}, {0, 0, 0, 1, 1},
                {3, 3, 2, 2, 1}, {0, 0, 1, 1, -0.1}, {0, 0, 1, 1, 1.1},
                {0, 0, 1, 1, std::numeric_limits<double>::quiet_NaN()}}) {
            require(!compositeImageWatermark(target, mark, region, &error), "invalid placement accepted");
            require(!error.empty() && pixels == before, "failure mutated target or omitted error");
        }
        require(!compositeImageWatermark({pixels, 4, 4, 15}, mark, {0, 0, 1, 1, 1}, &error), "short stride accepted");
        require(!compositeImageWatermark({std::span(pixels).first(63), 4, 4, 16}, mark, {0, 0, 1, 1, 1}, &error), "short buffer accepted");
        require(!compositeImageWatermark(target, {pixels, 4, 4, 16}, {0, 0, 1, 1, 1}, &error), "aliased source accepted");
        require(!compositeImageWatermark(target, {white, 1, 2, SIZE_MAX}, {0, 0, 1, 1, 1}, &error), "overflow accepted");
        require(pixels == before, "validation was not atomic");
        require(!compositeImageWatermark(target, {{}, 1, 1, 4}, {0, 0, 1, 1, 1}, &error), "empty source accepted");
        const std::array<uint8_t, 4> invisible{255, 123, 87, 0};
        require(compositeImageWatermark(target, {invisible, 1, 1, 4}, {0, 0, 4, 4, 1}, &error), "transparent mark rejected");
        require(pixels == before, "transparent mark changed hidden colors");
        require(compositeImageWatermark(target, mark, {0, 0, 4, 4, 0}, &error) && error.empty(), "zero opacity rejected");
        require(pixels == before, "zero opacity changed pixels");
        // Transparent colors must not bleed into interpolated edges.
        const std::array<uint8_t, 8> edge{255, 0, 0, 0, 0, 255, 0, 255};
        std::array<uint8_t, 8> transparent{77, 88, 99, 0, 31, 32, 33, 0};
        require(compositeImageWatermark({transparent, 2, 1, 8}, {edge, 2, 1, 8}, {0, 0, 1, 1, 1}, &error), "resampling failed");
        require(transparent[0] == 0 && transparent[1] == 255 && transparent[2] == 0 && transparent[3] == 128,
            "premultiplied interpolation or target alpha incorrect");
        require(transparent[4] == 31 && transparent[7] == 0, "outside transparent pixel changed");
        std::array<uint8_t, 4> translucentRed{255, 0, 0, 128};
        const std::array<uint8_t, 4> translucentBlue{0, 0, 255, 128};
        require(compositeImageWatermark({translucentRed, 1, 1, 4}, {translucentBlue, 1, 1, 4},
            {0, 0, 1, 1, 1}, &error), "translucent destination rejected");
        require(translucentRed == std::array<uint8_t, 4>{85, 0, 170, 192}, "source-over alpha incorrect");
        // Padding belongs to the caller; source stride and output stride are independent.
        std::array<uint8_t, 12> padded{0, 0, 0, 255, 9, 9, 9, 9, 0, 0, 0, 255};
        require(compositeImageWatermark({padded, 1, 2, 8}, mark, {0, 0, 1, 2, 1}, &error), "padded target rejected");
        require(padded[0] == 255 && padded[8] == 255 && padded[4] == 9, "padding changed");
        std::cout << "Image watermark contracts passed\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
