#include "ImageWatermark.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace iiLocalDiffusion {
namespace {
bool valid(int width, int height, size_t stride, size_t size)
{
    if (width <= 0 || height <= 0 || size_t(width) > SIZE_MAX / 4) return false;
    const auto row = size_t(width) * 4;
    return stride >= row && size >= row && size_t(height - 1) <= (size - row) / stride;
}
std::array<double, 4> sample(RgbaImageSource mark, double x, double y)
{
    x = std::clamp(x, 0., double(mark.width - 1));
    y = std::clamp(y, 0., double(mark.height - 1));
    const int x0 = int(x), y0 = int(y);
    const int x1 = std::min(x0 + 1, mark.width - 1), y1 = std::min(y0 + 1, mark.height - 1);
    const double fx = x - x0, fy = y - y0;
    std::array<double, 4> result{};
    for (int row = 0; row < 2; ++row) for (int col = 0; col < 2; ++col) {
        const double weight = (col ? fx : 1 - fx) * (row ? fy : 1 - fy);
        const auto offset = size_t(row ? y1 : y0) * mark.stride + size_t(col ? x1 : x0) * 4;
        const double alpha = mark.bytes[offset + 3] / 255.;
        for (int channel = 0; channel < 3; ++channel)
            result[channel] += weight * alpha * mark.bytes[offset + channel] / 255.;
        result[3] += weight * alpha;
    }
    return result;
}
uint8_t byte(double value) { return uint8_t(std::lround(std::clamp(value, 0., 1.) * 255)); }
}
bool compositeImageWatermark(RgbaImageView destination, RgbaImageSource mark,
    WatermarkPlacement placement, std::string *error)
{
    if (error) error->clear();
    const auto fail = [&](const char *message) { if (error) *error = message; return false; };
    if (!destination.bytes.data() || !mark.bytes.data()
        || !valid(destination.width, destination.height, destination.stride, destination.bytes.size())
        || !valid(mark.width, mark.height, mark.stride, mark.bytes.size()))
        return fail("Watermark requires valid RGBA8 buffers and row strides.");
    if (placement.x < 0 || placement.y < 0 || placement.width <= 0 || placement.height <= 0
        || placement.x > destination.width || placement.y > destination.height
        || placement.width > destination.width - placement.x || placement.height > destination.height - placement.y
        || !std::isfinite(placement.opacity) || placement.opacity < 0 || placement.opacity > 1)
        return fail("Watermark rectangle must fit the image and opacity must be in [0, 1].");
    const auto targetAddress = reinterpret_cast<uintptr_t>(destination.bytes.data());
    const auto sourceAddress = reinterpret_cast<uintptr_t>(mark.bytes.data());
    if ((targetAddress >= sourceAddress && targetAddress - sourceAddress < mark.bytes.size())
        || (sourceAddress >= targetAddress && sourceAddress - targetAddress < destination.bytes.size()))
        return fail("Watermark source and destination must not overlap.");
    if (placement.opacity == 0) return true;
    for (int y = 0; y < placement.height; ++y) for (int x = 0; x < placement.width; ++x) {
        const auto source = sample(mark, (x + .5) * mark.width / placement.width - .5,
            (y + .5) * mark.height / placement.height - .5);
        const double alpha = source[3] * placement.opacity;
        if (alpha == 0) continue; // Preserve even hidden RGB when no coverage exists.
        const auto offset = size_t(placement.y + y) * destination.stride + size_t(placement.x + x) * 4;
        auto *pixel = destination.bytes.data() + offset;
        const double remaining = pixel[3] / 255. * (1 - alpha);
        const double outputAlpha = alpha + remaining;
        for (int channel = 0; channel < 3; ++channel)
            pixel[channel] = byte((source[channel] * placement.opacity + pixel[channel] / 255. * remaining) / outputAlpha);
        pixel[3] = byte(outputAlpha);
    }
    return true;
}
}
