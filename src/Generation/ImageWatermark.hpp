#pragma once
#include "Export.hpp"
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace iiLocalDiffusion {
// Straight-alpha RGBA8 in the same caller-selected color space. Buffers are
// borrowed only during the call; row padding is never modified.
struct RgbaImageView { std::span<std::uint8_t> bytes; int width, height; std::size_t stride; };
struct RgbaImageSource { std::span<const std::uint8_t> bytes; int width, height; std::size_t stride; };
struct WatermarkPlacement { int x, y, width, height; double opacity; };
// Resizes the supplied mark using premultiplied-alpha bilinear interpolation
// and source-over composites only the requested rectangle. All validation is
// completed before mutation. Overlapping source/destination storage is rejected.
// Brand asset, geometry and opacity are product policy, not SDK defaults.
IILD_EXPORT bool compositeImageWatermark(RgbaImageView destination, RgbaImageSource mark,
    WatermarkPlacement placement, std::string *error = nullptr);
}
