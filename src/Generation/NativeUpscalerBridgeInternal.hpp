#pragma once
#include "NativeUpscaler.hpp"
#include "NativeUpscalerBridge.hpp"

namespace iiLocalDiffusion::native_detail {
// Shared by standalone upscaling and the Hires generation bridge.
std::unique_ptr<Upscaler> makeBridgeUpscaler(const iild_upscale_request_v1 &,
    const iild_upscale_request_v2 *, const iild_upscale_request_v3 *);
UpscaleOptions bridgeUpscaleOptions(const iild_upscale_request_v1 &);
}
