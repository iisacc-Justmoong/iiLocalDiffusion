#pragma once
#include <functional>
#include <stop_token>
namespace iiLocalDiffusion::native_detail {
using UpscaleStopProbe = std::function<bool()>;
const UpscaleStopProbe *activeUpscaleStopProbe() noexcept;
bool upscaleStopRequested(std::stop_token, const UpscaleStopProbe *) noexcept;
class UpscaleStopScope {
    const UpscaleStopProbe *previous_;
public:
    explicit UpscaleStopScope(const UpscaleStopProbe *) noexcept;
    ~UpscaleStopScope();
    UpscaleStopScope(const UpscaleStopScope &) = delete;
    UpscaleStopScope &operator=(const UpscaleStopScope &) = delete;
};
}
