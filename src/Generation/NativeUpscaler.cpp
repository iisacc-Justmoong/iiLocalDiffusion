#include "NativeUpscaler.hpp"
#include "NativeDiffusion.hpp"
#include "NativeEngineAccess.hpp"
#include "NativeUpscalerInternal.hpp"
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <limits>
#include <mutex>
#include <numbers>
#include <utility>

#if IILD_HAS_NATIVE_DIFFUSION
#include <stable-diffusion.h>
#endif

namespace iiLocalDiffusion::native_detail {
namespace { thread_local const UpscaleStopProbe *activeProbe = nullptr; }
const UpscaleStopProbe *activeUpscaleStopProbe() noexcept { return activeProbe; }
bool upscaleStopRequested(std::stop_token stop, const UpscaleStopProbe *probe) noexcept {
    if (stop.stop_requested()) return true;
    try { return probe && (*probe)(); } catch (...) { return true; }
}
UpscaleStopScope::UpscaleStopScope(const UpscaleStopProbe *probe) noexcept : previous_(activeProbe) { activeProbe=probe; }
UpscaleStopScope::~UpscaleStopScope() { activeProbe=previous_; }
}
namespace iiLocalDiffusion {
namespace {
std::size_t imageBytes(std::uint32_t w, std::uint32_t h, std::uint32_t c) {
    if (!w || !h || (c != 1 && c != 3 && c != 4)) throw std::invalid_argument("Image must have positive dimensions and 1, 3, or 4 channels.");
    const auto limit = std::numeric_limits<std::size_t>::max();
    if (std::size_t(w) > limit / h || std::size_t(w) * h > limit / c)
        throw std::overflow_error("Image byte count overflows.");
    return std::size_t(w) * h * c;
}
void checkStop(std::stop_token stop) {
    if (native_detail::upscaleStopRequested(stop,native_detail::activeUpscaleStopProbe())) throw UpscaleCancelled();
}

class PixelSession final : public UpscalerSession {
    UpscaleFilter filter_;
    static double kernel(UpscaleFilter filter, double x) {
        x = std::abs(x);
        if (filter == UpscaleFilter::Bilinear) return std::max(0.0, 1.0 - x);
        if (filter == UpscaleFilter::Bicubic) {
            if (x < 1) return (1.5*x - 2.5)*x*x + 1;
            if (x < 2) return ((-.5*x + 2.5)*x - 4)*x + 2;
            return 0;
        }
        if (x < 1e-12) return 1;
        if (x >= 3) return 0;
        const double pi = std::numbers::pi;
        return std::sin(pi*x)*std::sin(pi*x/3)/(pi*pi*x*x/3);
    }
public:
    explicit PixelSession(UpscaleFilter filter) : filter_(filter) {}
    UpscaleImage process(const UpscaleImage &input, std::uint32_t width, std::uint32_t height,
                         std::stop_token stop) override {
        UpscaleImage out{width, height, input.channels, std::vector<std::uint8_t>(imageBytes(width,height,input.channels))};
        const auto sample = [&](std::int64_t x, std::int64_t y, std::uint32_t ch) {
            x = std::clamp<std::int64_t>(x, 0, input.width-1);
            y = std::clamp<std::int64_t>(y, 0, input.height-1);
            return input.pixels[(std::size_t(y)*input.width + std::size_t(x))*input.channels + ch];
        };
        const int radius = filter_ == UpscaleFilter::Bilinear ? 1 : filter_ == UpscaleFilter::Bicubic ? 2 : 3;
        for (std::uint32_t y=0; y<height; ++y) {
            checkStop(stop);
            const double sy = (double(y)+.5)*input.height/height-.5;
            const auto iy = static_cast<std::int64_t>(std::floor(sy));
            std::array<double,6> wy{};
            double sumY = 0;
            if (filter_ != UpscaleFilter::Nearest) for (int dy=1-radius; dy<=radius; ++dy) {
                const auto slot = static_cast<std::size_t>(dy+radius-1);
                wy[slot] = kernel(filter_,sy-double(iy+dy)); sumY += wy[slot];
            }
            for (std::uint32_t x=0; x<width; ++x) {
                const double sx = (double(x)+.5)*input.width/width-.5;
                const auto ix = static_cast<std::int64_t>(std::floor(sx));
                std::array<double,6> wx{};
                double sumX = 0;
                if (filter_ != UpscaleFilter::Nearest) for (int dx=1-radius; dx<=radius; ++dx) {
                    const auto slot = static_cast<std::size_t>(dx+radius-1);
                    wx[slot] = kernel(filter_,sx-double(ix+dx)); sumX += wx[slot];
                }
                for (std::uint32_t ch=0; ch<input.channels; ++ch) {
                    double value = 0;
                    if (filter_ == UpscaleFilter::Nearest) {
                        value = sample(static_cast<std::int64_t>(std::floor(sx+.5)), static_cast<std::int64_t>(std::floor(sy+.5)), ch);
                    } else {
                        for (int dy=1-radius; dy<=radius; ++dy) for (int dx=1-radius; dx<=radius; ++dx) {
                            const double weight = wx[static_cast<std::size_t>(dx+radius-1)] * wy[static_cast<std::size_t>(dy+radius-1)];
                            value += weight * sample(ix+dx,iy+dy,ch);
                        }
                        value = sumX != 0.0 && sumY != 0.0 ? value/(sumX*sumY) : 0;
                    }
                    out.pixels[(std::size_t(y)*width+x)*input.channels+ch] = static_cast<std::uint8_t>(std::lround(std::clamp(value,0.0,255.0)));
                }
            }
        }
        return out;
    }
};
class PixelUpscaler final : public Upscaler {
    UpscaleFilter filter_;
public:
    explicit PixelUpscaler(UpscaleFilter f) : filter_(f) {}
    std::string name() const override {
        switch(filter_) {
            case UpscaleFilter::Nearest: return "nearest";
            case UpscaleFilter::Bilinear: return "bilinear";
            case UpscaleFilter::Bicubic: return "bicubic";
            case UpscaleFilter::Lanczos3: return "lanczos3";
        }
        throw std::invalid_argument("Unknown interpolation filter.");
    }
    std::unique_ptr<UpscalerSession> open(const UpscaleOptions &options) override {
        if (!options.settings.empty()) throw std::invalid_argument("Interpolation does not accept provider-specific settings.");
        return std::make_unique<PixelSession>(filter_);
    }
};

#if IILD_HAS_NATIVE_DIFFUSION
// The image engine owns this lock while an injected Hires provider executes.
// Do not re-enter the same non-recursive mutex on that thread.
class EngineGuard {
    std::unique_lock<std::timed_mutex> lock_{native_detail::nativeEngineMutex(), std::defer_lock};
public:
    EngineGuard() { if (!native_detail::nativeEngineOwnedByCurrentThread()) lock_.lock(); }
};
class EsrganSession final : public UpscalerSession {
    EsrganUpscalerOptions options_;
    upscaler_ctx_t *context_ = nullptr;
    int factor_ = 0;
public:
    explicit EsrganSession(EsrganUpscalerOptions options) : options_(std::move(options)) {
        const auto path = options_.model.string();
        const EngineGuard lock;
        context_ = new_upscaler_ctx(path.c_str(), false, options_.threads, options_.tileSize,
                                    options_.device == UpscaleDevice::Cpu ? "cpu" : "", "cpu");
        if (!context_) throw std::runtime_error("Cannot load a compatible ESRGAN/RRDB model.");
        factor_ = get_upscale_factor(context_);
        if (factor_ <= 1 || factor_ > 16) {
            free_upscaler_ctx(context_); context_ = nullptr;
            throw std::runtime_error("Unsupported ESRGAN model scale.");
        }
    }
    ~EsrganSession() override {
        if (context_) { const EngineGuard lock; free_upscaler_ctx(context_); }
    }
    UpscaleImage process(const UpscaleImage &image, std::uint32_t w, std::uint32_t h, std::stop_token stop) override {
        if (image.channels != 3) throw std::invalid_argument("ESRGAN requires RGB input.");
        if (std::uint64_t(image.width)*std::uint64_t(factor_) != w || std::uint64_t(image.height)*std::uint64_t(factor_) != h)
            throw std::invalid_argument("Requested size does not match the model's native scale.");
        if (w > std::uint32_t(std::numeric_limits<int>::max()) || h > std::uint32_t(std::numeric_limits<int>::max()))
            throw std::invalid_argument("Dimensions exceed the ESRGAN backend's integer range.");
        checkStop(stop);
        sd_image_t input{image.width, image.height, image.channels, const_cast<std::uint8_t *>(image.pixels.data())};
        sd_image_t *images = nullptr; int count = 0;
        {
            const EngineGuard lock;
            checkStop(stop);
            if (!::upscale(context_, input, static_cast<std::uint32_t>(factor_), &images, &count)) {
                if (images) free_sd_images(images, count);
                throw std::runtime_error("ESRGAN inference failed.");
            }
        }
        const auto cleanup = [count](sd_image_t *p) { if (p) free_sd_images(p, std::max(count, 0)); };
        std::unique_ptr<sd_image_t, decltype(cleanup)> owned(images, cleanup);
        checkStop(stop);
        if (count != 1 || !images || !images[0].data || images[0].width != w || images[0].height != h || images[0].channel != 3)
            throw std::runtime_error("ESRGAN returned an invalid image.");
        const auto n = imageBytes(w,h,3);
        return {w,h,3,std::vector<std::uint8_t>(images[0].data,images[0].data+n)};
    }
    double defaultScale() const override { return static_cast<double>(factor_); }
    std::map<std::string,std::string> details() const override {
        return {{"model", options_.model.string()}, {"model_scale",std::to_string(factor_)},
                {"requested_device", options_.device == UpscaleDevice::Cpu ? "cpu" : "automatic"},
                {"tile_size",std::to_string(options_.tileSize)}, {"threads",std::to_string(options_.threads)}};
    }
};
#endif
class EsrganUpscaler final : public Upscaler {
    EsrganUpscalerOptions options_;
public:
    explicit EsrganUpscaler(EsrganUpscalerOptions options) : options_(std::move(options)) {
        auto ext = options_.model.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
        if (ext != ".safetensors" && ext != ".gguf") throw std::invalid_argument("ESRGAN weights must be .safetensors or .gguf.");
        if ((options_.device != UpscaleDevice::Cpu && options_.device != UpscaleDevice::Automatic)
            || options_.tileSize < 0 || options_.tileSize > 8192 || options_.threads < 1 || options_.threads > 256)
            throw std::invalid_argument("Invalid ESRGAN device, tile size, or thread count.");
    }
    std::string name() const override { return "esrgan-rrdb"; }
    void validateInput(const UpscaleImage &image) const override {
        if (image.channels != 3) throw std::invalid_argument("ESRGAN requires RGB input.");
    }
    std::unique_ptr<UpscalerSession> open(const UpscaleOptions &options) override {
        if (!options.settings.empty()) throw std::invalid_argument("ESRGAN does not accept unknown settings.");
        std::error_code error;
        if (!std::filesystem::is_regular_file(options_.model,error) || error) throw std::invalid_argument("ESRGAN model file does not exist.");
#if IILD_HAS_NATIVE_DIFFUSION
        return std::make_unique<EsrganSession>(options_);
#else
        throw std::runtime_error("This build has no native ESRGAN backend.");
#endif
    }
};
}
UpscaleCancelled::UpscaleCancelled() : std::runtime_error("Upscaling cancelled.") {}
std::unique_ptr<Upscaler> makeInterpolationUpscaler(UpscaleFilter filter) {
    auto provider = std::make_unique<PixelUpscaler>(filter); (void)provider->name(); return provider;
}
std::unique_ptr<Upscaler> makeEsrganUpscaler(EsrganUpscalerOptions options) { return std::make_unique<EsrganUpscaler>(std::move(options)); }
UpscaleExecution upscaleImage(UpscaleImage image, Upscaler &upscaler, const UpscaleOptions &options,
                              std::stop_token stop, const UpscaleProgress &progress) {
    checkStop(stop);
    if (image.pixels.size() != imageBytes(image.width,image.height,image.channels)) throw std::invalid_argument("Image byte count does not match its dimensions.");
    if (options.passes < 1 || options.passes > 8 || !options.maxOutputBytes)
        throw std::invalid_argument("Pass count or output byte limit is invalid.");
    const bool target = options.targetWidth || options.targetHeight;
    if (!std::isfinite(options.scale) || (target && (!options.targetWidth || !options.targetHeight || options.passes != 1))
        || (!target && options.scale != 0 && options.scale < 1))
        throw std::invalid_argument("Specify a finite scale or both target dimensions for one pass.");
    const auto plan = [&](double scale) {
        if (!target && (!std::isfinite(scale) || scale < 1))
            throw std::invalid_argument("The upscaler default scale is invalid.");
        std::vector<UpscalePass> planned;
        auto width=image.width, height=image.height;
        for (int i=0; i<options.passes; ++i) {
            if (target) { width=options.targetWidth; height=options.targetHeight; }
            else {
                const double w = std::round(double(width)*scale), h = std::round(double(height)*scale);
                if (w > std::numeric_limits<std::uint32_t>::max() || h > std::numeric_limits<std::uint32_t>::max())
                    throw std::overflow_error("Upscale dimensions overflow.");
                width=static_cast<std::uint32_t>(w); height=static_cast<std::uint32_t>(h);
            }
            if (width < image.width || height < image.height || imageBytes(width,height,image.channels) > options.maxOutputBytes)
                throw std::invalid_argument("Upscale output exceeds its limit or would shrink the input.");
            planned.push_back({width,height});
        }
        return planned;
    };
    // Explicit plans can be rejected before allocating a backend. A model's
    // default is discovered during open and validated before its first pass.
    std::vector<UpscalePass> passes;
    if (target || options.scale != 0) passes = plan(options.scale);
    const auto name = upscaler.name();
    if (name.empty()) throw std::invalid_argument("Upscaler name is empty.");
    upscaler.validateInput(image);
    auto session = upscaler.open(options);
    if (!session) throw std::runtime_error("Upscaler did not open a session.");
    checkStop(stop);
    const double appliedScale = target ? 0 : options.scale == 0 ? session->defaultScale() : options.scale;
    if (!target && options.scale == 0) passes = plan(appliedScale);
    for (std::size_t i=0; i<passes.size(); ++i) {
        checkStop(stop);
        auto result = session->process(image,passes[i].width,passes[i].height,stop);
        checkStop(stop);
        if (result.width != passes[i].width || result.height != passes[i].height || result.channels != image.channels
            || result.pixels.size() != imageBytes(result.width,result.height,result.channels))
            throw std::runtime_error("Upscaler returned an invalid image.");
        image=std::move(result);
        if (progress && !progress(static_cast<int>(i+1),options.passes)) throw UpscaleCancelled();
        checkStop(stop);
    }
    auto details = session->details();
    return {std::move(image),name,std::move(passes),std::move(details),appliedScale};
}
UpscaleExecution upscaleGeneratedImage(NativeGenerationResult image, Upscaler &upscaler,
                                        const UpscaleOptions &options, std::stop_token stop,
                                        const UpscaleProgress &progress) {
    if (image.cancelled || !image.error.empty() || image.width <= 0 || image.height <= 0)
        throw std::invalid_argument("Cannot upscale an unsuccessful generation result.");
    return upscaleImage({static_cast<std::uint32_t>(image.width), static_cast<std::uint32_t>(image.height),
                         3, std::move(image.rgb)}, upscaler, options, stop, progress);
}
}
