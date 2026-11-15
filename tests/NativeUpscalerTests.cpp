#include "Generation/NativeUpscaler.hpp"
#include "Generation/NativeDiffusion.hpp"
#include <algorithm>
#include <iostream>
#include <future>
#include <limits>
#include <stdexcept>
#include <string>
using namespace iiLocalDiffusion;

static void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
template<class Function> static void fails(Function fn) {
    try { fn(); } catch (const std::exception &) { return; }
    throw std::runtime_error("Expected an error");
}
struct Custom final : Upscaler {
    int opened = 0, called = 0, closed = 0;
    bool wrongShape = false, invalidDefault = false;
    struct Session final : UpscalerSession {
        Custom &owner;
        explicit Session(Custom &value) : owner(value) {}
        ~Session() override { ++owner.closed; }
        double defaultScale() const override { return owner.invalidDefault ? 0 : 3; }
        UpscaleImage process(const UpscaleImage &input, std::uint32_t width, std::uint32_t height,
                             std::stop_token) override {
            ++owner.called;
            if (owner.wrongShape) return input;
            return {width, height, input.channels, std::vector<std::uint8_t>(std::size_t(width) * height * input.channels, 42)};
        }
    };
    std::string name() const override { return "custom"; }
    std::unique_ptr<UpscalerSession> open(const UpscaleOptions &options) override {
        require(options.settings.at("model_option") == "unchanged", "Custom options were altered");
        ++opened; return std::make_unique<Session>(*this);
    }
};
int main() {
    try {
#if !defined(IILD_HAS_NATIVE_DIFFUSION) || IILD_HAS_NATIVE_DIFFUSION
        auto hiresEntry = &generateNativeAdvancedImageWithUpscaler;
        require(hiresEntry != nullptr, "Hires adapter entry point missing");
#endif
        const UpscaleImage checker{2, 2, 1, {0, 255, 255, 0}};
        UpscaleOptions options; options.scale = 2;
        auto nearest = makeInterpolationUpscaler(UpscaleFilter::Nearest);
        auto output = upscaleImage(checker, *nearest, options);
        const auto defaults = upscaleImage(checker, *nearest, UpscaleOptions{});
        require(defaults.appliedScale == 2 && defaults.image.width == 4, "Default scale was not resolved");
        require(output.image.width == 4 && output.image.height == 4 && output.image.channels == 1,
                "Nearest output dimensions");
        require(output.image.pixels == std::vector<std::uint8_t>({0,0,255,255,0,0,255,255,255,255,0,0,255,255,0,0}),
                "Nearest interpolation");
        require(output.passes.size() == 1 && output.upscaler == "nearest", "Execution report");
        NativeGenerationResult generated; generated.width=2; generated.height=1;
        generated.rgb={1,2,3,4,5,6};
        require(upscaleGeneratedImage(generated,*nearest,options).image.width == 4, "Generated output is not connected");
        generated.error="generation failed";
        fails([&]{ (void)upscaleGeneratedImage(generated,*nearest,options); });
        for (auto filter : {UpscaleFilter::Bilinear, UpscaleFilter::Bicubic, UpscaleFilter::Lanczos3}) {
            auto upscaler = makeInterpolationUpscaler(filter);
            const auto result = upscaleImage({2, 2, 4, std::vector<std::uint8_t>(16, 97)}, *upscaler, options);
            require(std::all_of(result.image.pixels.begin(), result.image.pixels.end(), [](auto p){ return p == 97; }),
                    "Interpolation must preserve a constant RGBA image");
        }
        auto linear = makeInterpolationUpscaler(UpscaleFilter::Bilinear);
        const auto linearPixel = upscaleImage(checker, *linear, options).image.pixels[5];
        auto cubic = makeInterpolationUpscaler(UpscaleFilter::Bicubic);
        auto lanczos = makeInterpolationUpscaler(UpscaleFilter::Lanczos3);
        const auto cubicPixel = upscaleImage(checker, *cubic, options).image.pixels[5];
        const auto lanczosPixel = upscaleImage(checker, *lanczos, options).image.pixels[5];
        require(linearPixel == 96, "Bilinear pixel center");
        require(cubicPixel != linearPixel && lanczosPixel != cubicPixel, "Distinct filters must use their own kernels");
        options.targetWidth = 5; options.targetHeight = 3;
        output = upscaleImage(checker, *nearest, options);
        require(output.image.width == 5 && output.image.height == 3, "Explicit dimensions");
        options.targetWidth = 0; options.targetHeight = 0; options.passes = 2;
        output = upscaleImage(checker, *nearest, options);
        require(output.image.width == 8 && output.passes.size() == 2 && output.passes[0].width == 4,
                "Repeat count and report");
        Custom custom; options.settings["model_option"] = "unchanged";
        int events = 0;
        output = upscaleImage(checker, custom, options, {}, [&](int pass, int total) {
            ++events; require(pass == events && total == 2, "Progress mismatch"); return true;
        });
        require(custom.opened == 1 && custom.called == 2 && custom.closed == 1 && events == 2,
                "Injected upscaler session lifetime");
        custom.wrongShape = true; fails([&]{ (void)upscaleImage(checker, custom, options); });
        require(custom.closed == 2, "A failed session leaked");
        custom.wrongShape = false;
        fails([&]{ (void)upscaleImage(checker, custom, options, {}, [](int, int){ return false; }); });
        require(custom.closed == 3, "A cancelled session leaked");
        const auto opens = custom.opened;
        options.maxOutputBytes = 5;
        fails([&]{ (void)upscaleImage(checker, custom, options); });
        require(custom.opened == opens, "Invalid output must fail before opening a backend");
        options.maxOutputBytes = 512 * 1024 * 1024; options.passes = 1;
        options.scale = std::numeric_limits<double>::infinity();
        fails([&]{ (void)upscaleImage(checker, custom, options); });
        options.scale = 2; options.targetWidth = 4; options.targetHeight = 0;
        fails([&]{ (void)upscaleImage(checker, custom, options); });
        options.targetWidth = 0;
        fails([&]{ (void)upscaleImage({2,2,3,{1,2}}, custom, options); });
        Custom defaultsProvider;
        UpscaleOptions automatic; automatic.settings["model_option"] = "unchanged";
        const auto autoResult = upscaleImage(checker, defaultsProvider, automatic);
        require(autoResult.image.width == 6 && autoResult.appliedScale == 3, "Provider default was not applied");
        defaultsProvider.invalidDefault = true;
        fails([&]{ (void)upscaleImage(checker, defaultsProvider, automatic); });
        require(defaultsProvider.closed == 2 && defaultsProvider.called == 1, "Invalid provider default ran or leaked a session");
        NativeExecutionControl paused;
        paused.setPaused(true);
        std::atomic_bool cancelled{false}; std::stop_source pausedStop; std::promise<void> entered;
        auto waiting=std::async(std::launch::async,[&] { entered.set_value(); return paused.waitUntilRunnable(cancelled,pausedStop.get_token()); });
        entered.get_future().wait(); pausedStop.request_stop();
        require(!waiting.get(), "A stopped Hires operation remained paused");
        std::stop_source stop; stop.request_stop();
        fails([&]{ (void)upscaleImage(checker, custom, options, stop.get_token()); });
        fails([&]{ (void)makeEsrganUpscaler({"bad.onnx"}); });
        auto unavailable = makeEsrganUpscaler({"absent.safetensors"});
        bool invalidRgb = false;
        try { (void)upscaleImage(checker, *unavailable, UpscaleOptions{}); }
        catch (const std::invalid_argument &e) {
            invalidRgb = std::string(e.what()).find("RGB") != std::string::npos;
        }
        require(invalidRgb, "Input should be rejected before model loading");
        std::cout << "Upscaler contract passed\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
