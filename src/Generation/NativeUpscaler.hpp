#pragma once
#include "Export.hpp"
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <vector>

namespace iiLocalDiffusion {
// Interleaved, unpremultiplied 8-bit gray, RGB, or RGBA. Storage is owned.
struct NativeGenerationResult;
struct UpscaleImage {
    std::uint32_t width = 0, height = 0, channels = 0;
    std::vector<std::uint8_t> pixels;
};
struct UpscaleOptions {
    double scale = 0.0; // 0 chooses the session default; otherwise applied on every pass.
    std::uint32_t targetWidth = 0, targetHeight = 0; // Explicit target requires one pass.
    int passes = 1;
    std::uint64_t maxOutputBytes = 512ULL * 1024 * 1024;
    // Provider-specific, passed unchanged. Built-in providers reject unknown keys.
    std::map<std::string, std::string> settings;
};
struct UpscalePass { std::uint32_t width = 0, height = 0; };
struct UpscaleExecution {
    UpscaleImage image;
    std::string upscaler;
    std::vector<UpscalePass> passes;
    std::map<std::string, std::string> details;
    double appliedScale = 0; // 0 when explicit target dimensions were used.
};
class IILD_EXPORT UpscaleCancelled : public std::runtime_error {
public:
    UpscaleCancelled();
};
class IILD_EXPORT UpscalerSession {
public:
    virtual ~UpscalerSession() = default;
    // Exactly the requested size and original channel count must be returned.
    virtual UpscaleImage process(const UpscaleImage &, std::uint32_t width, std::uint32_t height,
                                 std::stop_token) = 0;
    virtual std::map<std::string, std::string> details() const { return {}; }
    virtual double defaultScale() const { return 2.0; }
};
class IILD_EXPORT Upscaler {
public:
    virtual ~Upscaler() = default;
    virtual std::string name() const = 0;
    virtual void validateInput(const UpscaleImage &) const {}
    // A call owns a separate session; resources are released on all exits.
    virtual std::unique_ptr<UpscalerSession> open(const UpscaleOptions &) = 0;
};
enum class UpscaleFilter { Nearest, Bilinear, Bicubic, Lanczos3 };
enum class UpscaleDevice { Automatic, Cpu, Metal, DirectX, Vulkan };
struct EsrganUpscalerOptions {
    std::filesystem::path model;
    UpscaleDevice device = UpscaleDevice::Automatic;
    int tileSize = 128; // 0 disables tiling.
    int threads = 1;
};
enum class UpscaleModelArchitecture { Automatic, SRVGG, SwinIR };
enum class SrvggActivation { PRelu, Relu, LeakyRelu };
struct NativeModelUpscalerOptions {
    std::filesystem::path model;
    UpscaleModelArchitecture architecture = UpscaleModelArchitecture::Automatic;
    UpscaleDevice device = UpscaleDevice::Automatic;
    int tileSize = 0; // Core size in input pixels; 0 processes the whole image.
    int threads = 1;
    SrvggActivation srvggActivation = SrvggActivation::PRelu;
    float swinIRImageRange = 1.0f;
    float swinIRQkScale = 0.0f; // 0 uses 1/sqrt(head dimension).
    std::uint64_t maxWeightBytes = 1024ULL * 1024 * 1024;
    std::uint64_t maxWorkingBytes = 512ULL * 1024 * 1024;
};
// Native model execution from compatible .safetensors weights.
IILD_EXPORT std::unique_ptr<Upscaler> makeNativeModelUpscaler(NativeModelUpscalerOptions);
IILD_EXPORT std::unique_ptr<Upscaler> makeInterpolationUpscaler(UpscaleFilter);
// Compatible ESRGAN/RRDB safetensors or GGUF only; no model is downloaded.
IILD_EXPORT std::unique_ptr<Upscaler> makeEsrganUpscaler(EsrganUpscalerOptions);
// Synchronous; run on an application-owned worker when desired. Return false
// from progress or request stop to cancel. No callback runs under an engine lock.
using UpscaleProgress = std::function<bool(int completedPasses, int totalPasses)>;
IILD_EXPORT UpscaleExecution upscaleImage(UpscaleImage, Upscaler &, const UpscaleOptions &,
                                         std::stop_token = {}, const UpscaleProgress & = {});
// Consume a successful generation result without altering generation defaults.
IILD_EXPORT UpscaleExecution upscaleGeneratedImage(NativeGenerationResult, Upscaler &, const UpscaleOptions &,
                                                  std::stop_token = {}, const UpscaleProgress & = {});
}
