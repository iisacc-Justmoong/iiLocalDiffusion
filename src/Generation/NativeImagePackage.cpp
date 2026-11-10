#include "NativeDiffusion.hpp"
#include "ImageModelRouting.hpp"
#include "NativeModelInspection.hpp"
#include "ModelPackaging/ModelPackaging.hpp"
#include <algorithm>
#include <chrono>
#include <limits>
#include <stop_token>
#include <stdexcept>

namespace iiLocalDiffusion {
NativeImagePackage prepareNativeImagePackage(const NativeGenerationRequest &request,
    const std::filesystem::path &cacheDirectory, const NativeModelComponents &components,
    const std::atomic_bool &cancelled, const NativeProgressCallback &progress,
    const std::shared_ptr<NativeExecutionControl> &control)
{
    const auto check = [&] {
        if (cancelled || (control && !control->waitUntilRunnable(cancelled)))
            throw std::runtime_error("Model package preparation was cancelled.");
    };
    check();
    NativeImagePackage result{request, components, false};
    if (inspectImageModel(request.modelPath).architecture != ImageModelArchitecture::Anima) return result;
    const auto embedded = inspectEmbeddedImageComponents(request.modelPath);
    if (!embedded.packaged || embedded.transformer.empty()) return result;
    result.packaged = true;
    const bool extractModel = native_detail::embeddedAnimaTransformerNeedsExtraction(request.modelPath);
    std::stop_source stop;
    auto lastProgress = std::chrono::steady_clock::time_point{};
    const auto observer = [&](const iild::ModelPackagingProgress &event) {
        if (cancelled || (control && !control->waitUntilRunnable(cancelled))) stop.request_stop();
        constexpr std::uint64_t unit = 1024 * 1024;
        const auto total = std::min<std::uint64_t>((event.totalBytes + unit - 1) / unit,
                                                   std::numeric_limits<int>::max());
        const auto done = event.completedBytes == event.totalBytes ? total
            : std::min<std::uint64_t>(event.completedBytes / unit, total);
        const auto now = std::chrono::steady_clock::now();
        if (progress && !cancelled && (lastProgress == std::chrono::steady_clock::time_point{}
            || event.completedBytes == event.totalBytes || now - lastProgress >= std::chrono::milliseconds(200))) {
            lastProgress = now;
            progress({NativeGenerationStage::Preparing, static_cast<int>(done), static_cast<int>(total)});
        }
    };
    const auto extract = [&](const std::string &path) {
        check();
        if (cacheDirectory.empty()) throw std::invalid_argument("Choose a cache directory for embedded image model components.");
        const std::u8string utf8(reinterpret_cast<const char8_t *>(path.data()), path.size());
        return iild::materializeModelPackageAssetByPath(request.modelPath, cacheDirectory,
            std::filesystem::path(utf8), stop.get_token(), observer);
    };
    if (extractModel) result.request.modelPath = extract(embedded.transformer);
    if (result.components.llm.empty() && !embedded.textEncoder.empty())
        result.components.llm = extract(embedded.textEncoder);
    if (result.components.vae.empty() && !embedded.vae.empty())
        result.components.vae = extract(embedded.vae);
    check();
    return result;
}
}
