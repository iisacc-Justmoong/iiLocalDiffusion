#pragma once
#include "NativeDiffusion.hpp"
#include <span>

namespace iiLocalDiffusion {
struct NativeVideoShot {
    std::string prompt, negativePrompt;
    std::int64_t seed = 0;
    int frames = 9;
    int fps = 24;
    NativeReferenceImage firstFrame, lastFrame;
    float conditioningStrength = 1;
    bool continuePrevious = false;
};
struct NativeVideoRequest {
    std::filesystem::path modelPath, textEncoderPath, assetCacheDirectory;
    int width = 512, height = 512, steps = 30, threads = 0;
    float guidanceScale = 3;
    bool cpu = false, vaeTiling = true;
    std::string backend = "auto", precision = "auto";
    bool cpuTextEncoding = false;
    std::vector<NativeVideoShot> shots;
};
struct NativeVideoResult {
    bool cancelled = false;
    std::string error;
    std::vector<int> frameCounts;
    int threads = 0;
    std::uint64_t memoryBudgetBytes = 0;
};
// Frames are borrowed until the observer returns. A single native context is
// shared by all shots and released on completion/cancellation. No Qt/Python.
using NativeVideoFrameObserver = std::function<void(std::size_t shot, int index,
    int width, int height, std::span<const std::uint8_t> rgb)>;
[[nodiscard]] IILD_EXPORT NativeVideoResult generateNativeVideo(const NativeVideoRequest &request,
    const std::atomic_bool &cancelled, const NativeProgressCallback &progress,
    const NativeVideoFrameObserver &frame);
}
