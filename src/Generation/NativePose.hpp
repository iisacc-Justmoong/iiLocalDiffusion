#pragma once
#include "NativeDiffusion.hpp"

namespace iiLocalDiffusion {
// One owner retains both fully-read model byte arrays and native CPU sessions.
// No mmap, external tensor file, optimized-model cache, subprocess or downloads.
// Keep this owner in the application runtime, including between queued jobs and
// pause/idle transitions; destroy only for explicit release or runtime teardown.
class IILD_EXPORT NativePoseSession final {
public:
    NativePoseSession(const std::filesystem::path &detector, const std::filesystem::path &pose,
        const std::atomic_bool &cancelled, unsigned threads = 0,
        const std::shared_ptr<NativeExecutionControl> &control = {});
    ~NativePoseSession();
    NativePoseSession(const NativePoseSession &) = delete;
    NativePoseSession &operator=(const NativePoseSession &) = delete;
    // Serializes requests on this owner. Cooperative pause between model runs;
    // an in-flight ONNX Run is terminated on cancellation, never evicted on pause.
    NativeReferenceImage process(const NativeReferenceImage &image, const std::atomic_bool &cancelled,
        const std::shared_ptr<NativeExecutionControl> &control = {});
    std::uint64_t modelBytes() const noexcept;
    unsigned threads() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
IILD_EXPORT bool nativePoseAvailable() noexcept;
struct NativePoseResult {
    NativeReferenceImage image;
    bool modelCacheHit = false;
    std::uint64_t modelBytes = 0;
    unsigned threads = 0;
};
// Runtime-wide residency, independent of the selected diffusion checkpoint.
// An absolute model pair is pinned until explicit release; replacing files at
// those paths requires release. Cache hits perform no source-file I/O or stat.
IILD_EXPORT NativePoseResult processNativePose(const std::filesystem::path &detector,
    const std::filesystem::path &pose, const NativeReferenceImage &image,
    const std::atomic_bool &cancelled, unsigned threads = 0,
    const std::shared_ptr<NativeExecutionControl> &control = {});
// Nonblocking. An active request finishes or cancels before deferred release.
IILD_EXPORT void releaseNativePoseCache() noexcept;
}
