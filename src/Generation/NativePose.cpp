#include "NativePose.hpp"
#include <stdexcept>

namespace iiLocalDiffusion {
namespace { [[noreturn]] void unavailable() { throw std::runtime_error("Automatic pose inference has been retired; supply an already prepared control hint."); } }
struct NativePoseSession::Impl {};
NativePoseSession::NativePoseSession(const std::filesystem::path &, const std::filesystem::path &,
    const std::atomic_bool &, unsigned, const std::shared_ptr<NativeExecutionControl> &) { unavailable(); }
NativePoseSession::~NativePoseSession() = default;
bool nativePoseAvailable() noexcept { return false; }
std::uint64_t NativePoseSession::modelBytes() const noexcept { return 0; }
unsigned NativePoseSession::threads() const noexcept { return 0; }
NativeReferenceImage NativePoseSession::process(const NativeReferenceImage &, const std::atomic_bool &,
    const std::shared_ptr<NativeExecutionControl> &) { unavailable(); }
NativePoseResult processNativePose(const std::filesystem::path &, const std::filesystem::path &,
    const NativeReferenceImage &, const std::atomic_bool &, unsigned,
    const std::shared_ptr<NativeExecutionControl> &) { unavailable(); }
void releaseNativePoseCache() noexcept {}
}
