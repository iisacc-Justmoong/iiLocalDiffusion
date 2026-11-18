#include "NativeVaeInspection.hpp"

#include <stdexcept>

#if IILD_HAS_NATIVE_DIFFUSION
#include <stable-diffusion.h>
#endif

namespace iiLocalDiffusion {
namespace {
void requireFile(const std::filesystem::path &path) {
    if (path.empty() || !std::filesystem::is_regular_file(path))
        throw std::invalid_argument("VAE metadata inspection requires existing model files.");
}
}

NativeVaeInspection inspectNativeVae(const std::filesystem::path &model) {
    requireFile(model);
#if IILD_HAS_NATIVE_DIFFUSION
    sd_model_vae_info_t info{};
    if (!sd_model_inspect_vae(model.string().c_str(), &info))
        throw std::runtime_error("The native engine cannot inspect this model's VAE contract.");
    NativeVaeState state;
    switch (info.state) {
    case SD_VAE_UNSUPPORTED: state = NativeVaeState::Unsupported; break;
    case SD_VAE_MISSING: state = NativeVaeState::Missing; break;
    case SD_VAE_EMBEDDED: state = NativeVaeState::Embedded; break;
    case SD_VAE_INCOMPATIBLE: state = NativeVaeState::Incompatible; break;
    default: throw std::runtime_error("The native engine returned an unknown VAE state.");
    }
    return {state, info.model_family ? info.model_family : "", info.vae_family ? info.vae_family : ""};
#else
    return {NativeVaeState::Unsupported, "", ""};
#endif
}

bool validateNativeVae(const std::filesystem::path &model, const std::filesystem::path &candidate) {
    requireFile(model);
    requireFile(candidate);
#if IILD_HAS_NATIVE_DIFFUSION
    if (inspectNativeVae(model).state == NativeVaeState::Unsupported)
        throw std::runtime_error("The native engine has no VAE contract for this model.");
    return sd_model_validate_vae(model.string().c_str(), candidate.string().c_str());
#else
    throw std::runtime_error("This build does not include the native image engine.");
#endif
}

} // namespace iiLocalDiffusion
