#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#    if defined(iiLocalDiffusion_EXPORTS)
#        define IILD_EXPORT __declspec(dllexport)
#    else
#        define IILD_EXPORT __declspec(dllimport)
#    endif
#else
#    define IILD_EXPORT __attribute__((visibility("default")))
#endif

namespace iild
{

enum class ModelManifestErrorCode
{
    inputUnavailable,
    invalidPackage,
    unsupportedModel
};

class IILD_EXPORT ModelManifestError final : public std::runtime_error
{
public:
    ModelManifestError(ModelManifestErrorCode code, std::string message);

    [[nodiscard]] ModelManifestErrorCode code() const noexcept;

private:
    ModelManifestErrorCode code_;
};

struct IILD_EXPORT ModelComponentManifest
{
    std::filesystem::path directory;
    std::filesystem::path configuration;
    std::vector<std::filesystem::path> artifacts;
};

} // namespace iild
