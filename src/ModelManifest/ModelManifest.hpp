#pragma once

#include "Export.hpp"

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

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
