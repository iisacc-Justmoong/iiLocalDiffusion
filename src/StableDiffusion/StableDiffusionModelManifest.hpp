#pragma once

#include "ModelManifest/ModelManifest.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace iild
{

enum class StableDiffusionCompatibility
{
    v1,
    xlBase
};

class IILD_EXPORT StableDiffusionModelManifest final
{
public:
    [[nodiscard]] static StableDiffusionModelManifest load(
        const std::filesystem::path &modelRoot);

    [[nodiscard]] StableDiffusionCompatibility compatibilityKind() const noexcept;
    [[nodiscard]] std::string_view compatibility() const noexcept;

    [[nodiscard]] const std::filesystem::path &root() const noexcept;
    [[nodiscard]] const std::filesystem::path &modelIndex() const noexcept;
    [[nodiscard]] const std::string &pipelineClass() const noexcept;
    [[nodiscard]] const ModelComponentManifest &tokenizer() const noexcept;
    [[nodiscard]] const std::optional<ModelComponentManifest> &tokenizer2() const noexcept;
    [[nodiscard]] const ModelComponentManifest &textEncoder() const noexcept;
    [[nodiscard]] const std::optional<ModelComponentManifest> &textEncoder2() const noexcept;
    [[nodiscard]] const ModelComponentManifest &unet() const noexcept;
    [[nodiscard]] const ModelComponentManifest &vae() const noexcept;
    [[nodiscard]] const ModelComponentManifest &scheduler() const noexcept;

private:
    StableDiffusionModelManifest(
        std::filesystem::path root,
        std::filesystem::path modelIndex,
        std::string pipelineClass,
        StableDiffusionCompatibility compatibility,
        ModelComponentManifest tokenizer,
        std::optional<ModelComponentManifest> tokenizer2,
        ModelComponentManifest textEncoder,
        std::optional<ModelComponentManifest> textEncoder2,
        ModelComponentManifest unet,
        ModelComponentManifest vae,
        ModelComponentManifest scheduler);

    std::filesystem::path root_;
    std::filesystem::path modelIndex_;
    std::string pipelineClass_;
    StableDiffusionCompatibility compatibility_;
    ModelComponentManifest tokenizer_;
    std::optional<ModelComponentManifest> tokenizer2_;
    ModelComponentManifest textEncoder_;
    std::optional<ModelComponentManifest> textEncoder2_;
    ModelComponentManifest unet_;
    ModelComponentManifest vae_;
    ModelComponentManifest scheduler_;
};

} // namespace iild
