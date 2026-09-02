#pragma once

#include "ModelManifest/ModelManifest.hpp"

#include <filesystem>
#include <string>
#include <string_view>

namespace iild
{

enum class FluxCompatibility
{
    v1Schnell
};

class IILD_EXPORT FluxModelManifest final
{
public:
    [[nodiscard]] static FluxModelManifest load(const std::filesystem::path &modelRoot);

    [[nodiscard]] FluxCompatibility compatibilityKind() const noexcept;
    [[nodiscard]] std::string_view compatibility() const noexcept;

    [[nodiscard]] const std::filesystem::path &root() const noexcept;
    [[nodiscard]] const std::filesystem::path &modelIndex() const noexcept;
    [[nodiscard]] const std::string &pipelineClass() const noexcept;
    [[nodiscard]] const ModelComponentManifest &tokenizer() const noexcept;
    [[nodiscard]] const ModelComponentManifest &tokenizer2() const noexcept;
    [[nodiscard]] const ModelComponentManifest &textEncoder() const noexcept;
    [[nodiscard]] const ModelComponentManifest &textEncoder2() const noexcept;
    [[nodiscard]] const ModelComponentManifest &transformer() const noexcept;
    [[nodiscard]] const ModelComponentManifest &vae() const noexcept;
    [[nodiscard]] const ModelComponentManifest &scheduler() const noexcept;

private:
    FluxModelManifest(
        std::filesystem::path root,
        std::filesystem::path modelIndex,
        std::string pipelineClass,
        ModelComponentManifest tokenizer,
        ModelComponentManifest tokenizer2,
        ModelComponentManifest textEncoder,
        ModelComponentManifest textEncoder2,
        ModelComponentManifest transformer,
        ModelComponentManifest vae,
        ModelComponentManifest scheduler);

    std::filesystem::path root_;
    std::filesystem::path modelIndex_;
    std::string pipelineClass_;
    ModelComponentManifest tokenizer_;
    ModelComponentManifest tokenizer2_;
    ModelComponentManifest textEncoder_;
    ModelComponentManifest textEncoder2_;
    ModelComponentManifest transformer_;
    ModelComponentManifest vae_;
    ModelComponentManifest scheduler_;
};

} // namespace iild
