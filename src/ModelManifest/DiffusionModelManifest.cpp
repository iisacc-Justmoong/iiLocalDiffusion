#include "ModelManifest/DiffusionModelManifest.hpp"

#include "ModelManifest/ModelManifestParser.hpp"

namespace iild
{

DiffusionModelManifest loadModelManifest(const std::filesystem::path &modelRoot)
{
    const auto root = detail::canonicalModelRoot(modelRoot);
    const auto modelIndex = detail::parseObject(root / "model_index.json", "model_index.json");
    const auto pipelineClass = detail::requiredString(
        modelIndex.get(), "_class_name", "model_index.json");

    if (pipelineClass == "StableDiffusionPipeline" ||
        pipelineClass == "StableDiffusionXLPipeline")
    {
        return StableDiffusionModelManifest::load(root);
    }
    if (pipelineClass == "FluxPipeline")
    {
        return FluxModelManifest::load(root);
    }

    detail::fail(
        ModelManifestErrorCode::unsupportedModel,
        "unsupported model_index.json _class_name: '" +
            detail::displayString(pipelineClass) + "'");
}

} // namespace iild
