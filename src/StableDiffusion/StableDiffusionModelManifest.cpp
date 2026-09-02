#include "StableDiffusion/StableDiffusionModelManifest.hpp"

#include "ModelManifest/ModelManifestParser.hpp"

#include <cstdint>
#include <optional>
#include <utility>

namespace
{

namespace fs = std::filesystem;

iild::ModelComponentManifest loadUnet(
    const fs::path &root,
    const iild::StableDiffusionCompatibility compatibility)
{
    const auto directory = root / "unet";
    const auto configuration = directory / "config.json";

    iild::detail::requireDirectory(directory, "UNet component");
    const auto config = iild::detail::parseObject(configuration, "unet/config.json");
    iild::detail::requireSupportedString(
        config.get(), "_class_name", "UNet2DConditionModel", "UNet configuration");
    iild::detail::requireSupportedInteger(config.get(), "in_channels", 4, "UNet configuration");
    iild::detail::requireSupportedInteger(config.get(), "out_channels", 4, "UNet configuration");
    if (compatibility == iild::StableDiffusionCompatibility::v1)
    {
        iild::detail::requireSupportedInteger(config.get(), "sample_size", 64, "UNet configuration");
        iild::detail::requireSupportedInteger(
            config.get(), "cross_attention_dim", 768, "UNet configuration");
    }
    else
    {
        iild::detail::requireSupportedInteger(config.get(), "sample_size", 128, "UNet configuration");
        iild::detail::requireSupportedInteger(
            config.get(), "cross_attention_dim", 2048, "UNet configuration");
        iild::detail::requireSupportedString(
            config.get(), "addition_embed_type", "text_time", "UNet configuration");
        iild::detail::requireSupportedInteger(
            config.get(), "addition_time_embed_dim", 256, "UNet configuration");
        iild::detail::requireSupportedInteger(
            config.get(),
            "projection_class_embeddings_input_dim",
            2816,
            "UNet configuration");
        iild::detail::requireSupportedBoolean(
            config.get(), "use_linear_projection", true, "UNet configuration");
    }

    return {
        directory,
        configuration,
        iild::detail::findSafetensorFiles(directory, "UNet", "diffusion_pytorch_model")};
}

iild::ModelComponentManifest loadVae(
    const fs::path &root,
    const iild::StableDiffusionCompatibility compatibility)
{
    const auto directory = root / "vae";
    const auto configuration = directory / "config.json";

    iild::detail::requireDirectory(directory, "VAE component");
    const auto config = iild::detail::parseObject(configuration, "vae/config.json");
    iild::detail::requireSupportedString(
        config.get(), "_class_name", "AutoencoderKL", "VAE configuration");
    iild::detail::requireSupportedInteger(config.get(), "in_channels", 3, "VAE configuration");
    iild::detail::requireSupportedInteger(config.get(), "out_channels", 3, "VAE configuration");
    iild::detail::requireSupportedInteger(config.get(), "latent_channels", 4, "VAE configuration");
    if (compatibility == iild::StableDiffusionCompatibility::v1)
    {
        iild::detail::requireSupportedInteger(config.get(), "sample_size", 512, "VAE configuration");
    }
    else
    {
        iild::detail::requireSupportedInteger(config.get(), "sample_size", 1024, "VAE configuration");
        iild::detail::requireSupportedNumber(
            config.get(), "scaling_factor", 0.13025, "VAE configuration");
        iild::detail::requireSupportedBoolean(
            config.get(), "force_upcast", true, "VAE configuration");
    }

    return {
        directory,
        configuration,
        iild::detail::findSafetensorFiles(directory, "VAE", "diffusion_pytorch_model")};
}

iild::ModelComponentManifest loadScheduler(
    const fs::path &root,
    const iild::StableDiffusionCompatibility compatibility)
{
    const auto directory = root / "scheduler";
    const auto configuration = directory / "scheduler_config.json";

    iild::detail::requireDirectory(directory, "scheduler component");
    const auto config = iild::detail::parseObject(configuration, "scheduler/scheduler_config.json");
    iild::detail::requireSupportedInteger(
        config.get(), "num_train_timesteps", 1000, "scheduler configuration");
    iild::detail::requireSupportedString(
        config.get(), "beta_schedule", "scaled_linear", "scheduler configuration");
    if (compatibility == iild::StableDiffusionCompatibility::v1)
    {
        iild::detail::requireSupportedString(
            config.get(), "_class_name", "PNDMScheduler", "scheduler configuration");
        iild::detail::requireSupportedBoolean(
            config.get(), "skip_prk_steps", true, "scheduler configuration");
    }
    else
    {
        iild::detail::requireSupportedString(
            config.get(), "_class_name", "EulerDiscreteScheduler", "scheduler configuration");
        iild::detail::requireSupportedNumber(
            config.get(), "beta_start", 0.00085, "scheduler configuration");
        iild::detail::requireSupportedNumber(
            config.get(), "beta_end", 0.012, "scheduler configuration");
        iild::detail::requireSupportedString(
            config.get(), "prediction_type", "epsilon", "scheduler configuration");
        iild::detail::requireSupportedString(
            config.get(), "timestep_spacing", "leading", "scheduler configuration");
    }

    return {directory, configuration, {}};
}

} // namespace

namespace iild
{

StableDiffusionModelManifest StableDiffusionModelManifest::load(
    const std::filesystem::path &modelRoot)
{
    const auto root = detail::canonicalModelRoot(modelRoot);

    const auto modelIndexPath = root / "model_index.json";
    const auto modelIndex = detail::parseObject(modelIndexPath, "model_index.json");
    const auto pipelineClass = detail::requiredString(
        modelIndex.get(), "_class_name", "model_index.json");
    StableDiffusionCompatibility compatibility;
    if (pipelineClass == "StableDiffusionPipeline")
    {
        compatibility = StableDiffusionCompatibility::v1;
    }
    else if (pipelineClass == "StableDiffusionXLPipeline")
    {
        compatibility = StableDiffusionCompatibility::xlBase;
    }
    else
    {
        detail::fail(
            ModelManifestErrorCode::unsupportedModel,
            "unsupported model_index.json _class_name: '" +
                detail::displayString(pipelineClass) + "'");
    }

    detail::requirePipelineComponent(
        modelIndex.get(), "tokenizer", "transformers", "CLIPTokenizer");
    detail::requirePipelineComponent(
        modelIndex.get(), "text_encoder", "transformers", "CLIPTextModel");
    detail::requirePipelineComponent(
        modelIndex.get(), "unet", "diffusers", "UNet2DConditionModel");
    detail::requirePipelineComponent(
        modelIndex.get(), "vae", "diffusers", "AutoencoderKL");
    if (compatibility == StableDiffusionCompatibility::v1)
    {
        detail::requirePipelineComponent(
            modelIndex.get(), "scheduler", "diffusers", "PNDMScheduler");
    }
    else
    {
        detail::requireSupportedBoolean(
            modelIndex.get(), "force_zeros_for_empty_prompt", true, "model_index.json");
        detail::requirePipelineComponent(
            modelIndex.get(), "tokenizer_2", "transformers", "CLIPTokenizer");
        detail::requirePipelineComponent(
            modelIndex.get(),
            "text_encoder_2",
            "transformers",
            "CLIPTextModelWithProjection");
        detail::requirePipelineComponent(
            modelIndex.get(), "scheduler", "diffusers", "EulerDiscreteScheduler");
    }

    auto tokenizer = detail::loadClipTokenizer(root, "tokenizer");
    std::optional<ModelComponentManifest> tokenizer2;
    auto textEncoder = detail::loadClipTextEncoder(
        root,
        "text_encoder",
        768,
        compatibility == StableDiffusionCompatibility::xlBase
            ? std::optional<std::int64_t>{768}
            : std::nullopt);
    std::optional<ModelComponentManifest> textEncoder2;
    if (compatibility == StableDiffusionCompatibility::xlBase)
    {
        tokenizer2 = detail::loadClipTokenizer(root, "tokenizer_2");
        textEncoder2 = detail::loadClipTextEncoder(root, "text_encoder_2", 1280, 1280);
    }
    auto unet = loadUnet(root, compatibility);
    auto vae = loadVae(root, compatibility);
    auto scheduler = loadScheduler(root, compatibility);

    return StableDiffusionModelManifest{
        root,
        modelIndexPath,
        pipelineClass,
        compatibility,
        std::move(tokenizer),
        std::move(tokenizer2),
        std::move(textEncoder),
        std::move(textEncoder2),
        std::move(unet),
        std::move(vae),
        std::move(scheduler)};
}

StableDiffusionModelManifest::StableDiffusionModelManifest(
    std::filesystem::path root,
    std::filesystem::path modelIndex,
    std::string pipelineClass,
    const StableDiffusionCompatibility compatibility,
    ModelComponentManifest tokenizer,
    std::optional<ModelComponentManifest> tokenizer2,
    ModelComponentManifest textEncoder,
    std::optional<ModelComponentManifest> textEncoder2,
    ModelComponentManifest unet,
    ModelComponentManifest vae,
    ModelComponentManifest scheduler)
    : root_{std::move(root)},
      modelIndex_{std::move(modelIndex)},
      pipelineClass_{std::move(pipelineClass)},
      compatibility_{compatibility},
      tokenizer_{std::move(tokenizer)},
      tokenizer2_{std::move(tokenizer2)},
      textEncoder_{std::move(textEncoder)},
      textEncoder2_{std::move(textEncoder2)},
      unet_{std::move(unet)},
      vae_{std::move(vae)},
      scheduler_{std::move(scheduler)}
{
}

const std::filesystem::path &StableDiffusionModelManifest::root() const noexcept
{
    return root_;
}

const std::filesystem::path &StableDiffusionModelManifest::modelIndex() const noexcept
{
    return modelIndex_;
}

const std::string &StableDiffusionModelManifest::pipelineClass() const noexcept
{
    return pipelineClass_;
}

StableDiffusionCompatibility StableDiffusionModelManifest::compatibilityKind() const noexcept
{
    return compatibility_;
}

std::string_view StableDiffusionModelManifest::compatibility() const noexcept
{
    switch (compatibility_)
    {
    case StableDiffusionCompatibility::v1:
        return "StableDiffusionV1";
    case StableDiffusionCompatibility::xlBase:
        return "StableDiffusionXLBase";
    }
    return {};
}

const ModelComponentManifest &StableDiffusionModelManifest::tokenizer() const noexcept
{
    return tokenizer_;
}

const std::optional<ModelComponentManifest> &StableDiffusionModelManifest::tokenizer2() const noexcept
{
    return tokenizer2_;
}

const ModelComponentManifest &StableDiffusionModelManifest::textEncoder() const noexcept
{
    return textEncoder_;
}

const std::optional<ModelComponentManifest> &StableDiffusionModelManifest::textEncoder2() const noexcept
{
    return textEncoder2_;
}

const ModelComponentManifest &StableDiffusionModelManifest::unet() const noexcept
{
    return unet_;
}

const ModelComponentManifest &StableDiffusionModelManifest::vae() const noexcept
{
    return vae_;
}

const ModelComponentManifest &StableDiffusionModelManifest::scheduler() const noexcept
{
    return scheduler_;
}

} // namespace iild
