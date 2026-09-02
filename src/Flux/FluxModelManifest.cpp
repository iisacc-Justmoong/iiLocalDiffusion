#include "Flux/FluxModelManifest.hpp"

#include "ModelManifest/ModelManifestParser.hpp"

#include <utility>

namespace
{

namespace fs = std::filesystem;

iild::ModelComponentManifest loadFluxTextEncoder(const fs::path &root)
{
    auto component = iild::detail::loadClipTextEncoder(root, "text_encoder", 768, 768);
    const auto config = iild::detail::parseObject(
        component.configuration,
        "text_encoder/config.json");
    iild::detail::requireSupportedInteger(
        config.get(), "intermediate_size", 3072, "text_encoder configuration");
    iild::detail::requireSupportedInteger(
        config.get(), "num_attention_heads", 12, "text_encoder configuration");
    iild::detail::requireSupportedInteger(
        config.get(), "num_hidden_layers", 12, "text_encoder configuration");
    iild::detail::requireSupportedInteger(
        config.get(), "vocab_size", 49408, "text_encoder configuration");
    return component;
}

iild::ModelComponentManifest loadFluxTokenizer2(const fs::path &root)
{
    const auto directory = root / "tokenizer_2";
    const auto configuration = directory / "tokenizer_config.json";
    const auto sentencePieceModel = directory / "spiece.model";
    const auto fastTokenizer = directory / "tokenizer.json";

    iild::detail::requireDirectory(directory, "tokenizer_2 component");
    const auto config = iild::detail::parseObject(
        configuration,
        "tokenizer_2/tokenizer_config.json");
    iild::detail::requireSupportedInteger(
        config.get(), "model_max_length", 512, "tokenizer_2 configuration");
    iild::detail::requireSupportedInteger(
        config.get(), "extra_ids", 100, "tokenizer_2 configuration");
    iild::detail::requireSupportedString(
        config.get(), "tokenizer_class", "T5Tokenizer", "tokenizer_2 configuration");
    iild::detail::requireFile(sentencePieceModel, "tokenizer_2 SentencePiece model");
    iild::detail::requireFile(fastTokenizer, "tokenizer_2 fast-tokenizer data");

    return {directory, configuration, {sentencePieceModel, fastTokenizer}};
}

iild::ModelComponentManifest loadFluxTextEncoder2(const fs::path &root)
{
    const auto directory = root / "text_encoder_2";
    const auto configuration = directory / "config.json";

    iild::detail::requireDirectory(directory, "text_encoder_2 component");
    const auto config = iild::detail::parseObject(configuration, "text_encoder_2/config.json");
    iild::detail::requireSupportedInteger(
        config.get(), "d_model", 4096, "text_encoder_2 configuration");
    iild::detail::requireSupportedInteger(
        config.get(), "d_ff", 10240, "text_encoder_2 configuration");
    iild::detail::requireSupportedInteger(
        config.get(), "d_kv", 64, "text_encoder_2 configuration");
    iild::detail::requireSupportedInteger(
        config.get(), "num_heads", 64, "text_encoder_2 configuration");
    iild::detail::requireSupportedInteger(
        config.get(), "num_layers", 24, "text_encoder_2 configuration");
    iild::detail::requireSupportedInteger(
        config.get(), "vocab_size", 32128, "text_encoder_2 configuration");
    iild::detail::requireSupportedString(
        config.get(), "feed_forward_proj", "gated-gelu", "text_encoder_2 configuration");

    return {
        directory,
        configuration,
        iild::detail::findSafetensorFiles(directory, "text_encoder_2", "model")};
}

iild::ModelComponentManifest loadFluxTransformer(const fs::path &root)
{
    const auto directory = root / "transformer";
    const auto configuration = directory / "config.json";

    iild::detail::requireDirectory(directory, "transformer component");
    const auto config = iild::detail::parseObject(configuration, "transformer/config.json");
    iild::detail::requireSupportedString(
        config.get(), "_class_name", "FluxTransformer2DModel", "transformer configuration");
    iild::detail::requireSupportedInteger(
        config.get(), "patch_size", 1, "transformer configuration");
    iild::detail::requireSupportedInteger(
        config.get(), "in_channels", 64, "transformer configuration");
    iild::detail::requireOptionalSupportedIntegerOrNull(
        config.get(), "out_channels", 64, "transformer configuration");
    iild::detail::requireSupportedInteger(
        config.get(), "num_layers", 19, "transformer configuration");
    iild::detail::requireSupportedInteger(
        config.get(), "num_single_layers", 38, "transformer configuration");
    iild::detail::requireSupportedInteger(
        config.get(), "attention_head_dim", 128, "transformer configuration");
    iild::detail::requireSupportedInteger(
        config.get(), "num_attention_heads", 24, "transformer configuration");
    iild::detail::requireSupportedInteger(
        config.get(), "joint_attention_dim", 4096, "transformer configuration");
    iild::detail::requireSupportedInteger(
        config.get(), "pooled_projection_dim", 768, "transformer configuration");
    iild::detail::requireSupportedBoolean(
        config.get(), "guidance_embeds", false, "transformer configuration");
    iild::detail::requireOptionalSupportedIntegerArray(
        config.get(), "axes_dims_rope", {16, 56, 56}, "transformer configuration");

    return {
        directory,
        configuration,
        iild::detail::findSafetensorFiles(
            directory,
            "transformer",
            "diffusion_pytorch_model")};
}

iild::ModelComponentManifest loadFluxVae(const fs::path &root)
{
    const auto directory = root / "vae";
    const auto configuration = directory / "config.json";

    iild::detail::requireDirectory(directory, "VAE component");
    const auto config = iild::detail::parseObject(configuration, "vae/config.json");
    iild::detail::requireSupportedString(
        config.get(), "_class_name", "AutoencoderKL", "VAE configuration");
    iild::detail::requireSupportedInteger(config.get(), "in_channels", 3, "VAE configuration");
    iild::detail::requireSupportedInteger(config.get(), "out_channels", 3, "VAE configuration");
    iild::detail::requireSupportedInteger(config.get(), "latent_channels", 16, "VAE configuration");
    iild::detail::requireSupportedInteger(config.get(), "sample_size", 1024, "VAE configuration");
    iild::detail::requireSupportedInteger(config.get(), "layers_per_block", 2, "VAE configuration");
    iild::detail::requireSupportedInteger(config.get(), "norm_num_groups", 32, "VAE configuration");
    iild::detail::requireSupportedIntegerArray(
        config.get(), "block_out_channels", {128, 256, 512, 512}, "VAE configuration");
    iild::detail::requireSupportedStringArray(
        config.get(),
        "down_block_types",
        {"DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"},
        "VAE configuration");
    iild::detail::requireSupportedStringArray(
        config.get(),
        "up_block_types",
        {"UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"},
        "VAE configuration");
    iild::detail::requireSupportedBoolean(
        config.get(), "mid_block_add_attention", true, "VAE configuration");
    iild::detail::requireSupportedNumber(
        config.get(), "scaling_factor", 0.3611, "VAE configuration");
    iild::detail::requireSupportedNumber(
        config.get(), "shift_factor", 0.1159, "VAE configuration");
    iild::detail::requireSupportedBoolean(
        config.get(), "force_upcast", true, "VAE configuration");
    iild::detail::requireSupportedBoolean(
        config.get(), "use_quant_conv", false, "VAE configuration");
    iild::detail::requireSupportedBoolean(
        config.get(), "use_post_quant_conv", false, "VAE configuration");

    return {
        directory,
        configuration,
        iild::detail::findSafetensorFiles(directory, "VAE", "diffusion_pytorch_model")};
}

iild::ModelComponentManifest loadFluxScheduler(const fs::path &root)
{
    const auto directory = root / "scheduler";
    const auto configuration = directory / "scheduler_config.json";

    iild::detail::requireDirectory(directory, "scheduler component");
    const auto config = iild::detail::parseObject(configuration, "scheduler/scheduler_config.json");
    iild::detail::requireSupportedString(
        config.get(),
        "_class_name",
        "FlowMatchEulerDiscreteScheduler",
        "scheduler configuration");
    iild::detail::requireSupportedInteger(
        config.get(), "num_train_timesteps", 1000, "scheduler configuration");
    iild::detail::requireSupportedInteger(
        config.get(), "base_image_seq_len", 256, "scheduler configuration");
    iild::detail::requireSupportedInteger(
        config.get(), "max_image_seq_len", 4096, "scheduler configuration");
    iild::detail::requireSupportedNumber(
        config.get(), "base_shift", 0.5, "scheduler configuration");
    iild::detail::requireSupportedNumber(
        config.get(), "max_shift", 1.15, "scheduler configuration");
    iild::detail::requireSupportedNumber(
        config.get(), "shift", 1.0, "scheduler configuration");
    iild::detail::requireSupportedBoolean(
        config.get(), "use_dynamic_shifting", false, "scheduler configuration");

    return {directory, configuration, {}};
}

} // namespace

namespace iild
{

FluxModelManifest FluxModelManifest::load(const std::filesystem::path &modelRoot)
{
    const auto root = detail::canonicalModelRoot(modelRoot);
    const auto modelIndexPath = root / "model_index.json";
    const auto modelIndex = detail::parseObject(modelIndexPath, "model_index.json");
    const auto pipelineClass = detail::requiredString(
        modelIndex.get(), "_class_name", "model_index.json");
    if (pipelineClass != "FluxPipeline")
    {
        detail::fail(
            ModelManifestErrorCode::unsupportedModel,
            "unsupported model_index.json _class_name: '" +
                detail::displayString(pipelineClass) + "'");
    }

    detail::requirePipelineComponent(
        modelIndex.get(), "tokenizer", "transformers", "CLIPTokenizer");
    detail::requirePipelineComponent(
        modelIndex.get(), "tokenizer_2", "transformers", "T5TokenizerFast");
    detail::requirePipelineComponent(
        modelIndex.get(), "text_encoder", "transformers", "CLIPTextModel");
    detail::requirePipelineComponent(
        modelIndex.get(), "text_encoder_2", "transformers", "T5EncoderModel");
    detail::requirePipelineComponent(
        modelIndex.get(), "transformer", "diffusers", "FluxTransformer2DModel");
    detail::requirePipelineComponent(
        modelIndex.get(), "vae", "diffusers", "AutoencoderKL");
    detail::requirePipelineComponent(
        modelIndex.get(), "scheduler", "diffusers", "FlowMatchEulerDiscreteScheduler");

    auto tokenizer = detail::loadClipTokenizer(root, "tokenizer");
    auto tokenizer2 = loadFluxTokenizer2(root);
    auto textEncoder = loadFluxTextEncoder(root);
    auto textEncoder2 = loadFluxTextEncoder2(root);
    auto transformer = loadFluxTransformer(root);
    auto vae = loadFluxVae(root);
    auto scheduler = loadFluxScheduler(root);

    return FluxModelManifest{
        root,
        modelIndexPath,
        pipelineClass,
        std::move(tokenizer),
        std::move(tokenizer2),
        std::move(textEncoder),
        std::move(textEncoder2),
        std::move(transformer),
        std::move(vae),
        std::move(scheduler)};
}

FluxModelManifest::FluxModelManifest(
    std::filesystem::path root,
    std::filesystem::path modelIndex,
    std::string pipelineClass,
    ModelComponentManifest tokenizer,
    ModelComponentManifest tokenizer2,
    ModelComponentManifest textEncoder,
    ModelComponentManifest textEncoder2,
    ModelComponentManifest transformer,
    ModelComponentManifest vae,
    ModelComponentManifest scheduler)
    : root_{std::move(root)},
      modelIndex_{std::move(modelIndex)},
      pipelineClass_{std::move(pipelineClass)},
      tokenizer_{std::move(tokenizer)},
      tokenizer2_{std::move(tokenizer2)},
      textEncoder_{std::move(textEncoder)},
      textEncoder2_{std::move(textEncoder2)},
      transformer_{std::move(transformer)},
      vae_{std::move(vae)},
      scheduler_{std::move(scheduler)}
{
}

FluxCompatibility FluxModelManifest::compatibilityKind() const noexcept
{
    return FluxCompatibility::v1Schnell;
}

std::string_view FluxModelManifest::compatibility() const noexcept
{
    return "Flux1Schnell";
}

const std::filesystem::path &FluxModelManifest::root() const noexcept
{
    return root_;
}

const std::filesystem::path &FluxModelManifest::modelIndex() const noexcept
{
    return modelIndex_;
}

const std::string &FluxModelManifest::pipelineClass() const noexcept
{
    return pipelineClass_;
}

const ModelComponentManifest &FluxModelManifest::tokenizer() const noexcept
{
    return tokenizer_;
}

const ModelComponentManifest &FluxModelManifest::tokenizer2() const noexcept
{
    return tokenizer2_;
}

const ModelComponentManifest &FluxModelManifest::textEncoder() const noexcept
{
    return textEncoder_;
}

const ModelComponentManifest &FluxModelManifest::textEncoder2() const noexcept
{
    return textEncoder2_;
}

const ModelComponentManifest &FluxModelManifest::transformer() const noexcept
{
    return transformer_;
}

const ModelComponentManifest &FluxModelManifest::vae() const noexcept
{
    return vae_;
}

const ModelComponentManifest &FluxModelManifest::scheduler() const noexcept
{
    return scheduler_;
}

} // namespace iild
