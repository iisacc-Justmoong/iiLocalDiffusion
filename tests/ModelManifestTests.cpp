#include "StableDiffusion/StableDiffusionModelManifest.hpp"

#include "Flux/FluxModelManifest.hpp"
#include "ModelManifest/DiffusionModelManifest.hpp"

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace
{

namespace fs = std::filesystem;

class TestFailure final : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

void require(const bool condition, const std::string &message)
{
    if (!condition)
    {
        throw TestFailure{message};
    }
}

void writeFile(const fs::path &path, const std::string &contents)
{
    fs::create_directories(path.parent_path());
    std::ofstream output{path, std::ios::binary};
    if (!output)
    {
        throw TestFailure{"could not create test file: " + path.string()};
    }
    output << contents;
}

const std::string fluxTransformerConfiguration =
    R"json({"_class_name": "FluxTransformer2DModel", "attention_head_dim": 128, "axes_dims_rope": [16, 56, 56], "guidance_embeds": false, "in_channels": 64, "joint_attention_dim": 4096, "num_attention_heads": 24, "num_layers": 19, "num_single_layers": 38, "out_channels": null, "patch_size": 1, "pooled_projection_dim": 768})json";

const std::string fluxClipTextEncoderConfiguration =
    R"json({"architectures": ["CLIPTextModel"], "hidden_size": 768, "intermediate_size": 3072, "max_position_embeddings": 77, "num_attention_heads": 12, "num_hidden_layers": 12, "projection_dim": 768, "vocab_size": 49408})json";

const std::string fluxT5TextEncoderConfiguration =
    R"json({"architectures": ["T5EncoderModel"], "d_ff": 10240, "d_kv": 64, "d_model": 4096, "feed_forward_proj": "gated-gelu", "num_heads": 64, "num_layers": 24, "vocab_size": 32128})json";

const std::string fluxVaeConfiguration =
    R"json({"_class_name": "AutoencoderKL", "block_out_channels": [128, 256, 512, 512], "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"], "force_upcast": true, "in_channels": 3, "latent_channels": 16, "layers_per_block": 2, "mid_block_add_attention": true, "norm_num_groups": 32, "out_channels": 3, "sample_size": 1024, "scaling_factor": 0.3611, "shift_factor": 0.1159, "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"], "use_post_quant_conv": false, "use_quant_conv": false})json";

std::string replaceOnce(
    std::string contents,
    const std::string_view expected,
    const std::string_view replacement)
{
    const auto position = contents.find(expected);
    if (position == std::string::npos)
    {
        throw TestFailure{"test fixture replacement target is missing"};
    }
    contents.replace(position, expected.size(), replacement);
    return contents;
}

void createValidFixture(const fs::path &root)
{
    writeFile(root / "model_index.json", R"json({
        "_class_name": "StableDiffusionPipeline",
        "tokenizer": ["transformers", "CLIPTokenizer"],
        "text_encoder": ["transformers", "CLIPTextModel"],
        "unet": ["diffusers", "UNet2DConditionModel"],
        "vae": ["diffusers", "AutoencoderKL"],
        "scheduler": ["diffusers", "PNDMScheduler"]
    })json");
    writeFile(root / "tokenizer" / "tokenizer_config.json",
              R"json({"model_max_length": 77, "tokenizer_class": "CLIPTokenizer"})json");
    writeFile(root / "tokenizer" / "vocab.json", R"json({"test": 0})json");
    writeFile(root / "tokenizer" / "merges.txt", "#version: 0.2\nt e\n");
    writeFile(root / "text_encoder" / "config.json",
              R"json({"architectures": ["CLIPTextModel"], "hidden_size": 768, "max_position_embeddings": 77})json");
    writeFile(root / "text_encoder" / "model.safetensors", "test weights");
    writeFile(root / "unet" / "config.json",
              R"json({"_class_name": "UNet2DConditionModel", "in_channels": 4, "out_channels": 4, "sample_size": 64, "cross_attention_dim": 768})json");
    writeFile(root / "unet" / "diffusion_pytorch_model.safetensors", "test weights");
    writeFile(root / "vae" / "config.json",
              R"json({"_class_name": "AutoencoderKL", "in_channels": 3, "out_channels": 3, "latent_channels": 4, "sample_size": 512})json");
    writeFile(root / "vae" / "diffusion_pytorch_model.safetensors", "test weights");
    writeFile(root / "scheduler" / "scheduler_config.json",
              R"json({"_class_name": "PNDMScheduler", "num_train_timesteps": 1000, "beta_schedule": "scaled_linear", "skip_prk_steps": true})json");
}

void createValidSdxlFixture(const fs::path &root)
{
    writeFile(root / "model_index.json", R"json({
        "_class_name": "StableDiffusionXLPipeline",
        "force_zeros_for_empty_prompt": true,
        "tokenizer": ["transformers", "CLIPTokenizer"],
        "tokenizer_2": ["transformers", "CLIPTokenizer"],
        "text_encoder": ["transformers", "CLIPTextModel"],
        "text_encoder_2": ["transformers", "CLIPTextModelWithProjection"],
        "unet": ["diffusers", "UNet2DConditionModel"],
        "vae": ["diffusers", "AutoencoderKL"],
        "scheduler": ["diffusers", "EulerDiscreteScheduler"]
    })json");
    for (const auto *name : {"tokenizer", "tokenizer_2"})
    {
        const auto directory = root / name;
        writeFile(directory / "tokenizer_config.json",
                  R"json({"model_max_length": 77, "tokenizer_class": "CLIPTokenizer"})json");
        writeFile(directory / "vocab.json", R"json({"test": 0})json");
        writeFile(directory / "merges.txt", "#version: 0.2\nt e\n");
    }
    writeFile(root / "text_encoder" / "config.json",
              R"json({"architectures": ["CLIPTextModel"], "hidden_size": 768, "max_position_embeddings": 77, "projection_dim": 768})json");
    writeFile(root / "text_encoder" / "model.safetensors", "test weights");
    writeFile(root / "text_encoder_2" / "config.json",
              R"json({"architectures": ["CLIPTextModelWithProjection"], "hidden_size": 1280, "max_position_embeddings": 77, "projection_dim": 1280})json");
    writeFile(root / "text_encoder_2" / "model.safetensors", "test weights");
    writeFile(root / "unet" / "config.json",
              R"json({"_class_name": "UNet2DConditionModel", "addition_embed_type": "text_time", "addition_time_embed_dim": 256, "in_channels": 4, "out_channels": 4, "sample_size": 128, "cross_attention_dim": 2048, "projection_class_embeddings_input_dim": 2816, "use_linear_projection": true})json");
    writeFile(root / "unet" / "diffusion_pytorch_model.safetensors", "test weights");
    writeFile(root / "vae" / "config.json",
              R"json({"_class_name": "AutoencoderKL", "force_upcast": true, "in_channels": 3, "out_channels": 3, "latent_channels": 4, "sample_size": 1024, "scaling_factor": 0.13025})json");
    writeFile(root / "vae" / "diffusion_pytorch_model.safetensors", "test weights");
    writeFile(root / "scheduler" / "scheduler_config.json",
              R"json({"_class_name": "EulerDiscreteScheduler", "num_train_timesteps": 1000, "beta_start": 0.00085, "beta_end": 0.012, "beta_schedule": "scaled_linear", "prediction_type": "epsilon", "timestep_spacing": "leading"})json");
}

void createValidFluxFixture(const fs::path &root)
{
    writeFile(root / "model_index.json", R"json({
        "_class_name": "FluxPipeline",
        "tokenizer": ["transformers", "CLIPTokenizer"],
        "tokenizer_2": ["transformers", "T5TokenizerFast"],
        "text_encoder": ["transformers", "CLIPTextModel"],
        "text_encoder_2": ["transformers", "T5EncoderModel"],
        "transformer": ["diffusers", "FluxTransformer2DModel"],
        "vae": ["diffusers", "AutoencoderKL"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"]
    })json");
    writeFile(root / "tokenizer" / "tokenizer_config.json",
              R"json({"model_max_length": 77, "tokenizer_class": "CLIPTokenizer"})json");
    writeFile(root / "tokenizer" / "vocab.json", R"json({"test": 0})json");
    writeFile(root / "tokenizer" / "merges.txt", "#version: 0.2\nt e\n");
    writeFile(root / "tokenizer_2" / "tokenizer_config.json",
              R"json({"extra_ids": 100, "model_max_length": 512, "tokenizer_class": "T5Tokenizer"})json");
    writeFile(root / "tokenizer_2" / "spiece.model", "test model");
    writeFile(root / "tokenizer_2" / "tokenizer.json", R"json({"version": "1.0"})json");
    writeFile(root / "text_encoder" / "config.json", fluxClipTextEncoderConfiguration);
    writeFile(root / "text_encoder" / "model.safetensors", "test weights");
    writeFile(root / "text_encoder_2" / "config.json", fluxT5TextEncoderConfiguration);
    writeFile(root / "text_encoder_2" / "model-00001-of-00002.safetensors", "test weights 1");
    writeFile(root / "text_encoder_2" / "model-00002-of-00002.safetensors", "test weights 2");
    writeFile(root / "text_encoder_2" / "model.safetensors.index.json",
              R"json({"metadata": {"total_size": 2}, "weight_map": {"encoder.block.0": "model-00001-of-00002.safetensors", "encoder.block.23": "model-00002-of-00002.safetensors"}})json");
    writeFile(root / "transformer" / "config.json", fluxTransformerConfiguration);
    writeFile(root / "transformer" / "diffusion_pytorch_model-00001-of-00003.safetensors", "test weights 1");
    writeFile(root / "transformer" / "diffusion_pytorch_model-00002-of-00003.safetensors", "test weights 2");
    writeFile(root / "transformer" / "diffusion_pytorch_model-00003-of-00003.safetensors", "test weights 3");
    writeFile(root / "transformer" / "diffusion_pytorch_model.safetensors.index.json",
              R"json({"metadata": {"total_size": 3}, "weight_map": {"context_embedder.bias": "diffusion_pytorch_model-00001-of-00003.safetensors", "transformer_blocks.0": "diffusion_pytorch_model-00002-of-00003.safetensors", "single_transformer_blocks.37": "diffusion_pytorch_model-00003-of-00003.safetensors"}})json");
    writeFile(root / "vae" / "config.json", fluxVaeConfiguration);
    writeFile(root / "vae" / "diffusion_pytorch_model.safetensors", "test weights");
    writeFile(root / "scheduler" / "scheduler_config.json",
              R"json({"_class_name": "FlowMatchEulerDiscreteScheduler", "base_image_seq_len": 256, "base_shift": 0.5, "max_image_seq_len": 4096, "max_shift": 1.15, "num_train_timesteps": 1000, "shift": 1.0, "use_dynamic_shifting": false})json");
}

void requireManifestError(
    const std::function<void()> &operation,
    const iild::ModelManifestErrorCode expectedCode,
    const std::string &expectedMessage)
{
    try
    {
        operation();
    }
    catch (const iild::ModelManifestError &error)
    {
        require(error.code() == expectedCode, "unexpected manifest error category");
        require(
            std::string{error.what()}.find(expectedMessage) != std::string::npos,
            "unexpected error message: " + std::string{error.what()});
        return;
    }

    throw TestFailure{"expected ModelManifestError"};
}

void loadsStableDiffusionV1Package(const fs::path &workDirectory)
{
    const auto root = workDirectory / "valid";
    createValidFixture(root);

    const auto manifest = iild::StableDiffusionModelManifest::load(root);

    require(manifest.compatibility() == "StableDiffusionV1", "compatibility mismatch");
    require(
        manifest.compatibilityKind() == iild::StableDiffusionCompatibility::v1,
        "compatibility kind mismatch");
    require(manifest.pipelineClass() == "StableDiffusionPipeline", "pipeline class mismatch");
    require(manifest.root() == fs::canonical(root), "root should be canonical");
    require(manifest.tokenizer().artifacts.size() == 2, "tokenizer artifacts mismatch");
    require(manifest.textEncoder().artifacts.size() == 1, "text encoder weight file missing");
    require(manifest.unet().artifacts.size() == 1, "UNet weight file missing");
    require(manifest.vae().artifacts.size() == 1, "VAE weight file missing");
    require(manifest.scheduler().artifacts.empty(), "scheduler should not have weight files");
    require(!manifest.tokenizer2().has_value(), "SD v1 must not expose a second tokenizer");
    require(!manifest.textEncoder2().has_value(), "SD v1 must not expose a second text encoder");
}

void loadsStableDiffusionXlBasePackage(const fs::path &workDirectory)
{
    const auto root = workDirectory / "valid-sdxl";
    createValidSdxlFixture(root);

    const auto manifest = iild::StableDiffusionModelManifest::load(root);

    require(manifest.compatibility() == "StableDiffusionXLBase", "SDXL compatibility mismatch");
    require(
        manifest.compatibilityKind() == iild::StableDiffusionCompatibility::xlBase,
        "SDXL compatibility kind mismatch");
    require(manifest.pipelineClass() == "StableDiffusionXLPipeline", "SDXL pipeline class mismatch");
    require(manifest.tokenizer2().has_value(), "SDXL second tokenizer missing");
    require(manifest.tokenizer2()->artifacts.size() == 2, "SDXL tokenizer 2 artifacts mismatch");
    require(manifest.textEncoder2().has_value(), "SDXL second text encoder missing");
    require(manifest.textEncoder2()->artifacts.size() == 1, "SDXL text encoder 2 weights missing");
}

void loadsFlux1SchnellPackage(const fs::path &workDirectory)
{
    const auto root = workDirectory / "valid-flux";
    createValidFluxFixture(root);

    const auto manifest = iild::FluxModelManifest::load(root);
    require(manifest.compatibility() == "Flux1Schnell", "FLUX compatibility mismatch");
    require(
        manifest.compatibilityKind() == iild::FluxCompatibility::v1Schnell,
        "FLUX compatibility kind mismatch");
    require(manifest.pipelineClass() == "FluxPipeline", "FLUX pipeline class mismatch");
    require(manifest.root() == fs::canonical(root), "FLUX root should be canonical");
    require(manifest.tokenizer().artifacts.size() == 2, "FLUX tokenizer artifacts mismatch");
    require(manifest.tokenizer2().artifacts.size() == 2, "FLUX tokenizer 2 artifacts mismatch");
    require(manifest.textEncoder().artifacts.size() == 1, "FLUX CLIP weights mismatch");
    require(manifest.textEncoder2().artifacts.size() == 2, "FLUX T5 shards mismatch");
    require(manifest.transformer().artifacts.size() == 3, "FLUX transformer shards mismatch");
    require(manifest.vae().artifacts.size() == 1, "FLUX VAE weights mismatch");

    const auto genericManifest = iild::loadModelManifest(root);
    require(
        std::holds_alternative<iild::FluxModelManifest>(genericManifest),
        "generic loader did not dispatch to FLUX");
}

void acceptsFluxOutChannelDefaultForms(const fs::path &workDirectory)
{
    const auto absentRoot = workDirectory / "flux-out-channels-absent";
    createValidFluxFixture(absentRoot);
    writeFile(
        absentRoot / "transformer" / "config.json",
        replaceOnce(fluxTransformerConfiguration, R"json("out_channels": null, )json", ""));
    static_cast<void>(iild::FluxModelManifest::load(absentRoot));

    const auto explicitRoot = workDirectory / "flux-out-channels-explicit";
    createValidFluxFixture(explicitRoot);
    writeFile(
        explicitRoot / "transformer" / "config.json",
        replaceOnce(
            fluxTransformerConfiguration,
            R"json("out_channels": null)json",
            R"json("out_channels": 64)json"));
    static_cast<void>(iild::FluxModelManifest::load(explicitRoot));
}

void rejectsMismatchedFluxOutChannels(const fs::path &workDirectory)
{
    const auto root = workDirectory / "flux-wrong-out-channels";
    createValidFluxFixture(root);
    writeFile(
        root / "transformer" / "config.json",
        replaceOnce(
            fluxTransformerConfiguration,
            R"json("out_channels": null)json",
            R"json("out_channels": 32)json"));

    requireManifestError(
        [&] { static_cast<void>(iild::FluxModelManifest::load(root)); },
        iild::ModelManifestErrorCode::unsupportedModel,
        "out_channels");
}

void rejectsMismatchedFluxAxes(const fs::path &workDirectory)
{
    const auto root = workDirectory / "flux-wrong-axes";
    createValidFluxFixture(root);
    writeFile(
        root / "transformer" / "config.json",
        replaceOnce(fluxTransformerConfiguration, "[16, 56, 56]", "[16, 32, 80]"));

    requireManifestError(
        [&] { static_cast<void>(iild::FluxModelManifest::load(root)); },
        iild::ModelManifestErrorCode::unsupportedModel,
        "axes_dims_rope");
}

void rejectsMissingFluxShardIndex(const fs::path &workDirectory)
{
    const auto root = workDirectory / "flux-missing-shard-index";
    createValidFluxFixture(root);
    fs::remove(root / "text_encoder_2" / "model.safetensors.index.json");

    requireManifestError(
        [&] { static_cast<void>(iild::FluxModelManifest::load(root)); },
        iild::ModelManifestErrorCode::invalidPackage,
        "sharded safetensors index");
}

void rejectsMissingFluxShard(const fs::path &workDirectory)
{
    const auto root = workDirectory / "flux-missing-shard";
    createValidFluxFixture(root);
    fs::remove(
        root / "transformer" / "diffusion_pytorch_model-00002-of-00003.safetensors");

    requireManifestError(
        [&] { static_cast<void>(iild::FluxModelManifest::load(root)); },
        iild::ModelManifestErrorCode::invalidPackage,
        "referenced shard");
}

void rejectsMismatchedFluxShardIndex(const fs::path &workDirectory)
{
    const auto root = workDirectory / "flux-mismatched-shard-index";
    createValidFluxFixture(root);
    writeFile(root / "transformer" / "diffusion_pytorch_model.safetensors.index.json",
              R"json({"weight_map": {"first": "diffusion_pytorch_model-00001-of-00003.safetensors", "last": "diffusion_pytorch_model-00003-of-00003.safetensors"}})json");

    requireManifestError(
        [&] { static_cast<void>(iild::FluxModelManifest::load(root)); },
        iild::ModelManifestErrorCode::invalidPackage,
        "every shard");
}

void rejectsMismatchedFluxVaeProfile(const fs::path &workDirectory)
{
    const auto rejectProfileMember = [&](const std::string &suffix,
                                         const std::string_view expected,
                                         const std::string_view replacement,
                                         const std::string &member) {
        const auto root = workDirectory / suffix;
        createValidFluxFixture(root);
        writeFile(
            root / "vae" / "config.json",
            replaceOnce(fluxVaeConfiguration, expected, replacement));
        requireManifestError(
            [&] { static_cast<void>(iild::FluxModelManifest::load(root)); },
            iild::ModelManifestErrorCode::unsupportedModel,
            member);
    };

    rejectProfileMember(
        "flux-wrong-vae-block-channels",
        "[128, 256, 512, 512]",
        "[128, 256, 256, 512]",
        "block_out_channels");
    rejectProfileMember(
        "flux-wrong-vae-down-blocks",
        R"json(["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"])json",
        R"json(["DownEncoderBlock2D", "CrossAttnDownBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"])json",
        "down_block_types");
    rejectProfileMember(
        "flux-wrong-vae-up-blocks",
        R"json(["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"])json",
        R"json(["UpDecoderBlock2D", "CrossAttnUpBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"])json",
        "up_block_types");
    rejectProfileMember(
        "flux-wrong-vae-mid-attention",
        R"json("mid_block_add_attention": true)json",
        R"json("mid_block_add_attention": false)json",
        "mid_block_add_attention");
}

void rejectsFluxDevGuidanceContract(const fs::path &workDirectory)
{
    const auto root = workDirectory / "flux-dev-guidance";
    createValidFluxFixture(root);
    writeFile(root / "transformer" / "config.json",
              R"json({"_class_name": "FluxTransformer2DModel", "attention_head_dim": 128, "guidance_embeds": true, "in_channels": 64, "joint_attention_dim": 4096, "num_attention_heads": 24, "num_layers": 19, "num_single_layers": 38, "patch_size": 1, "pooled_projection_dim": 768})json");

    requireManifestError(
        [&] { static_cast<void>(iild::FluxModelManifest::load(root)); },
        iild::ModelManifestErrorCode::unsupportedModel,
        "guidance_embeds");
}

void rejectsMismatchedFluxT5Width(const fs::path &workDirectory)
{
    const auto root = workDirectory / "flux-wrong-t5-width";
    createValidFluxFixture(root);
    writeFile(root / "text_encoder_2" / "config.json",
              R"json({"d_ff": 10240, "d_kv": 64, "d_model": 2048, "num_heads": 64, "num_layers": 24, "vocab_size": 32128})json");

    requireManifestError(
        [&] { static_cast<void>(iild::FluxModelManifest::load(root)); },
        iild::ModelManifestErrorCode::unsupportedModel,
        "d_model");
}

void rejectsMismatchedFluxClipProfile(const fs::path &workDirectory)
{
    const auto rejectMember = [&](const std::string_view suffix,
                                  const std::string_view expected,
                                  const std::string_view replacement,
                                  const std::string &member) {
        const auto root = workDirectory / ("flux-wrong-clip-" + std::string{suffix});
        createValidFluxFixture(root);
        writeFile(
            root / "text_encoder" / "config.json",
            replaceOnce(fluxClipTextEncoderConfiguration, expected, replacement));
        requireManifestError(
            [&] { static_cast<void>(iild::FluxModelManifest::load(root)); },
            iild::ModelManifestErrorCode::unsupportedModel,
            member);
    };

    rejectMember(
        "intermediate-size",
        R"json("intermediate_size": 3072)json",
        R"json("intermediate_size": 2048)json",
        "intermediate_size");
    rejectMember(
        "heads",
        R"json("num_attention_heads": 12)json",
        R"json("num_attention_heads": 8)json",
        "num_attention_heads");
    rejectMember(
        "layers",
        R"json("num_hidden_layers": 12)json",
        R"json("num_hidden_layers": 8)json",
        "num_hidden_layers");
    rejectMember(
        "vocabulary",
        R"json("vocab_size": 49408)json",
        R"json("vocab_size": 32000)json",
        "vocab_size");
}

void rejectsMismatchedFluxT5Activation(const fs::path &workDirectory)
{
    const auto root = workDirectory / "flux-wrong-t5-activation";
    createValidFluxFixture(root);
    writeFile(
        root / "text_encoder_2" / "config.json",
        replaceOnce(
            fluxT5TextEncoderConfiguration,
            R"json("feed_forward_proj": "gated-gelu")json",
            R"json("feed_forward_proj": "relu")json"));

    requireManifestError(
        [&] { static_cast<void>(iild::FluxModelManifest::load(root)); },
        iild::ModelManifestErrorCode::unsupportedModel,
        "feed_forward_proj");
}

void rejectsMismatchedFluxPackedChannels(const fs::path &workDirectory)
{
    const auto root = workDirectory / "flux-wrong-packed-channels";
    createValidFluxFixture(root);
    writeFile(root / "transformer" / "config.json",
              R"json({"_class_name": "FluxTransformer2DModel", "attention_head_dim": 128, "guidance_embeds": false, "in_channels": 16, "joint_attention_dim": 4096, "num_attention_heads": 24, "num_layers": 19, "num_single_layers": 38, "patch_size": 1, "pooled_projection_dim": 768})json");

    requireManifestError(
        [&] { static_cast<void>(iild::FluxModelManifest::load(root)); },
        iild::ModelManifestErrorCode::unsupportedModel,
        "in_channels");
}

void rejectsMismatchedFluxVaeShift(const fs::path &workDirectory)
{
    const auto root = workDirectory / "flux-wrong-vae-shift";
    createValidFluxFixture(root);
    writeFile(
        root / "vae" / "config.json",
        replaceOnce(
            fluxVaeConfiguration,
            R"json("shift_factor": 0.1159)json",
            R"json("shift_factor": 0.0)json"));

    requireManifestError(
        [&] { static_cast<void>(iild::FluxModelManifest::load(root)); },
        iild::ModelManifestErrorCode::unsupportedModel,
        "shift_factor");
}

void rejectsMismatchedFluxScheduler(const fs::path &workDirectory)
{
    const auto root = workDirectory / "flux-wrong-scheduler";
    createValidFluxFixture(root);
    writeFile(root / "scheduler" / "scheduler_config.json",
              R"json({"_class_name": "FlowMatchEulerDiscreteScheduler", "base_image_seq_len": 256, "base_shift": 0.5, "max_image_seq_len": 4096, "max_shift": 1.15, "num_train_timesteps": 1000, "shift": 3.0, "use_dynamic_shifting": true})json");

    requireManifestError(
        [&] { static_cast<void>(iild::FluxModelManifest::load(root)); },
        iild::ModelManifestErrorCode::unsupportedModel,
        "shift");
}

void rejectsMissingFluxTransformerDeclaration(const fs::path &workDirectory)
{
    const auto root = workDirectory / "flux-missing-transformer";
    createValidFluxFixture(root);
    writeFile(root / "model_index.json", R"json({
        "_class_name": "FluxPipeline",
        "tokenizer": ["transformers", "CLIPTokenizer"],
        "tokenizer_2": ["transformers", "T5TokenizerFast"],
        "text_encoder": ["transformers", "CLIPTextModel"],
        "text_encoder_2": ["transformers", "T5EncoderModel"],
        "vae": ["diffusers", "AutoencoderKL"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"]
    })json");

    requireManifestError(
        [&] { static_cast<void>(iild::FluxModelManifest::load(root)); },
        iild::ModelManifestErrorCode::invalidPackage,
        "transformer");
}

void rejectsMissingSdxlSecondComponent(const fs::path &workDirectory)
{
    const auto root = workDirectory / "sdxl-missing-second-component";
    createValidSdxlFixture(root);
    writeFile(root / "model_index.json", R"json({
        "_class_name": "StableDiffusionXLPipeline",
        "force_zeros_for_empty_prompt": true,
        "tokenizer": ["transformers", "CLIPTokenizer"],
        "text_encoder": ["transformers", "CLIPTextModel"],
        "text_encoder_2": ["transformers", "CLIPTextModelWithProjection"],
        "unet": ["diffusers", "UNet2DConditionModel"],
        "vae": ["diffusers", "AutoencoderKL"],
        "scheduler": ["diffusers", "EulerDiscreteScheduler"]
    })json");

    requireManifestError(
        [&] { static_cast<void>(iild::StableDiffusionModelManifest::load(root)); },
        iild::ModelManifestErrorCode::invalidPackage,
        "tokenizer_2");
}

void rejectsMismatchedSdxlConditioningContract(const fs::path &workDirectory)
{
    const auto root = workDirectory / "sdxl-wrong-conditioning";
    createValidSdxlFixture(root);
    writeFile(root / "unet" / "config.json",
              R"json({"_class_name": "UNet2DConditionModel", "addition_embed_type": "text_time", "addition_time_embed_dim": 256, "in_channels": 4, "out_channels": 4, "sample_size": 128, "cross_attention_dim": 2048, "projection_class_embeddings_input_dim": 2048, "use_linear_projection": true})json");

    requireManifestError(
        [&] { static_cast<void>(iild::StableDiffusionModelManifest::load(root)); },
        iild::ModelManifestErrorCode::unsupportedModel,
        "projection_class_embeddings_input_dim");
}

void rejectsMismatchedSdxlProjectionMode(const fs::path &workDirectory)
{
    const auto root = workDirectory / "sdxl-wrong-projection-mode";
    createValidSdxlFixture(root);
    writeFile(root / "unet" / "config.json",
              R"json({"_class_name": "UNet2DConditionModel", "addition_embed_type": "text_time", "addition_time_embed_dim": 256, "in_channels": 4, "out_channels": 4, "sample_size": 128, "cross_attention_dim": 2048, "projection_class_embeddings_input_dim": 2816, "use_linear_projection": false})json");

    requireManifestError(
        [&] { static_cast<void>(iild::StableDiffusionModelManifest::load(root)); },
        iild::ModelManifestErrorCode::unsupportedModel,
        "use_linear_projection");
}

void rejectsMismatchedSdxlSchedulerBetas(const fs::path &workDirectory)
{
    const auto root = workDirectory / "sdxl-wrong-scheduler-betas";
    createValidSdxlFixture(root);
    writeFile(root / "scheduler" / "scheduler_config.json",
              R"json({"_class_name": "EulerDiscreteScheduler", "num_train_timesteps": 1000, "beta_start": 0.00085, "beta_end": 0.02, "beta_schedule": "scaled_linear", "prediction_type": "epsilon", "timestep_spacing": "leading"})json");

    requireManifestError(
        [&] { static_cast<void>(iild::StableDiffusionModelManifest::load(root)); },
        iild::ModelManifestErrorCode::unsupportedModel,
        "beta_end");
}

void rejectsMissingRoot(const fs::path &workDirectory)
{
    requireManifestError(
        [&] { static_cast<void>(iild::StableDiffusionModelManifest::load(workDirectory / "missing")); },
        iild::ModelManifestErrorCode::inputUnavailable,
        "model root");
}

void rejectsWrongPipelineClass(const fs::path &workDirectory)
{
    const auto root = workDirectory / "wrong-pipeline";
    createValidFixture(root);
    writeFile(root / "model_index.json", R"json({
        "_class_name": "UnknownPipeline",
        "tokenizer": ["transformers", "CLIPTokenizer"],
        "text_encoder": ["transformers", "CLIPTextModel"],
        "unet": ["diffusers", "UNet2DConditionModel"],
        "vae": ["diffusers", "AutoencoderKL"],
        "scheduler": ["diffusers", "PNDMScheduler"]
    })json");

    requireManifestError(
        [&] { static_cast<void>(iild::StableDiffusionModelManifest::load(root)); },
        iild::ModelManifestErrorCode::unsupportedModel,
        "UnknownPipeline");
}

void rejectsMissingComponentDeclaration(const fs::path &workDirectory)
{
    const auto root = workDirectory / "missing-declaration";
    createValidFixture(root);
    writeFile(root / "model_index.json", R"json({
        "_class_name": "StableDiffusionPipeline",
        "tokenizer": ["transformers", "CLIPTokenizer"],
        "text_encoder": ["transformers", "CLIPTextModel"],
        "unet": ["diffusers", "UNet2DConditionModel"],
        "vae": ["diffusers", "AutoencoderKL"]
    })json");

    requireManifestError(
        [&] { static_cast<void>(iild::StableDiffusionModelManifest::load(root)); },
        iild::ModelManifestErrorCode::invalidPackage,
        "scheduler");
}

void rejectsMissingWeightFile(const fs::path &workDirectory)
{
    const auto root = workDirectory / "missing-weight-file";
    createValidFixture(root);
    fs::remove(root / "unet" / "diffusion_pytorch_model.safetensors");
    writeFile(root / "unet" / "unrelated.safetensors", "not canonical weights");

    requireManifestError(
        [&] { static_cast<void>(iild::StableDiffusionModelManifest::load(root)); },
        iild::ModelManifestErrorCode::invalidPackage,
        "UNet weight file");
}

void rejectsMismatchedTensorContract(const fs::path &workDirectory)
{
    const auto root = workDirectory / "wrong-contract";
    createValidFixture(root);
    writeFile(root / "unet" / "config.json",
              R"json({"_class_name": "UNet2DConditionModel", "in_channels": 4, "out_channels": 4, "sample_size": 64, "cross_attention_dim": 1024})json");

    requireManifestError(
        [&] { static_cast<void>(iild::StableDiffusionModelManifest::load(root)); },
        iild::ModelManifestErrorCode::unsupportedModel,
        "cross_attention_dim");
}

void rejectsMalformedJson(const fs::path &workDirectory)
{
    const auto root = workDirectory / "malformed-json";
    createValidFixture(root);
    writeFile(root / "vae" / "config.json", "not json");

    requireManifestError(
        [&] { static_cast<void>(iild::StableDiffusionModelManifest::load(root)); },
        iild::ModelManifestErrorCode::invalidPackage,
        "vae/config.json");
}

void rejectsNonStrictJson(const fs::path &workDirectory)
{
    const auto root = workDirectory / "non-strict-json";
    createValidFixture(root);
    writeFile(root / "scheduler" / "scheduler_config.json",
              R"json({"_class_name": "PNDMScheduler", "num_train_timesteps": 1000, "beta_schedule": "scaled_linear", "skip_prk_steps": true,})json");

    requireManifestError(
        [&] { static_cast<void>(iild::StableDiffusionModelManifest::load(root)); },
        iild::ModelManifestErrorCode::invalidPackage,
        "scheduler/scheduler_config.json");
}

void rejectsEmbeddedNullInClassName(const fs::path &workDirectory)
{
    const auto root = workDirectory / "embedded-null";
    createValidFixture(root);
    writeFile(root / "model_index.json", R"json({
        "_class_name": "StableDiffusionPipeline\u0000evil",
        "tokenizer": ["transformers", "CLIPTokenizer"],
        "text_encoder": ["transformers", "CLIPTextModel"],
        "unet": ["diffusers", "UNet2DConditionModel"],
        "vae": ["diffusers", "AutoencoderKL"],
        "scheduler": ["diffusers", "PNDMScheduler"]
    })json");

    requireManifestError(
        [&] { static_cast<void>(iild::StableDiffusionModelManifest::load(root)); },
        iild::ModelManifestErrorCode::unsupportedModel,
        "\\x00evil");
}

void rejectsEmbeddedNullInComponentDeclaration(const fs::path &workDirectory)
{
    const auto root = workDirectory / "embedded-null-component";
    createValidFixture(root);
    writeFile(root / "model_index.json", R"json({
        "_class_name": "StableDiffusionPipeline",
        "tokenizer": ["transformers\u0000evil", "CLIPTokenizer"],
        "text_encoder": ["transformers", "CLIPTextModel"],
        "unet": ["diffusers", "UNet2DConditionModel"],
        "vae": ["diffusers", "AutoencoderKL"],
        "scheduler": ["diffusers", "PNDMScheduler"]
    })json");

    requireManifestError(
        [&] { static_cast<void>(iild::StableDiffusionModelManifest::load(root)); },
        iild::ModelManifestErrorCode::unsupportedModel,
        "tokenizer");
}

void acceptsUnknownMetadata(const fs::path &workDirectory)
{
    const auto root = workDirectory / "extra-metadata";
    createValidFixture(root);
    writeFile(root / "unet" / "config.json",
              R"json({"_class_name": "UNet2DConditionModel", "in_channels": 4, "out_channels": 4, "sample_size": 64, "cross_attention_dim": 768, "future_field": "allowed"})json");

    static_cast<void>(iild::StableDiffusionModelManifest::load(root));
}

void acceptsResolvedWeightSymlinks(const fs::path &workDirectory)
{
    const auto root = workDirectory / "symlinked-weight";
    const auto blob = workDirectory / "shared-blobs" / "unet.safetensors";
    createValidFixture(root);
    writeFile(blob, "shared test weights");
    fs::remove(root / "unet" / "diffusion_pytorch_model.safetensors");
    fs::create_symlink(blob, root / "unet" / "diffusion_pytorch_model.safetensors");

    static_cast<void>(iild::StableDiffusionModelManifest::load(root));
}

void rejectsOversizedConfiguration(const fs::path &workDirectory)
{
    const auto root = workDirectory / "oversized-configuration";
    createValidFixture(root);
    writeFile(root / "model_index.json", std::string(1024U * 1024U + 1U, ' '));

    requireManifestError(
        [&] { static_cast<void>(iild::StableDiffusionModelManifest::load(root)); },
        iild::ModelManifestErrorCode::invalidPackage,
        "1 MiB");
}

} // namespace

int main()
{
    const fs::path workDirectory{IILD_TEST_WORK_DIRECTORY};
    fs::remove_all(workDirectory);
    fs::create_directories(workDirectory);

    int failures = 0;
    const auto run = [&](const std::string &name, const std::function<void()> &test) {
        try
        {
            test();
            std::cout << "PASS " << name << '\n';
        }
        catch (const std::exception &error)
        {
            ++failures;
            std::cerr << "FAIL " << name << ": " << error.what() << '\n';
        }
    };

    run("loads Stable Diffusion v1 package", [&] { loadsStableDiffusionV1Package(workDirectory); });
    run("loads Stable Diffusion XL Base package", [&] { loadsStableDiffusionXlBasePackage(workDirectory); });
    run("loads FLUX.1-schnell package", [&] { loadsFlux1SchnellPackage(workDirectory); });
    run("accepts FLUX out-channel default forms", [&] { acceptsFluxOutChannelDefaultForms(workDirectory); });
    run("rejects mismatched FLUX out channels", [&] { rejectsMismatchedFluxOutChannels(workDirectory); });
    run("rejects mismatched FLUX RoPE axes", [&] { rejectsMismatchedFluxAxes(workDirectory); });
    run("rejects missing FLUX shard index", [&] { rejectsMissingFluxShardIndex(workDirectory); });
    run("rejects missing FLUX shard", [&] { rejectsMissingFluxShard(workDirectory); });
    run("rejects mismatched FLUX shard index", [&] { rejectsMismatchedFluxShardIndex(workDirectory); });
    run("rejects mismatched FLUX VAE profile", [&] { rejectsMismatchedFluxVaeProfile(workDirectory); });
    run("rejects FLUX.1-dev guidance contract", [&] { rejectsFluxDevGuidanceContract(workDirectory); });
    run("rejects mismatched FLUX T5 width", [&] { rejectsMismatchedFluxT5Width(workDirectory); });
    run("rejects mismatched FLUX CLIP profile", [&] { rejectsMismatchedFluxClipProfile(workDirectory); });
    run("rejects mismatched FLUX T5 activation", [&] { rejectsMismatchedFluxT5Activation(workDirectory); });
    run("rejects mismatched FLUX packed channels", [&] { rejectsMismatchedFluxPackedChannels(workDirectory); });
    run("rejects mismatched FLUX VAE shift", [&] { rejectsMismatchedFluxVaeShift(workDirectory); });
    run("rejects mismatched FLUX scheduler", [&] { rejectsMismatchedFluxScheduler(workDirectory); });
    run("rejects missing FLUX transformer declaration", [&] { rejectsMissingFluxTransformerDeclaration(workDirectory); });
    run("rejects missing SDXL second component", [&] { rejectsMissingSdxlSecondComponent(workDirectory); });
    run("rejects mismatched SDXL conditioning contract", [&] { rejectsMismatchedSdxlConditioningContract(workDirectory); });
    run("rejects mismatched SDXL projection mode", [&] { rejectsMismatchedSdxlProjectionMode(workDirectory); });
    run("rejects mismatched SDXL scheduler betas", [&] { rejectsMismatchedSdxlSchedulerBetas(workDirectory); });
    run("rejects missing root", [&] { rejectsMissingRoot(workDirectory); });
    run("rejects wrong pipeline class", [&] { rejectsWrongPipelineClass(workDirectory); });
    run("rejects missing component declaration", [&] { rejectsMissingComponentDeclaration(workDirectory); });
    run("rejects missing weight file", [&] { rejectsMissingWeightFile(workDirectory); });
    run("rejects mismatched tensor contract", [&] { rejectsMismatchedTensorContract(workDirectory); });
    run("rejects malformed JSON", [&] { rejectsMalformedJson(workDirectory); });
    run("rejects non-strict JSON", [&] { rejectsNonStrictJson(workDirectory); });
    run("rejects embedded NUL in class name", [&] { rejectsEmbeddedNullInClassName(workDirectory); });
    run("rejects embedded NUL in component declaration", [&] { rejectsEmbeddedNullInComponentDeclaration(workDirectory); });
    run("accepts unknown metadata", [&] { acceptsUnknownMetadata(workDirectory); });
    run("accepts resolved weight symlinks", [&] { acceptsResolvedWeightSymlinks(workDirectory); });
    run("rejects oversized configuration", [&] { rejectsOversizedConfiguration(workDirectory); });

    fs::remove_all(workDirectory);
    return failures == 0 ? 0 : 1;
}
