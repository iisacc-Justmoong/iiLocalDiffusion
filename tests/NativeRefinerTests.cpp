#include "model_loader.h"
#include "name_conversion.h"
#include "model/diffusion/unet.hpp"
#include "conditioning/conditioner.hpp"
#include <ggml-cpu.h>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>

void require(bool ok, const std::string &message) {
    if (!ok) throw std::runtime_error(message);
}

void add(ModelLoader &loader, const std::string &name, std::initializer_list<int64_t> shape) {
    const std::vector<int64_t> dims(shape);
    loader.get_tensor_storage_map()[name] = TensorStorage(name, GGML_TYPE_F16,
        dims.data(), static_cast<int>(dims.size()), 0);
}

void detection() {
    for (bool diffusers : {false, true}) {
        ModelLoader loader;
        const auto input = diffusers ? "unet.conv_in.weight" : "model.diffusion_model.input_blocks.0.0.weight";
        const auto adm = diffusers ? "unet.add_embedding.linear_1.weight" : "model.diffusion_model.label_emb.0.0.weight";
        const auto token = diffusers ? "text_encoder_2.text_model.embeddings.token_embedding.weight"
                                    : "conditioner.embedders.0.model.token_embedding.weight";
        add(loader, input, {3, 3, 4, 384});
        add(loader, adm, {2560, 1536});
        require(loader.get_sd_version() == VERSION_SDXL_REFINER, "UNet-only Refiner detection");
        add(loader, token, {1280, 49408});
        require(loader.get_sd_version() == VERSION_SDXL_REFINER, "Checkpoint Refiner detection");
        loader.convert_tensors_name();
        require(loader.get_sd_version() == VERSION_SDXL_REFINER, "Converted Refiner detection");
        auto &storage = loader.get_tensor_storage_map();
        auto &inputTensor = storage.at("model.diffusion_model.input_blocks.0.0.weight");
        inputTensor.ne[2] = 9;
        require(loader.get_sd_version() == VERSION_COUNT, "Unsupported Refiner input channels accepted");
        inputTensor.ne[2] = 4;
        auto &admTensor = storage.at("model.diffusion_model.label_emb.0.0.weight");
        admTensor.ne[1] = 1280;
        require(loader.get_sd_version() == VERSION_COUNT, "Invalid Refiner time width accepted");
        admTensor.ne[1] = 1536;
        auto &tokenTensor = storage.at("cond_stage_model.transformer.text_model.embeddings.token_embedding.weight");
        tokenTensor.ne[0] = 768;
        require(loader.get_sd_version() == VERSION_COUNT, "CLIP-L accepted as Refiner bigG");
        tokenTensor.ne[0] = 1280;
        add(loader, "cond_stage_model.1.transformer.text_model.embeddings.token_embedding.weight", {1280, 49408});
        require(loader.get_sd_version() == VERSION_COUNT, "Unexpected second Refiner encoder accepted");
    }
    ModelLoader base;
    add(base, "model.diffusion_model.input_blocks.0.0.weight", {3, 3, 4, 320});
    add(base, "model.diffusion_model.label_emb.0.0.weight", {2816, 1280});
    add(base, "model.diffusion_model.middle_block.1.proj_in.weight", {1280, 1280});
    add(base, "cond_stage_model.1.transformer.text_model.embeddings.token_embedding.weight", {1280, 49408});
    require(base.get_sd_version() == VERSION_SDXL, "Base SDXL detection changed");
}

void names() {
    const std::vector<std::pair<std::string, std::string>> pairs = {
        {"conv_in.weight", "input_blocks.0.0.weight"},
        {"add_embedding.linear_1.weight", "label_emb.0.0.weight"},
        {"add_embedding.linear_2.bias", "label_emb.0.2.bias"},
        {"down_blocks.1.attentions.0.transformer_blocks.3.attn2.to_k.weight", "input_blocks.4.1.transformer_blocks.3.attn2.to_k.weight"},
        {"down_blocks.2.attentions.1.proj_in.weight", "input_blocks.8.1.proj_in.weight"},
        {"down_blocks.3.resnets.1.conv1.weight", "input_blocks.11.0.in_layers.2.weight"},
        {"down_blocks.2.downsamplers.0.conv.weight", "input_blocks.9.0.op.weight"},
        {"up_blocks.0.upsamplers.0.conv.weight", "output_blocks.2.1.conv.weight"},
        {"up_blocks.1.upsamplers.0.conv.weight", "output_blocks.5.2.conv.weight"},
        {"up_blocks.2.upsamplers.0.conv.weight", "output_blocks.8.2.conv.weight"},
        {"up_blocks.3.resnets.2.conv2.weight", "output_blocks.11.0.out_layers.3.weight"},
        {"mid_block.attentions.0.transformer_blocks.3.attn2.to_k.weight", "middle_block.1.transformer_blocks.3.attn2.to_k.weight"},
        {"conv_out.weight", "out.2.weight"},
    };
    for (const auto &[source, destination] : pairs) {
        const auto expected = "model.diffusion_model." + destination;
        require(convert_tensor_name("unet." + source, VERSION_SDXL_REFINER) == expected, "Refiner name: " + source);
        require(convert_tensor_name(expected, VERSION_SDXL_REFINER) == expected, "Canonical name changed: " + expected);
    }
    const std::string token = "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight";
    for (const auto &source : {"conditioner.embedders.0.model.token_embedding.weight",
            "text_encoder_2.text_model.embeddings.token_embedding.weight",
            "clip_g.text_model.embeddings.token_embedding.weight"})
        require(convert_tensor_name(source, VERSION_SDXL_REFINER) == token, "Refiner text encoder name");
}

void architecture(ggml_backend_t backend) {
    const auto config = UNetConfig::detect_from_weights({}, "model.diffusion_model", VERSION_SDXL_REFINER);
    require(sd_version_is_sdxl(config.version) && sd_version_is_unet(config.version), "Refiner family predicates");
    require(config.model_channels == 384 && config.time_embed_dim == 1536 && config.context_dim == 1280
        && config.adm_in_channels == 2560 && config.num_head_channels == 64 && config.num_res_blocks == 2
        && config.use_linear_projection, "Refiner UNet dimensions");
    require(config.channel_mult == std::vector<int>({1, 2, 4, 4})
        && config.attention_resolutions == std::vector<int>({4, 2})
        && config.transformer_depth == std::vector<int>({4, 4, 4, 4}), "Refiner UNet topology");
    // Actual production architecture; only tensor metadata, no pretrained data allocation.
    UNetModelRunner unet(backend, {}, "model.diffusion_model", VERSION_SDXL_REFINER);
    std::map<std::string, ggml_tensor *> tensors;
    unet.get_param_tensors(tensors, "model.diffusion_model");
    const auto shape = [&](const std::string &name, std::initializer_list<int64_t> expected) {
        const auto found = tensors.find("model.diffusion_model." + name);
        require(found != tensors.end(), "Missing Refiner tensor: " + name);
        require(std::equal(expected.begin(), expected.end(), found->second->ne), "Refiner tensor shape: " + name);
    };
    shape("input_blocks.0.0.weight", {3, 3, 4, 384});
    shape("label_emb.0.0.weight", {2560, 1536});
    shape("input_blocks.4.1.transformer_blocks.3.attn2.to_k.weight", {1280, 768});
    shape("input_blocks.8.1.transformer_blocks.3.attn2.to_k.weight", {1280, 1536});
    shape("middle_block.1.transformer_blocks.3.attn2.to_k.weight", {1280, 1536});
    shape("output_blocks.11.0.out_layers.3.weight", {3, 3, 384, 384});
    shape("out.2.weight", {3, 3, 384, 4});
    for (const auto &[name, tensor] : tensors) {
        require(name.find("input_blocks.1.1.") == std::string::npos, "Unexpected level-zero attention");
        require(name.find("input_blocks.10.1.") == std::string::npos, "Unexpected deepest input attention");
        require(name.find("transformer_blocks.4.") == std::string::npos, "Refiner transformer too deep");
    }
    FrozenCLIPEmbedderWithCustomWords conditioner(backend, {}, {}, VERSION_SDXL_REFINER);
    require(conditioner.text_model && !conditioner.text_model2, "Refiner must have one encoder");
    require(conditioner.text_model->model.hidden_size == 1280 && !conditioner.text_model->model.with_final_ln,
        "Refiner must use penultimate bigG hidden states");
    tensors.clear();
    conditioner.get_param_tensors(tensors);
    require(tensors.at("cond_stage_model.transformer.text_model.embeddings.token_embedding.weight")->ne[0] == 1280,
        "Refiner embedding parameter shape");
    for (const auto &[name, tensor] : tensors)
        require(name.find("cond_stage_model.1.") == std::string::npos, "Second encoder parameter emitted");
    conditioner.set_max_graph_vram_bytes(0);
    conditioner.set_runtime_backends({backend});
    conditioner.set_graph_cut_layer_split_enabled(false);
    conditioner.set_graph_cut_layer_split_backend_vram_limits({});
    conditioner.set_flash_attention_enabled(false);
    conditioner.set_weight_adapter(nullptr);
    conditioner.runner_end(); // Optional second encoder must not be dereferenced.
    const auto base = UNetConfig::detect_from_weights({}, "model.diffusion_model", VERSION_SDXL);
    require(base.model_channels == 320 && base.context_dim == 2048 && base.adm_in_channels == 2816
        && base.channel_mult == std::vector<int>({1, 2, 4}), "Base UNet configuration changed");
}

void conditioning() {
    sd::Tensor<float> pooled({1280});
    for (size_t i = 0; i < pooled.values().size(); ++i) pooled.values()[i] = float(i) / 1280;
    for (const auto version : {VERSION_SDXL, VERSION_SDXL_REFINER}) {
        for (const bool negative : {false, true}) {
            const auto vector = FrozenCLIPEmbedderWithCustomWords::sdxl_condition_vector(version, pooled, 832, 1216, negative);
            const bool refiner = version == VERSION_SDXL_REFINER;
            require(vector.numel() == (refiner ? 2560 : 2816), "ADM vector length");
            require(std::equal(pooled.values().begin(), pooled.values().end(), vector.values().begin()), "Pooled bigG changed");
            std::vector<float> values{1216, 832, 0, 0};
            if (refiner) values.push_back(negative ? 2.5f : 6.f);
            else { values.push_back(1216); values.push_back(832); }
            // Independent numerical oracle, including non-square H/W ordering.
            for (size_t component = 0; component < values.size(); ++component)
                for (int frequency = 0; frequency < 128; ++frequency) {
                    const double angle = values[component] * std::pow(10000., -double(frequency) / 128);
                    const auto offset = 1280 + component * 256 + frequency;
                    require(std::abs(vector.values()[offset] - std::cos(angle)) < 0.00015, "ADM cosine mismatch");
                    require(std::abs(vector.values()[offset + 128] - std::sin(angle)) < 0.00015, "ADM sine mismatch");
                }
        }
    }
    require(FrozenCLIPEmbedderWithCustomWords::sdxl_condition_vector(VERSION_SD1, pooled, 832, 1216, false).empty(), "Non-SDXL ADM accepted");
    require(FrozenCLIPEmbedderWithCustomWords::sdxl_condition_vector(VERSION_SDXL_REFINER, {}, 832, 1216, false).empty(), "Empty pooled vector accepted");
}

int main() {
    try {
        detection();
        names();
        auto backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(ggml_backend_cpu_init(), &ggml_backend_free);
        require(bool(backend), "CPU metadata backend");
        architecture(backend.get());
        conditioning();
        std::cout << "Native Refiner detection, UNet/CLIP metadata and numerical conditioning passed\n";
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
