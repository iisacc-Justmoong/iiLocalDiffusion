#include <name_conversion.h>
#include <model_loader.h>
#include <stable-diffusion.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}
void fixture(const std::filesystem::path &path, const std::string &prefix, bool embeddedLlm = false) {
    std::string header = "{";
    std::uint64_t offset = 0;
    for (const auto *name : {"txtfusion.projector.weight", "first.weight", "blocks.0.attn.wq.weight",
                             "txtmlp.1.weight", "last.linear.weight"}) {
        if (offset) header += ',';
        header += '"' + prefix + name + "\":{\"dtype\":\"F32\",\"shape\":[1,1],\"data_offsets\":["
            + std::to_string(offset) + ',' + std::to_string(offset + 4) + "]}";
        offset += 4;
    }
    if (embeddedLlm) {
        header += ",\"text_encoders.qwen3vl_4b.transformer.model.embed_tokens.weight\":{\"dtype\":\"F32\",\"shape\":[1,1],\"data_offsets\":["
            + std::to_string(offset) + ',' + std::to_string(offset + 4) + "]}";
        offset += 4;
    }
    header += '}';
    const auto length = static_cast<std::uint64_t>(header.size());
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(&length), sizeof(length));
    file << header;
    file.seekp(8 + length + offset - 1); file.put(0);
    require(bool(file), "Cannot write metadata fixture");
}
}

int main(int argc, char **argv) try {
    for (const auto *suffix : {"model.embed_tokens.weight", "model.layers.0.mlp.down_proj.weight", "model.layers.0.mlp.down_proj.weight_scale", "visual.blocks.0.attn.qkv.weight"}) {
        const std::string canonical = std::string("text_encoders.llm.") + suffix;
        const std::string comfy = std::string("text_encoders.qwen3vl_4b.transformer.") + suffix;
        if (convert_tensor_name(comfy, VERSION_KREA2) != convert_tensor_name(canonical, VERSION_KREA2)) {
            std::cerr << comfy << " was not normalized\n"; return 1;
        }
    }
    for (const auto *suffix : {"embed_tokens.weight", "layers.31.self_attn.q_proj.weight",
                              "layers.0.mlp.down_proj.weight_scale", "norm.weight"}) {
        const auto canonical = std::string("text_encoders.llm.model.") + suffix;
        const auto original = std::string("text_encoders.llm.model.language_model.") + suffix;
        require(convert_tensor_name(original, VERSION_KREA2) == canonical,
                "Original Qwen3-VL text tower was not mounted: " + original);
        require(convert_tensor_name(std::string("text_encoders.qwen3vl_4b.transformer.model.language_model.")
                                   + suffix, VERSION_KREA2) == canonical,
                "Embedded original Qwen3-VL text tower was not mounted");
    }
    for (const auto *name : {"first.weight", "blocks.0.attn.wq.weight", "blocks.0.attn.wq.weight_scale",
                             "txtfusion.projector.weight", "txtfusion.refiner_blocks.0.attn.gate.weight",
                             "txtmlp.1.weight", "tmlp.0.weight", "tproj.weight", "last.linear.weight"}) {
        const auto expected = convert_tensor_name(std::string("model.diffusion_model.") + name, VERSION_KREA2);
        require(convert_tensor_name(name, VERSION_KREA2) == expected,
                std::string("Bare Krea2 denoiser tensor was not mounted: ") + name);
    }
    for (const auto *name : {"backup.blocks.0.attn.wq.weight", "first_stage_model.encoder.conv1.weight",
                             "text_encoders.llm.model.embed_tokens.weight"})
        require(convert_tensor_name(name, VERSION_KREA2) == name,
                std::string("Unrelated component was renamed: ") + name);
    require(argc == 2, "Pass the fixture directory");
    const auto root = std::filesystem::absolute(argv[1]);
    std::filesystem::create_directories(root);
    int index = 0;
    for (const auto *prefix : {"", "diffusion_model.", "model.diffusion_model.", "transformer."}) {
        const auto path = root / ("layout-" + std::to_string(index++) + ".safetensors");
        fixture(path, prefix);
        ModelLoader loader;
        require(loader.init_from_file(path.string()), "Cannot read the model metadata");
        loader.convert_tensors_name();
        require(loader.get_sd_version() == VERSION_KREA2,
                std::string("Krea2 layout not identified: ") + prefix);
        const auto &tensors = loader.get_tensor_storage_map();
        require(tensors.find("model.diffusion_model.first.weight") != tensors.end(),
                "Denoiser prefix is missing after conversion");
        sd_model_vae_info_t info{};
        require(sd_model_inspect_vae(path.string().c_str(), &info)
            && std::string(info.model_family) == "krea2", "VAE inspector disagrees with model loading");
        bool embedded = true;
        require(sd_model_has_embedded_llm(path.string().c_str(), &embedded) && !embedded,
                "Split denoiser was reported as an embedded encoder");
    }
    const auto partial = root / "partial-embedded.safetensors";
    fixture(partial, "", true);
    bool embedded = false;
    require(sd_model_has_embedded_llm(partial.string().c_str(), &embedded) && embedded,
            "Partial embedded encoder would be silently replaced");
    require(!sd_model_validate_krea2_llm(partial.string().c_str()),
            "Partial model was accepted as a complete Qwen3-VL-4B encoder");
    require(!sd_model_has_embedded_llm(nullptr, &embedded)
        && !sd_model_has_embedded_llm(partial.string().c_str(), nullptr)
        && !sd_model_validate_krea2_llm(""), "Invalid inspection input was accepted");
    const auto decoy = root / "backup.safetensors";
    fixture(decoy, "backup.");
    ModelLoader loader;
    require(loader.init_from_file(decoy.string()), "Cannot read negative fixture");
    require(loader.get_sd_version() != VERSION_KREA2, "Backup namespace was accepted as a denoiser");
    std::cout << "Krea2 split/full layouts, embedded encoders and quant scales normalized; unrelated names preserved\n";
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n'; return 1;
}
