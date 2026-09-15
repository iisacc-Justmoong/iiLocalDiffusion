#include <stable-diffusion.h>
#include <model_loader.h>
#include <model_manager.h>
#include <model/vae/wan_vae.hpp>
#include <model/vae/auto_encoder_kl.hpp>
#include <ggml-cpu.h>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
void fixture(const std::filesystem::path &path, const std::vector<std::string> &keys) {
    std::string header = "{";
    for (std::size_t i = 0; i < keys.size(); ++i) {
        if (i) header += ',';
        header += '"' + keys[i] + "\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":["
            + std::to_string(i * 4) + ',' + std::to_string((i + 1) * 4) + "]}";
    }
    header += '}';
    const std::uint64_t length = header.size();
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(&length), sizeof(length));
    file << header;
    const float zero = 0;
    for (std::size_t i = 0; i < keys.size(); ++i) file.write(reinterpret_cast<const char *>(&zero), sizeof(zero));
}
void metadataFixture(const std::filesystem::path &path, const std::vector<std::string> &modelKeys,
                     std::vector<TensorStorage> vae, const std::string &prefix) {
    std::string header = "{";
    std::uint64_t offset = 0;
    for (const auto &key : modelKeys) vae.emplace_back(key, GGML_TYPE_F32, nullptr, 0, 0);
    for (std::size_t i = 0; i < vae.size(); ++i) {
        auto &tensor = vae[i];
        if (i) header += ',';
        auto name = tensor.name;
        if (name.find("first_stage_model.") == 0) name.replace(0, 18, prefix);
        header += '"' + name + "\":{\"dtype\":\"F32\",\"shape\":[";
        if (!tensor.n_dims) header += '1';
        for (int dim = tensor.n_dims - 1; dim >= 0; --dim) {
            if (dim != tensor.n_dims - 1) header += ',';
            header += std::to_string(tensor.ne[dim]);
        }
        const auto end = offset + tensor.nelements() * sizeof(float);
        header += "],\"data_offsets\":[" + std::to_string(offset) + ',' + std::to_string(end) + "]}";
        offset = end;
    }
    header += '}';
    const std::uint64_t length = header.size();
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(&length), sizeof(length));
    file << header;
    // Sparse metadata fixture: the mounting inspector must never load weights.
    file.seekp(length + 8 + offset - 1);
    file.put(0);
}
struct Family {
    const char *name;
    const char *directory;
    SDVersion version;
    int channels;
    std::vector<std::string> keys;
};
void decode(const std::filesystem::path &model, const std::filesystem::path &weights, const Family &family,
            int width = 64, int height = 64, bool tiled = false) {
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(), ggml_backend_free);
    auto manager = std::make_shared<ModelManager>();
    manager->set_n_threads(4);
    manager->set_enable_mmap(true);
    require(manager->loader().init_from_file(model.string()), "Cannot read decoder fixture");
    require(manager->loader().init_from_file(weights.string(), "vae."), "Cannot load official VAE for decode");
    manager->loader().convert_tensors_name();
    std::unique_ptr<VAE> vae;
    if (sd_version_uses_wan_vae(family.version))
        vae = std::make_unique<WAN::WanVAERunner>(cpu.get(), manager->loader().get_tensor_storage_map(),
            "first_stage_model", true, family.version, manager);
    else
        vae = std::make_unique<AutoEncoderKL>(cpu.get(), manager->loader().get_tensor_storage_map(),
            "first_stage_model", true, false, family.version, manager);
    // Release mapped storage before the runner's parameter contexts are freed.
    struct ReleaseStorage {
        ModelManager &manager;
        ~ReleaseStorage() { manager.unregister_param_tensors("VAE"); }
    } release{*manager};
    std::map<std::string, ggml_tensor *> parameters;
    vae->get_param_tensors(parameters);
    require(manager->register_param_tensors("VAE", parameters, ModelManager::ResidencyMode::ParamBackend,
        cpu.get(), cpu.get()), "Cannot register official decoder weights");
    require(manager->validate_registered_tensors(), "Missing or incompatible official decoder weights");
    const int scale = family.version == VERSION_FLUX2_KLEIN ? 16 : 8;
    sd::Tensor<float> latents({width / scale, height / scale, family.channels, 1});
    sd_img_gen_params_t defaults;
    sd_img_gen_params_init(&defaults);
    auto tiling = defaults.vae_tiling_params;
    tiling.enabled = tiled;
    const auto started = std::chrono::steady_clock::now();
    const auto pixels = vae->decode(4, vae->diffusion_to_vae_latents(latents), tiling, false, false, false, true);
    require(pixels.numel() == width * height * 3, "Official native VAE returned an incomplete RGB canvas");
    for (float value : pixels.values()) require(std::isfinite(value), "Nonfinite native VAE output");
    std::cout << family.name << ": actual official native CPU VAE decode passed, " << width << 'x' << height
              << " RGB, tiled=" << tiled << ", overlap=" << tiling.target_overlap
              << ", " << parameters.size() << " weight tensors, "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() << " seconds\n";
}
}

int main(int argc, char **argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--inspect") {
            sd_model_vae_info_t info{};
            require(sd_model_inspect_vae(argv[2], &info), "Cannot inspect the model");
            std::cout << "{\"model_family\":\"" << info.model_family << "\",\"vae_family\":\""
                << info.vae_family << "\",\"state\":" << info.state << "}\n";
            return 0;
        }
        if (argc != 3 && argc != 4) return 1;
        const std::string mode = argc == 4 ? argv[3] : "";
        require(mode.empty() || mode == "--decode" || mode == "--decode-portrait", "Unknown decoder test mode");
        const auto directory = std::filesystem::path(argv[1]);
        std::filesystem::create_directories(directory);
        const auto model = directory / "model.safetensors";
        const std::string qwen = "model.diffusion_model.transformer_blocks.0.img_mod.1.weight";
        const std::vector<Family> families = {
            {"qwen-image", "qwen-image/vae", VERSION_QWEN_IMAGE, 16, {qwen}},
            {"sdxl-base", "sdxl", VERSION_SDXL, 4, {"model.diffusion_model.input_blocks.0.0.weight",
                "model.diffusion_model.middle_block.1.proj_in.weight", "conditioner.embedders.1.model.positional_embedding"}},
            {"flux1", "flux1", VERSION_FLUX, 16, {"model.diffusion_model.double_blocks.0.img_attn.qkv.weight"}},
            {"flux2", "flux2/vae", VERSION_FLUX2_KLEIN, 128, {"model.diffusion_model.double_stream_modulation_img.lin.weight"}},
            {"qwen-image", "qwen-image/vae", VERSION_ANIMA, 16,
                {"model.diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj.weight"}},
            {"flux1", "flux1", VERSION_Z_IMAGE, 16, {"model.diffusion_model.cap_embedder.0.weight"}},
        };
        const char *missing = nullptr;
        sd_model_vae_info_t info{};
        for (const auto &family : families) {
            fixture(model, family.keys);
            require(sd_model_inspect_vae(model.string().c_str(), &info) && info.state == SD_VAE_MISSING
                    && std::string(info.vae_family) == family.name,
                    "Input tensor metadata must identify a missing VAE independently of its filename");
            const auto modelFamily = family.version == VERSION_ANIMA ? "anima"
                : family.version == VERSION_Z_IMAGE ? "z-image" : family.name;
            require(std::string(info.model_family) == modelFamily, "Model and VAE families must remain distinct");
            require(sd_model_missing_vae_family(model.string().c_str(), &missing) && std::string(missing) == family.name,
                    "Missing VAE must select the correct model family");
            const auto weights = std::filesystem::path(argv[2]) / family.directory / "diffusion_pytorch_model.safetensors";
            require(sd_model_validate_vae(model.string().c_str(), weights.string().c_str()),
                    "The official external VAE must satisfy the actual loader contract");
            const auto wrongWeights = std::filesystem::path(argv[2])
                / (family.channels == 4 ? "qwen-image/vae" : "sdxl") / "diffusion_pytorch_model.safetensors";
            require(!sd_model_validate_vae(model.string().c_str(), wrongWeights.string().c_str()),
                    "An external VAE from an incompatible architecture must be rejected before mounting");
            ModelLoader loader;
            require(loader.init_from_file(model.string()), "Cannot read model metadata");
            require(loader.init_from_file(weights.string(), "vae."), "Cannot read official VAE");
            loader.convert_tensors_name();
            const auto &tensors = loader.get_tensor_storage_map();
            std::vector<TensorStorage> completeVae;
            for (const auto &[name, tensor] : tensors)
                if (name.find("first_stage_model.") == 0) completeVae.push_back(tensor);
            const bool isQwen = sd_version_uses_wan_vae(family.version);
            require(tensors.count(isQwen ? "first_stage_model.decoder.conv1.weight" : "first_stage_model.decoder.conv_in.weight"),
                    "Official decoder names were not converted");
            require(tensors.count(isQwen ? "first_stage_model.encoder.conv1.weight" : "first_stage_model.encoder.conv_in.weight"),
                    "Official encoder names were not converted");
            if (mode == "--decode") decode(model, weights, family);
            if (mode == "--decode-portrait" && family.version == VERSION_SDXL)
                decode(model, weights, family, 1024, 1856, true);
            for (const auto &prefix : {"first_stage_model.", "vae."}) {
                auto keys = family.keys;
                keys.push_back(std::string(prefix) + "decoder.conv_in.weight");
                fixture(model, keys);
                require(sd_model_missing_vae_family(model.string().c_str(), &missing) && std::string(missing) == family.name,
                        "A lone VAE tensor is not a complete compatible VAE");
                metadataFixture(model, family.keys, completeVae, prefix);
                require(sd_model_inspect_vae(model.string().c_str(), &info) && info.state == SD_VAE_EMBEDDED,
                        "A complete embedded VAE must be distinguished from unsupported inspection");
                require(sd_model_missing_vae_family(model.string().c_str(), &missing) && std::string(missing).empty(),
                        "A complete compatible embedded VAE must take precedence");
                auto partial = completeVae;
                const auto decoder = std::find_if(partial.begin(), partial.end(), [](const auto &tensor) {
                    return tensor.name.find(".decoder.") != std::string::npos;
                });
                require(decoder != partial.end(), "Decoder fixture is empty");
                partial.erase(decoder);
                const auto external = directory / "partial-external.safetensors";
                metadataFixture(external, {}, partial, "");
                require(!sd_model_validate_vae(model.string().c_str(), external.string().c_str()),
                        "A partial external VAE must not borrow missing tensors from a complete embedded VAE");
                metadataFixture(model, family.keys, partial, prefix);
                require(sd_model_inspect_vae(model.string().c_str(), &info) && info.state == SD_VAE_INCOMPATIBLE,
                        "A partial embedded VAE must report incompatibility before fallback selection");
                require(sd_model_missing_vae_family(model.string().c_str(), &missing) && std::string(missing) == family.name,
                        "An incomplete decoder must select the matching fallback");
                auto incompatible = completeVae;
                for (auto &tensor : incompatible) {
                    if (tensor.name.find(".decoder.") != std::string::npos) { ++tensor.ne[0]; break; }
                }
                metadataFixture(model, family.keys, incompatible, prefix);
                require(sd_model_missing_vae_family(model.string().c_str(), &missing) && std::string(missing) == family.name,
                        "Incompatible VAE shapes must select the matching fallback");
                ModelLoader replacement;
                require(replacement.init_from_file(model.string()) && replacement.init_from_file(weights.string(), "vae."),
                        "Cannot mount the replacement over incompatible embedded tensors");
                replacement.convert_tensors_name();
                for (const auto &tensor : completeVae) {
                    const auto &mounted = replacement.get_tensor_storage_map().at(tensor.name);
                    require(std::equal(tensor.ne, tensor.ne + SD_MAX_DIMS, mounted.ne),
                            "External fallback must replace incompatible embedded tensors after name conversion");
                }
            }
        }
        fixture(model, {qwen, "model.diffusion_model.time_text_embed.addition_t_embedding.weight"});
        require(sd_model_inspect_vae(model.string().c_str(), &info) && info.state == SD_VAE_UNSUPPORTED
            && std::string(info.model_family) == "qwen-image-layered" && std::string(info.vae_family).empty(),
            "RGBA must not be reported as a validated RGB VAE");
        require(sd_model_missing_vae_family(model.string().c_str(), &missing) && std::string(missing).empty(),
                "The RGB fallback is incompatible with Qwen Layered RGBA");
        fixture(model, {"model.diffusion_model.joint_blocks.0.weight"});
        require(sd_model_inspect_vae(model.string().c_str(), &info) && info.state == SD_VAE_MISSING
            && std::string(info.vae_family) == "sd3", "SD3 must retain its distinct VAE contract");
        require(sd_model_missing_vae_family(model.string().c_str(), &missing) && std::string(missing).empty(),
                "SD3's 16 latent channels do not make its VAE interchangeable");
        require(!sd_model_missing_vae_family(nullptr, &missing), "Invalid input accepted");
        fixture(model, {"unknown.weight"});
        require(sd_model_inspect_vae(model.string().c_str(), &info) && info.state == SD_VAE_UNSUPPORTED,
            "Unknown model metadata must not be reported as a valid embedded VAE");
        std::cout << "Native family VAE detection, embedded precedence, exclusions and official weight conversion passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
