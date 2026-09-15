// Opt-in comparison with an independent float32 Diffusers decoder. No denoiser.
#include <model_manager.h>
#include <model/vae/auto_encoder_kl.hpp>
#include <ggml-cpu.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
    if (argc != 7) return 1;
    const int width = std::stoi(argv[3]), height = std::stoi(argv[4]);
    const int tile = std::stoi(argv[5]);
    if (width < 64 || height < 64 || width % 8 || height % 8 || tile < 0) return 1;
    const char *missingFamily = nullptr;
    if (!sd_model_missing_vae_family(argv[1], &missingFamily) || !missingFamily || *missingFamily) return 9;
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(), ggml_backend_free);
    auto manager = std::make_shared<ModelManager>();
    manager->set_n_threads(4);
    manager->set_enable_mmap(true);
    if (!manager->loader().init_from_file(argv[1])) return 2;
    manager->loader().convert_tensors_name();
    if (!sd_version_is_sdxl(manager->loader().get_sd_version())) return 3;
    AutoEncoderKL vae(cpu.get(), manager->loader().get_tensor_storage_map(), "first_stage_model",
                      true, false, VERSION_SDXL, manager);
    struct Release {
        ModelManager &manager;
        ~Release() { manager.unregister_param_tensors("VAE"); }
    } release{*manager};
    vae.set_conv2d_scale(1.f / 32.f);
    vae.set_flash_attention_enabled(true);
    std::map<std::string, ggml_tensor *> weights;
    vae.get_param_tensors(weights);
    if (!manager->register_param_tensors("VAE", weights, ModelManager::ResidencyMode::ParamBackend,
                                        cpu.get(), cpu.get()) || !manager->validate_registered_tensors()) return 4;
    sd::Tensor<float> latents({width / 8, height / 8, 4, 1});
    std::ifstream input(argv[2], std::ios::binary);
    input.read(reinterpret_cast<char *>(latents.data()), latents.numel() * sizeof(float));
    if (!input) return 5;
    sd_img_gen_params_t parameters;
    sd_img_gen_params_init(&parameters);
    auto tiling = parameters.vae_tiling_params;
    tiling.enabled = tile > 0;
    tiling.tile_size_x = tiling.tile_size_y = tile;
    const auto started = std::chrono::steady_clock::now();
    auto pixels = vae.decode(4, vae.diffusion_to_vae_latents(latents), tiling, false, false, false, true);
    if (pixels.numel() != width * height * 3) return 6;
    for (float value : pixels.values()) if (!std::isfinite(value)) return 7;
    std::ofstream output(argv[6], std::ios::binary);
    output.write(reinterpret_cast<const char *>(pixels.data()), pixels.numel() * sizeof(float));
    std::cout << "tile=" << tile << " seconds="
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() << '\n';
    return output ? 0 : 8;
}
