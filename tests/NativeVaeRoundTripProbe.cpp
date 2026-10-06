// Opt-in actual Krea2/Qwen VAE encode/decode isolation. No text encoder or denoiser.
#include <model_manager.h>
#include <model/vae/wan_vae.hpp>
#include <ggml-cpu.h>
#include <fstream>
#include <iostream>
#include <cmath>

int main(int argc, char **argv) {
    if (argc != 6) return 1;
    const int width = std::stoi(argv[3]), height = std::stoi(argv[4]);
    if (width < 64 || height < 64 || width % 8 || height % 8) return 1;
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(), ggml_backend_free);
    auto manager = std::make_shared<ModelManager>();
    manager->set_n_threads(4); manager->set_enable_mmap(true);
    if (!manager->loader().init_from_file(argv[1])) return 2;
    manager->loader().convert_tensors_name();
    const auto version = manager->loader().get_sd_version();
    if (!sd_version_is_krea2(version) && !sd_version_is_qwen_image(version)) return 3;
    WAN::WanVAERunner vae(cpu.get(), manager->loader().get_tensor_storage_map(),
        "first_stage_model", false, version, manager);
    struct Release { ModelManager &manager; ~Release() { manager.unregister_param_tensors("VAE"); } } release{*manager};
    std::map<std::string, ggml_tensor *> weights; vae.get_param_tensors(weights);
    if (!manager->register_param_tensors("VAE", weights, ModelManager::ResidencyMode::ParamBackend,
        cpu.get(), cpu.get()) || !manager->validate_registered_tensors()) return 4;
    std::vector<unsigned char> rgb(std::size_t(width)*height*3);
    std::ifstream input(argv[2], std::ios::binary); input.read(reinterpret_cast<char *>(rgb.data()), rgb.size());
    if (!input) return 5;
    sd::Tensor<float> pixels({width, height, 3, 1});
    for (int c=0;c<3;++c) for (int y=0;y<height;++y) for (int x=0;x<width;++x)
        pixels[c*width*height+y*width+x] = rgb[(y*width+x)*3+c]/255.f;
    sd_img_gen_params_t params; sd_img_gen_params_init(&params); params.vae_tiling_params.enabled=false;
    auto encoded = vae.encode(4, pixels, params.vae_tiling_params);
    if (encoded.empty()) return 6;
    auto normalized = vae.vae_to_diffusion_latents(vae.vae_output_to_latents(encoded, nullptr));
    for (float value : normalized.values()) if (!std::isfinite(value)) return 7;
    auto decoded = vae.decode(4, vae.diffusion_to_vae_latents(normalized), params.vae_tiling_params);
    if (decoded.numel()!=pixels.numel()) return 8;
    double squared=0; std::vector<unsigned char> output(rgb.size());
    for (int c=0;c<3;++c) for (int y=0;y<height;++y) for (int x=0;x<width;++x) {
        const auto index=c*width*height+y*width+x; const auto value=decoded[index];
        if (!std::isfinite(value)) return 9;
        squared += (value-pixels[index])*(value-pixels[index]);
        output[(y*width+x)*3+c]=std::lround(std::clamp(value,0.f,1.f)*255);
    }
    std::ofstream file(argv[5],std::ios::binary); file.write(reinterpret_cast<const char *>(output.data()),output.size());
    std::cout << "roundtrip_rmse=" << std::sqrt(squared/pixels.numel()) << '\n';
    return file ? 0 : 10;
}
