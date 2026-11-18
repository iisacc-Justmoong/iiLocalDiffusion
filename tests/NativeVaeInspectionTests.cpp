#include "Generation/NativeVaeInspection.hpp"
#include "Generation/NativeTextEncoderInspection.hpp"
#include "Generation/ImagePackageBridge.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
void modelFixture(const std::filesystem::path &path) {
    const std::string json = R"({"model.diffusion_model.input_blocks.0.0.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"model.diffusion_model.middle_block.1.proj_in.weight":{"dtype":"F32","shape":[1],"data_offsets":[4,8]},"conditioner.embedders.1.model.positional_embedding":{"dtype":"F32","shape":[1],"data_offsets":[8,12]}})";
    std::ofstream output(path, std::ios::binary);
    const std::uint64_t size = json.size();
    for (int i = 0; i < 8; ++i) output.put(static_cast<char>(size >> (8 * i)));
    output << json;
    for (int i = 0; i < 12; ++i) output.put(0);
}
void oneTensorModel(const std::filesystem::path &path, const std::string &key) {
    const auto json = "{\"" + key + "\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
    std::ofstream output(path, std::ios::binary);
    const std::uint64_t size = json.size();
    for (int i = 0; i < 8; ++i) output.put(static_cast<char>(size >> (8 * i)));
    output << json;
    for (int i = 0; i < 4; ++i) output.put(0);
}
void classicModel(const std::filesystem::path &path, int channels) {
    const auto bytes = channels * 4;
    const auto json = "{\"cond_stage_model.transformer.text_model.embeddings.token_embedding.weight\":{\"dtype\":\"F32\",\"shape\":[1,"
        + std::to_string(channels) + "],\"data_offsets\":[0," + std::to_string(bytes)
        + "]},\"model.diffusion_model.middle_block.1.proj_in.weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":["
        + std::to_string(bytes) + ',' + std::to_string(bytes + 4) + "]}}";
    std::ofstream output(path, std::ios::binary);
    const std::uint64_t size = json.size();
    for (int i = 0; i < 8; ++i) output.put(static_cast<char>(size >> (8 * i)));
    output << json;
    for (int i = 0; i < bytes + 4; ++i) output.put(0);
}
}

int main(int argc, char **argv) {
    try {
        require(argc == 3, "Expected a test directory and VAE resources");
        const auto directory = std::filesystem::path(argv[1]);
        std::filesystem::create_directories(directory);
        const auto model = directory / "model.safetensors";
        modelFixture(model);
        const auto correct = std::filesystem::path(argv[2]) / "sdxl/diffusion_pytorch_model.safetensors";
        const auto wrong = std::filesystem::path(argv[2]) / "flux1/diffusion_pytorch_model.safetensors";
        const auto info = iiLocalDiffusion::inspectNativeVae(model);
        require(info.state == iiLocalDiffusion::NativeVaeState::Missing && info.vaeFamily == "sdxl-base",
                "The C++ API must report the model's missing VAE contract");
        require(iiLocalDiffusion::validateNativeVae(model, correct), "The compatible VAE was rejected");
        require(!iiLocalDiffusion::validateNativeVae(model, wrong), "An incompatible VAE was accepted");
        auto *result = iild_image_vae_inspect_v1(model.string().c_str());
        require(result != nullptr, "The C API did not return a report");
        const std::string report = iild_image_package_json_v1(result);
        iild_image_package_free_v1(result);
        require(report.find("\"state\":\"missing\"") != std::string::npos, "The C report omitted the VAE state");
        result = iild_image_vae_validate_v1(model.string().c_str(), correct.string().c_str());
        require(result != nullptr, "The C API did not return a validation report");
        const std::string validation = iild_image_package_json_v1(result);
        iild_image_package_free_v1(result);
        require(validation.find("\"valid\":true") != std::string::npos, "The C report omitted validation");
        bool missingRejected = false;
        try { iiLocalDiffusion::inspectNativeVae(directory / "nonexistent"); }
        catch (const std::invalid_argument &) { missingRejected = true; }
        require(missingRejected, "Missing model paths must be rejected");
        const auto flux = directory / "flux.safetensors";
        oneTensorModel(flux, "model.diffusion_model.double_blocks.0.img_attn.qkv.weight");
        const auto encoders = std::filesystem::path(argv[2]).parent_path() / "text_encoders";
        const auto clip = encoders / "flux1_clip_l.safetensors";
        const auto t5 = encoders / "flux1_t5xxl_fp8_scaled.safetensors";
        if (!std::filesystem::is_regular_file(clip) || !std::filesystem::is_regular_file(t5)) {
            std::cout << "Native VAE APIs verified; optional text encoder metadata fixtures unavailable\n";
            return 0;
        }
        const auto sd1 = directory / "sd1.safetensors";
        const auto sd2 = directory / "sd2.safetensors";
        const auto sd3 = directory / "sd3.safetensors";
        classicModel(sd1, 768);
        classicModel(sd2, 1024);
        oneTensorModel(sd3, "model.diffusion_model.joint_blocks.0.weight");
        require(iiLocalDiffusion::validateNativeTextEncoder(sd1, "clip_l", clip), "SD1 rejected a compatible CLIP-L");
        require(!iiLocalDiffusion::validateNativeTextEncoder(sd2, "clip_l", clip), "SD2 accepted a 768-wide CLIP-L");
        require(iiLocalDiffusion::validateNativeTextEncoder(sd3, "t5xxl", t5), "SD3 rejected a compatible T5-XXL");
        require(iiLocalDiffusion::inspectNativeTextEncoder(model, "clip_l") == iiLocalDiffusion::NativeTextEncoderState::Missing,
                "SDXL must identify its missing CLIP-L");
        require(iiLocalDiffusion::inspectNativeTextEncoder(model, "clip_g") == iiLocalDiffusion::NativeTextEncoderState::Incompatible,
                "An incomplete embedded CLIP-G must not be treated as complete");
        require(iiLocalDiffusion::validateNativeTextEncoder(model, "clip_l", clip), "The SDXL CLIP-L contract rejected its encoder");
        require(!iiLocalDiffusion::validateNativeTextEncoder(model, "clip_g", clip), "CLIP-L was accepted as SDXL CLIP-G");
        require(iiLocalDiffusion::inspectNativeTextEncoder(flux, "clip_l") == iiLocalDiffusion::NativeTextEncoderState::Missing,
                "The C++ API must identify a missing CLIP-L");
        require(iiLocalDiffusion::validateNativeTextEncoder(flux, "clip_l", clip), "The compatible CLIP-L was rejected");
        require(!iiLocalDiffusion::validateNativeTextEncoder(flux, "clip_l", t5), "T5 was accepted as CLIP-L");
        require(iiLocalDiffusion::validateNativeTextEncoder(flux, "t5xxl", t5), "The compatible scaled-FP8 T5 was rejected");
        result = iild_image_text_encoder_inspect_v1(flux.string().c_str(), "clip_l");
        require(result != nullptr, "The C API did not return a text encoder report");
        const std::string textReport = iild_image_package_json_v1(result);
        iild_image_package_free_v1(result);
        require(textReport.find("\"state\":\"missing\"") != std::string::npos, "The C report omitted the encoder state");
        std::cout << "Native VAE inspection and compatibility APIs verified using metadata only\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
