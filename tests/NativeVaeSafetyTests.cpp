#include <model/vae/auto_encoder_kl.hpp>
#include <ggml-cpu.h>
#include <limits>
#include <iostream>

// Exercise the real VAE normalization boundary without expensive inference.
struct ProbeDecoder : AutoEncoderKL {
    float value = 0;
    int calls = 0;
    explicit ProbeDecoder(ggml_backend_t cpu) : AutoEncoderKL(cpu, {}, "first_stage_model", true) {}
    sd::Tensor<float> _compute(int, const sd::Tensor<float>&, bool) override {
        ++calls;
        sd::Tensor<float> output({64, 64, 3, 1});
        output.fill_(value);
        return output;
    }
};

// Exercise the actual spatial tile selection without loading a trained model.
struct ProbeEncoder : AutoEncoderKL {
    int maxWidth = 0, maxHeight = 0, calls = 0;
    ProbeEncoder(ggml_backend_t cpu, SDVersion family)
        : AutoEncoderKL(cpu, {}, "first_stage_model", false, false, family) { scale_input = false; }
    sd::Tensor<float> _compute(int, const sd::Tensor<float>& input, bool decode) override {
        if (decode) return {};
        ++calls;
        maxWidth = std::max(maxWidth, static_cast<int>(input.shape()[0]));
        maxHeight = std::max(maxHeight, static_cast<int>(input.shape()[1]));
        sd::Tensor<float> output({input.shape()[0] / 8, input.shape()[1] / 8, 4, 1});
        output.fill_(0);
        return output;
    }
};

int main() {
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(), ggml_backend_free);
    ProbeDecoder decoder(cpu.get());
    sd::Tensor<float> latents({8, 8, 4, 1});
    sd_img_gen_params_t params;
    sd_img_gen_params_init(&params);
    for (const bool tiled : {false, true}) {
        params.vae_tiling_params.enabled = tiled;
        latents[0] = std::numeric_limits<float>::quiet_NaN();
        const int before = decoder.calls;
        if (!decoder.decode(1, latents, params.vae_tiling_params).empty() || decoder.calls != before) return 3;
        latents[0] = 0;
        for (const float invalid : {std::numeric_limits<float>::quiet_NaN(),
                                   std::numeric_limits<float>::infinity(),
                                   -std::numeric_limits<float>::infinity()}) {
            decoder.value = invalid;
            if (!decoder.decode(1, latents, params.vae_tiling_params, false, false, false, true).empty()) return 1;
        }
        // Legitimate black, white and uniform images must not be rejected.
        for (const float valid : {-2.f, 0.f, 2.f}) {
            decoder.value = valid;
            const auto output = decoder.decode(1, latents, params.vae_tiling_params, false, false, false, true);
            if (output.numel() != 64 * 64 * 3 || !std::isfinite(output[0])) return 2;
        }
    }
    sd::Tensor<float> image({256, 384, 3, 1}); image.fill_(0);
    sd_tiling_params_t tiles{}; tiles.enabled = true; tiles.tile_size_x = tiles.tile_size_y = 16;
    tiles.target_overlap = 0.5f;
    ProbeEncoder legacy(cpu.get(), VERSION_SDXL);
    const auto original = legacy.encode(1, image, tiles);
    if (original.empty() || legacy.maxWidth != 256 || legacy.maxHeight != 256) return 4;
    tiles.extra_tiling_args = "iild_encode_tile_factor=1";
    ProbeEncoder bounded(cpu.get(), VERSION_SDXL);
    const auto encoded = bounded.encode(1, image, tiles);
    if (encoded.empty() || encoded.shape() != original.shape() || bounded.maxWidth > 128 || bounded.maxHeight > 128 ||
        bounded.calls <= legacy.calls) return 5;
    ProbeEncoder other(cpu.get(), VERSION_SD1);
    if (other.encode(1, image, tiles).empty() || other.maxWidth != 256 || other.maxHeight != 256) return 6;
    // The 1024x1368 request is padded to 1024x1408 for Hires. Verify that
    // the real tiler covers the complete aligned image with bounded tiles;
    // the probe deliberately does not load or run trained VAE weights.
    sd::Tensor<float> alignedHires({1024, 1408, 3, 1}); alignedHires.fill_(0);
    tiles.tile_size_x = tiles.tile_size_y = 64;
    ProbeEncoder hires(cpu.get(), VERSION_SDXL);
    const auto hiresEncoded = hires.encode(1, alignedHires, tiles);
    if (hiresEncoded.empty() || hiresEncoded.shape()[0] != 128 || hiresEncoded.shape()[1] != 176 ||
        hires.maxWidth > 512 || hires.maxHeight > 512 || hires.calls <= 1) return 7;
    std::cout << "Nonfinite VAE pixels rejected before clamping on full/tiled paths\n";
}
