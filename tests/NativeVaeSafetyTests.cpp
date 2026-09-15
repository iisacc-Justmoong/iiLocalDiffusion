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
    std::cout << "Nonfinite VAE pixels rejected before clamping on full/tiled paths\n";
}
