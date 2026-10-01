#include "model/diffusion/freeu.hpp"
#include <ggml-cpu.h>
#include <ggml-alloc.h>
#if defined(__APPLE__)
#include <ggml-metal.h>
#endif
#include <complex>
#include <iostream>
#include <memory>
#include <stdexcept>

void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
// Independent full DFT -> shifted 2x2 mask -> inverse DFT oracle. Production
// evaluates only the changed four frequencies; this oracle evaluates every bin.
std::vector<float> reference(const std::vector<float> &input, int w, int h, float scale) {
    constexpr double tau = 6.28318530717958647692;
    const int size = w * h;
    std::vector<std::complex<double>> frequencies(size);
    for (int ky = 0; ky < h; ++ky) for (int kx = 0; kx < w; ++kx) {
        auto &value = frequencies[ky * w + kx];
        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x)
            value += double(input[y * w + x]) * std::polar(1., -tau * (double(kx * x) / w + double(ky * y) / h));
        const int shiftedX = (kx + w / 2) % w, shiftedY = (ky + h / 2) % h;
        const int startX = w == 1 ? 0 : w / 2 - 1, startY = h == 1 ? 0 : h / 2 - 1;
        if (shiftedX >= startX && shiftedX < w / 2 + 1 && shiftedY >= startY && shiftedY < h / 2 + 1)
            value *= scale;
    }
    std::vector<float> result(size);
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
        std::complex<double> value{};
        for (int ky = 0; ky < h; ++ky) for (int kx = 0; kx < w; ++kx)
            value += frequencies[ky * w + kx] * std::polar(1., tau * (double(kx * x) / w + double(ky * y) / h));
        result[y * w + x] = float(value.real() / size);
    }
    return result;
}
int main() {
    try {
        const auto sd15 = sd::freeu::settings(true, sd::freeu::Profile::SD15);
        const auto sd2 = sd::freeu::settings(true, sd::freeu::Profile::SD2);
        const auto sdxl = sd::freeu::settings(true, sd::freeu::Profile::SDXL);
        require(sd15.backbone == std::array<float, 2>{1.2f, 1.4f} && sd15.skip == std::array<float, 2>{.9f, .2f}, "SD 1.5 profile changed");
        require(sd2.backbone == std::array<float, 2>{1.1f, 1.2f} && sd2.skip == sd15.skip, "SD 2 profile changed");
        require(sdxl.backbone == sd2.backbone && sdxl.skip == std::array<float, 2>{.6f, .4f}, "SDXL profile changed");
        for (const auto &[w, h] : std::vector<std::pair<int, int>>{{1, 1}, {1, 7}, {5, 7}, {8, 8}, {8, 4}}) {
            std::vector<float> input(w * h), output(w * h);
            for (std::size_t i = 0; i < input.size(); ++i) input[i] = float(std::sin(i * .71) + .3 * std::cos(i * 1.7));
            const sd::freeu::Basis basis(w, h);
            for (float scale : {.2f, .4f, .6f, .9f, 1.f}) {
                sd::freeu::filter_plane(input.data(), output.data(), basis, scale);
                const auto expected = reference(input, w, h, scale);
                for (std::size_t i = 0; i < input.size(); ++i)
                    require(std::abs(expected[i] - output[i]) < 2e-5f, "Native FreeU differs from the full Fourier oracle");
            }
        }
        auto backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(ggml_backend_cpu_init(), &ggml_backend_free);
        require(bool(backend), "Missing CPU backend");
        ggml_backend_cpu_set_n_threads(backend.get(), 3);
        auto metal = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(nullptr, &ggml_backend_free);
        std::vector<bool> placements{false};
#if defined(__APPLE__)
        metal.reset(ggml_backend_metal_init());
        require(bool(metal), "Metal backend unavailable for the mixed-backend FreeU contract");
        placements.push_back(true);
#endif
        for (bool mixed : placements) for (int stage : {0, 1, 2}) {
            ggml_init_params params{1024 * 1024, nullptr, true};
            auto context = std::unique_ptr<ggml_context, decltype(&ggml_free)>(ggml_init(params), &ggml_free);
            auto *ctx = context.get();
            auto *hidden = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 5, 7, 5, 2);
            auto *skip = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 5, 7, 3, 2);
            auto *originalHidden = hidden, *originalSkip = skip;
            ggml_set_input(hidden); ggml_set_input(skip);
            // Inputs are normally recyclable scratch after their last use. Pin
            // them as observable outputs when testing non-mutation explicitly.
            ggml_set_output(hidden); ggml_set_output(skip);
            const auto disabled = sd::freeu::settings(false, sd::freeu::Profile::SD15);
            sd::freeu::apply(ctx, hidden, skip, stage, disabled);
            require(hidden == originalHidden && skip == originalSkip, "Disabled FreeU must not add graph nodes");
            const auto config = sd::freeu::settings(true, sd::freeu::Profile::SDXL);
            sd::freeu::apply(ctx, hidden, skip, stage, config);
            require(stage < 2 || (hidden == originalHidden && skip == originalSkip), "FreeU changed a later decoder stage");
            auto *output = ggml_concat(ctx, hidden, skip, 2);
            ggml_set_output(output);
            auto *graph = ggml_new_graph_custom(ctx, 128, false);
            ggml_build_forward_expand(graph, output);
            auto buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>(nullptr, &ggml_backend_buffer_free);
            auto scheduler = std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)>(nullptr, &ggml_backend_sched_free);
            if (mixed) {
                ggml_backend_t backends[]{metal.get(), backend.get()};
                scheduler.reset(ggml_backend_sched_new(backends, nullptr, 2, 128, false, true));
                require(bool(scheduler), "Mixed-backend scheduler allocation failed");
                ggml_backend_sched_set_tensor_backend(scheduler.get(), output, metal.get());
                require(ggml_backend_sched_alloc_graph(scheduler.get(), graph), "Mixed FreeU graph allocation failed");
                require(ggml_backend_sched_get_tensor_backend(scheduler.get(), output) == metal.get(), "Output was not scheduled on Metal");
                if (stage < 2)
                    require(ggml_backend_sched_get_tensor_backend(scheduler.get(), skip) == backend.get(), "Fourier kernel was not scheduled on CPU");
            } else {
                buffer.reset(ggml_backend_alloc_ctx_tensors(ctx, backend.get()));
                require(bool(buffer), "FreeU graph allocation failed");
            }
            std::vector<float> hiddenValues(5 * 7 * 5 * 2), skipValues(5 * 7 * 3 * 2);
            for (std::size_t i = 0; i < hiddenValues.size(); ++i) hiddenValues[i] = float(i % 17 - 8.0) / 10.f;
            for (std::size_t i = 0; i < skipValues.size(); ++i) skipValues[i] = float(std::sin(i * .13));
            ggml_backend_tensor_set(originalHidden, hiddenValues.data(), 0, hiddenValues.size() * sizeof(float));
            ggml_backend_tensor_set(originalSkip, skipValues.data(), 0, skipValues.size() * sizeof(float));
            const auto status = mixed ? ggml_backend_sched_graph_compute(scheduler.get(), graph)
                : ggml_backend_graph_compute(backend.get(), graph);
            require(status == GGML_STATUS_SUCCESS, "FreeU native graph failed");
            std::vector<float> actual(5 * 7 * 8 * 2);
            ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
            std::vector<float> unchanged(hiddenValues.size());
            ggml_backend_tensor_get(originalHidden, unchanged.data(), 0, unchanged.size() * sizeof(float));
            require(unchanged == hiddenValues, "FreeU mutated caller-owned backbone input");
            for (int batch = 0; batch < 2; ++batch) {
                for (int c = 0; c < 5; ++c) for (int p = 0; p < 35; ++p) {
                    const float factor = stage < 2 && c < 2 ? config.backbone[stage] : 1.f;
                    require(std::abs(actual[(batch * 8 + c) * 35 + p] - hiddenValues[(batch * 5 + c) * 35 + p] * factor) < 1e-6f,
                        "Backbone scaling changed the wrong channels or batch");
                }
                for (int c = 0; c < 3; ++c) {
                    const auto begin = skipValues.begin() + (batch * 3 + c) * 35;
                    const std::vector<float> plane(begin, begin + 35);
                    const auto expected = stage < 2 ? reference(plane, 5, 7, config.skip[stage]) : plane;
                    for (int p = 0; p < 35; ++p)
                        require(std::abs(actual[(batch * 8 + 5 + c) * 35 + p] - expected[p]) < 2e-5f,
                            "Native graph skip filtering differs from Fourier reference");
                }
            }
        }
        std::cout << "Native FreeU Fourier and multi-threaded graph contracts passed\n";
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
