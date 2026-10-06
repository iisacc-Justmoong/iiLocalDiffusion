// Compile the production orchestration so test runners can replace only the
// pretrained network compute. Sampler state, conditioning selection and masks
// are production code, not a reimplementation in this fixture.
#include "stable-diffusion.cpp"
#include <iostream>
#include <stdexcept>
#include <ggml-cpu.h>

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

struct ProbeRunner : DiffusionModelRunner {
    bool refiner = false, fail = false;
    int calls = 0;
    std::vector<float> prompts;
    std::function<void()> after_compute;
    bool expect_multi_control = false;
    bool expect_ip = false;
    bool expect_indexed_ip = false;
    float expected_ip_unconditional = 10.f;
    ProbeRunner(ggml_backend_t backend, bool refining)
        : DiffusionModelRunner(backend, "test"), refiner(refining) {}
    std::string get_desc() override { return "refiner_probe"; }
    void get_param_tensors(std::map<std::string, ggml_tensor*>&, const std::string&) override {}
    sd::Tensor<float> compute(int, const DiffusionParams& params) override {
        require(runner_start(), "Probe runner could not start");
        ++calls;
        require(params.context && params.y, "Conditioning missing from production sampler");
        const auto marker = params.context->values()[0];
        require(params.y->values()[0] == (refiner ? 20.f : 10.f), "Wrong model ADM selected");
        require(std::abs(marker) == (refiner ? 2.f : 1.f), "Wrong model prompt selected");
        prompts.push_back(marker);
        const auto* extra = std::get_if<UNetDiffusionExtra>(&params.extra);
        require(extra && (!refiner || (!extra->ip_context && !extra->ip_adapters && (!extra->controls || extra->controls->empty()))),
            "Base IP/control context leaked into Refiner");
        if (expect_ip && !refiner)
            require(extra->ip_context && extra->ip_scale == .7f
                && extra->ip_context->values().front() == (marker < 0 ? expected_ip_unconditional : 12.f),
                "IP conditional/unconditional tokens or strength were lost before UNet");
        if (expect_indexed_ip && !refiner) {
            require(!extra->ip_context && extra->ip_adapters && extra->ip_adapters->size() == 2,
                "Indexed IP inputs were not forwarded independently");
            const auto& a = extra->ip_adapters->at(0); const auto& b = extra->ip_adapters->at(1);
            require(a.slot == 1 && b.slot == 0 && a.scale == .3f && b.scale == .7f
                && a.tokens->values()[0] == (marker < 0 ? 17.f : 12.f)
                && b.tokens->values()[0] == (marker < 0 ? 10.f : 12.f), "Wrong indexed IP conditional branch");
            const int w = int(params.x->shape()[0]), h = int(params.x->shape()[1]);
            const auto* coverage = a.coverage ? a.coverage(w, h) : nullptr;
            require(coverage && coverage->size() == size_t(w * h) && coverage->front() == 0
                && coverage->back() == 1 && !b.coverage, "Indexed mask omitted Base/Hires spatial shape");
        }
        if (expect_multi_control && !refiner)
            require(extra->controls && extra->controls->size() == 1 && extra->control_strength == 1.f
                && extra->controls->front().values()[0] == 7.f,
                "Multi-ControlNet residuals were lost or weighted twice before UNet");
        if (fail || sd_should_abort()) return {};
        auto out = *params.x * .03f;
        for (auto& value : out.values()) value += (refiner ? .2f : .1f) + marker * .01f;
        if (after_compute) after_compute();
        return out;
    }
};

struct ProbeConditioner : Conditioner {
    bool refiner;
    int calls = 0;
    std::vector<std::pair<int, int>> sizes;
    std::vector<std::string> texts;
    explicit ProbeConditioner(bool refining) : refiner(refining) {}
    SDCondition get_learned_condition(int, const ConditionerParams& params) override {
        ++calls;
        sizes.emplace_back(params.width, params.height);
        texts.push_back(params.text);
        SDCondition result;
        result.c_crossattn = sd::Tensor<float>::from_vector({(refiner ? 2.f : 1.f) * (params.is_negative_prompt ? -1.f : 1.f)});
        result.c_vector = sd::Tensor<float>::from_vector({refiner ? 20.f : 10.f});
        return result;
    }
    void get_param_tensors(std::map<std::string, ggml_tensor*>&) override {}
    void set_flash_attention_enabled(bool) override {}
};

struct ProbeVAE : FakeVAE {
    int encodes = 0, decodes = 0;
    explicit ProbeVAE(ggml_backend_t backend) : FakeVAE(VERSION_SDXL, backend) {}
    sd::Tensor<float> encode(int, const sd::Tensor<float>& input, sd_tiling_params_t, bool, bool) override {
        ++encodes;
        sd::Tensor<float> output({input.shape()[0] / 8, input.shape()[1] / 8, 4, 1});
        output.fill_(.1f);
        return output;
    }
    sd::Tensor<float> decode(int, const sd::Tensor<float>& input, sd_tiling_params_t, bool, bool, bool, bool) override {
        ++decodes;
        sd::Tensor<float> output({input.shape()[0] * 8, input.shape()[1] * 8, 3, 1});
        output.fill_(.5f);
        return output;
    }
};

SDCondition condition(float marker, float adm) {
    SDCondition value;
    value.c_crossattn = sd::Tensor<float>::from_vector({marker});
    value.c_vector = sd::Tensor<float>::from_vector({adm});
    return value;
}

void initialize(StableDiffusionGGML& engine, ggml_backend_t backend, bool refiner) {
    engine.version = refiner ? VERSION_SDXL_REFINER : VERSION_SDXL;
    engine.n_threads = 1;
    engine.diffusion_model = std::make_shared<ProbeRunner>(backend, refiner);
    engine.cond_stage_model = std::make_shared<ProbeConditioner>(refiner);
    engine.sampler_rng = std::make_shared<PhiloxRNG>();
    engine.refresh_compvis_denoiser_sigmas();
}

void sampling(ggml_backend_t backend) {
    StableDiffusionGGML base, refiner;
    initialize(base, backend, false); initialize(refiner, backend, true);
    auto base_runner = std::static_pointer_cast<ProbeRunner>(base.diffusion_model);
    auto refiner_runner = std::static_pointer_cast<ProbeRunner>(refiner.diffusion_model);
    const auto positive = condition(1, 10), negative = condition(-1, 10);
    StableDiffusionGGML::RefinerPass pass{&refiner, condition(2, 20), condition(-2, 20), .5f};
    sd::Tensor<float> initial({2, 2, 4, 1}), noise({2, 2, 4, 1}), mask({2, 2, 4, 1});
    for (size_t i = 0; i < initial.values().size(); ++i) {
        initial.values()[i] = .1f + float(i) * .01f;
        noise.values()[i] = .2f - float(i) * .005f;
        mask.values()[i] = i % 2 ? 1.f : 0.f;
    }
    const std::vector<float> sigmas{2.f, 1.5f, 1.f, .5f, 0.f};
    sd_guidance_params_t guidance{};
    guidance.txt_cfg = 2.f; guidance.img_cfg = 1.f;
    const std::vector<sample_method_t> methods{EULER_SAMPLE_METHOD, EULER_A_SAMPLE_METHOD,
        HEUN_SAMPLE_METHOD, DPMPP2M_SAMPLE_METHOD, DPMPP2M_SDE_SAMPLE_METHOD, DDIM_TRAILING_SAMPLE_METHOD};
    for (bool v_prediction : {false, true}) for (auto method : methods) for (float at : {0.f, .5f, 1.f}) for (bool masked : {false, true}) {
        if (v_prediction) base.denoiser = std::make_shared<CompVisVDenoiser>();
        else base.denoiser = std::make_shared<CompVisDenoiser>();
        base.refresh_compvis_denoiser_sigmas();
        pass.switch_at = at;
        base_runner->calls = refiner_runner->calls = 0;
        base_runner->prompts.clear(); refiner_runner->prompts.clear();
        base.sampler_rng->manual_seed(42);
        const auto output = base.sample(base.diffusion_model, true, initial, noise, positive, negative, {}, {},
            0.f, guidance, 0.f, 0, method, false, nullptr, sigmas, {}, {}, masked ? mask : sd::Tensor<float>{},
            {}, 1.f, 0, 1.f, nullptr, false, {}, &pass);
        require(!output.empty(), "Production Refiner sampling failed");
        require(!base_runner->runner_started() && !refiner_runner->runner_started(), "Successful sampler leaked runner workspace");
        auto oracle_rng = std::make_shared<PhiloxRNG>(); oracle_rng->manual_seed(42);
        int oracle_base = 0, oracle_refiner = 0;
        // Independent denoiser math and model-selection decision, inside ONE
        // unmodified production solver call. A split solver loses DPM++ history
        // and will not match this oracle even with the same boundary latent.
        auto oracle = [&](const sd::Tensor<float>& x, float sigma, int step) {
            const bool refining = std::abs(step) > (at == 0 ? 0 : at == 1 ? 4 : 2);
            (refining ? oracle_refiner : oracle_base) += 2; // Positive + negative CFG calls.
            auto epsilon = x * (.03f / std::sqrt(1.f + sigma * sigma));
            const float prompt = refining ? 2.f : 1.f;
            for (auto& value : epsilon.values()) value += (refining ? .2f : .1f) + .03f * prompt;
            sd::guidance::GuiderOutput result;
            result.pred = !refining && v_prediction
                ? x * (1.f / (1.f + sigma * sigma)) - epsilon * (sigma / std::sqrt(1.f + sigma * sigma))
                : x - epsilon * sigma;
            if (masked) result.pred = result.pred * mask + initial * (1.f - mask);
            return result;
        };
        const auto expected = sample_k_diffusion(method, oracle, initial + noise * sigmas[0], sigmas,
            oracle_rng, 0.f, false, nullptr, base.denoiser);
        require(output.shape() == expected.shape(), "Refiner changed latent shape");
        for (size_t i = 0; i < output.values().size(); ++i) {
            require(std::isfinite(output.values()[i]) && std::abs(output.values()[i] - expected.values()[i]) < 2e-5f,
                "Refiner lost continuous solver/noise state");
            if (masked && mask.values()[i] == 0)
                require(std::abs(output.values()[i] - initial.values()[i]) < 2e-5f, "Masked initial latent changed");
        }
        require(base_runner->calls == oracle_base && refiner_runner->calls == oracle_refiner,
            "Model switched inside a predictor/corrector pair or at the wrong boundary");
        require(base_runner->calls == int(base_runner->prompts.size())
            && refiner_runner->calls == int(refiner_runner->prompts.size()), "Lost condition calls");
    }
    pass.switch_at = .7f;
    require(!pass.active(-7, 10) && pass.active(8, 10), "Float switch boundary drift");
    pass.switch_at = .8f;
    require(!pass.active(120, 150) && pass.active(-121, 150), "Long decimal switch boundary drift");
    pass.switch_at = .5f;
    require(!pass.active(-2, 5) && !pass.active(3, 5) && pass.active(-4, 5), "Fractional step rounding");
    refiner_runner->fail = true;
    const auto failure = base.sample(base.diffusion_model, true, initial, noise, positive, negative, {}, {},
        0.f, guidance, 0.f, 0, EULER_SAMPLE_METHOD, false, nullptr, sigmas, {}, {}, {}, {}, 1.f, 0, 1.f,
        nullptr, false, {}, &pass);
    require(failure.empty(), "Refiner compute failure fell back to Base output");
    refiner_runner->fail = false;
    refiner.set_cancel_flag(SD_CANCEL_ALL);
    const int before = base_runner->calls + refiner_runner->calls;
    const auto cancelled = base.sample(base.diffusion_model, true, initial, noise, positive, negative, {}, {},
        0.f, guidance, 0.f, 0, EULER_SAMPLE_METHOD, false, nullptr, sigmas, {}, {}, {}, {}, 1.f, 0, 1.f,
        nullptr, false, {}, &pass);
    require(cancelled.empty() && before == base_runner->calls + refiner_runner->calls, "Refiner cancellation ignored");
    require(!base_runner->runner_started() && !refiner_runner->runner_started(), "Runner remained active after cancellation");
}

void api(ggml_backend_t backend) {
    StableDiffusionGGML base, refiner;
    initialize(base, backend, false); initialize(refiner, backend, true);
    sd_ctx_t base_ctx{&base}, refiner_ctx{&refiner};
    uint8_t coverage = 128;
    require(!sd_set_control_net_mask(nullptr, {}) && sd_set_control_net_mask(&base_ctx, {1, 1, 1, &coverage}),
        "Regional mask C API validation failed");
    coverage = 0;
    std::vector<sd::Tensor<float>> residuals{sd::Tensor<float>({1, 1, 1, 1}, {255.f})};
    require(base.control_region_mask.apply(residuals) && residuals[0].values()[0] == 128.f,
        "Regional mask C API did not retain an owned copy");
    require(sd_set_control_net_mask(&base_ctx, {}), "Regional mask clear failed");
    residuals[0].values()[0] = 255;
    require(base.control_region_mask.apply(residuals) && residuals[0].values()[0] == 255,
        "Cleared regional mask affected later generation");
    require(std::string(sd_get_model_family(&refiner_ctx)) == "sdxl-refiner", "Refiner family");
    require(std::string(model_version_to_str[VERSION_SDXL_REFINER]) == "SDXL Refiner", "Refiner logging name");
    const auto vae = sd_vae_contract(VERSION_SDXL_REFINER);
    require(std::string(vae.model_family) == "sdxl-refiner" && std::string(vae.vae_family) == "sdxl-base",
        "Refiner must retain shared SDXL VAE compatibility");
    require(sd_ctx_can_refine(&base_ctx, &refiner_ctx), "Compatible SDXL contexts rejected");
    require(!sd_ctx_can_refine(&base_ctx, &base_ctx) && !sd_ctx_can_refine(nullptr, &refiner_ctx), "Invalid contexts accepted");
    for (auto invalid : {VERSION_SD1, VERSION_SDXL_REFINER, VERSION_SDXL_INPAINT, VERSION_SDXL_PIX2PIX}) {
        base.version = invalid;
        require(!sd_ctx_can_refine(&base_ctx, &refiner_ctx), "Incompatible base accepted");
    }
    base.version = VERSION_SDXL;
    base.denoiser = std::make_shared<EDMVDenoiser>();
    require(!sd_ctx_can_refine(&base_ctx, &refiner_ctx), "EDM latent contract accepted");
    base.denoiser = std::make_shared<CompVisVDenoiser>();
    require(sd_ctx_can_refine(&base_ctx, &refiner_ctx), "SDXL v-prediction base rejected");
    sd_img_gen_params_t params; sd_img_gen_params_init(&params);
    for (float invalid : {-1.f, 2.f, std::numeric_limits<float>::quiet_NaN()}) {
        sd_image_t* images = reinterpret_cast<sd_image_t*>(1); int count = 7;
        require(!generate_image_with_refiner(&base_ctx, &refiner_ctx, invalid, &params, &images, &count, nullptr)
            && images == nullptr && count == 0, "Invalid switch did not fail cleanly");
    }
    params.cache.mode = SD_CACHE_UCACHE;
    sd_image_t* images = nullptr; int count = 0;
    require(!generate_image_with_refiner(&base_ctx, &refiner_ctx, .8f, &params, &images, &count, nullptr),
        "Cross-model denoiser cache silently reused");
    require(!generate_image_with_refiner(&base_ctx, nullptr, .8f, &params, &images, &count, nullptr), "Missing Refiner accepted");
}

struct ProbeControl : ControlNet {
    float value;
    bool fail = false;
    int calls = 0;
    std::vector<std::pair<int, int>> sizes;
    bool circular_x_enabled() { return get_context().circular_x_enabled; }
    bool circular_y_enabled() { return get_context().circular_y_enabled; }
    ProbeControl(ggml_backend_t backend, float output) : ControlNet(backend, backend), value(output) {}
    std::optional<std::vector<sd::Tensor<float>>> compute(int, const sd::Tensor<float>& x,
        const sd::Tensor<float>& hint, const sd::Tensor<float>&, const sd::Tensor<float>&,
        const sd::Tensor<float>&) override {
        ++calls; sizes.emplace_back(int(hint.shape()[0]), int(hint.shape()[1]));
        if (fail) return std::nullopt;
        require(runner_start(), "Control runner failed to start");
        return std::vector<sd::Tensor<float>>{sd::Tensor<float>(x.shape(), std::vector<float>(x.numel(), value))};
    }
};

void multi_control(ggml_backend_t backend) {
    StableDiffusionGGML engine; initialize(engine, backend, false);
    sd_ctx_t ctx{&engine};
    auto first = std::make_shared<ProbeControl>(backend, 2.f);
    auto second = std::make_shared<ProbeControl>(backend, 4.f);
    engine.control_models = {{nullptr, first}, {nullptr, second}};
    apply_circular_axes_to_diffusion(&ctx, true, false);
    require(first->circular_x_enabled() && second->circular_x_enabled()
        && !first->circular_y_enabled() && !second->circular_y_enabled(),
        "Multi-ControlNet did not inherit seamless axes");
    apply_circular_axes_to_diffusion(&ctx, false, false);
    require(!first->circular_x_enabled() && !second->circular_x_enabled(),
        "Disabling seamless tiling retained old ControlNet axes");
    std::vector<uint8_t> rgbA(12, 32), rgbB(12, 224);
    sd_control_input_t inputs[]{{{2, 2, 3, rgbA.data()}, {}, .5f}, {{2, 2, 3, rgbB.data()}, {}, 1.5f}};
    require(!sd_set_control_net_inputs(nullptr, inputs, 2)
        && !sd_set_control_net_inputs(&ctx, inputs, 1), "Unprepared model count accepted");
    require(sd_set_control_net_inputs(&ctx, inputs, 2), "Prepared stack inputs rejected");
    const auto positive = condition(1, 10), negative = condition(-1, 10);
    sd::Tensor<float> hint({16, 16, 3, 1}), initial({2, 2, 4, 1}), noise(initial.shape());
    noise.fill_(.2f); initial.fill_(.1f);
    sd_guidance_params_t guidance{}; guidance.txt_cfg = 2;
    auto runner = std::static_pointer_cast<ProbeRunner>(engine.diffusion_model);
    runner->expect_multi_control = true;
    const auto sample = [&] {
        return engine.sample(engine.diffusion_model, true, initial, noise, positive, negative, {}, hint,
            .2f, guidance, 0.f, 0, EULER_SAMPLE_METHOD, false, nullptr, {1.f, .5f, 0.f}, {}, {},
            {}, {}, 1.f, 0, 1.f, nullptr, false);
    };
    require(!sample().empty() && first->calls > 0 && first->calls == second->calls,
        "Actual sampler did not execute all ControlNets");
    require(!first->runner_started() && !second->runner_started(), "Successful stack leaked runner state");
    for (const auto size : first->sizes) require(size == std::pair<int, int>{16, 16}, "Wrong base-pass hint size");
    second->fail = true;
    require(sample().empty() && !first->runner_started() && !second->runner_started(),
        "Failed later ControlNet published a partial result or leaked runner state");
    second->fail = false;
    initial = sd::Tensor<float>({4, 3, 4, 1}); initial.fill_(.1f);
    noise = initial; hint = sd::Tensor<float>({32, 24, 3, 1});
    require(!sample().empty() && first->sizes.back() == std::pair<int, int>{32, 24}
        && second->sizes.back() == std::pair<int, int>{32, 24}, "Larger pass reused a stale hint");
    require(sd_set_control_net_inputs(&ctx, nullptr, 0) && engine.control_models.size() == 2
        && !engine.control_stack.active(), "Disabling inputs unloaded weights");
    require(sd_prepare_control_nets(&ctx, nullptr, 0) && engine.control_models.empty(),
        "Explicit model release retained the stack");
    const char* invalid[]{""};
    require(!sd_prepare_control_nets(nullptr, invalid, 1)
        && !sd_prepare_control_nets(&ctx, invalid, 1)
        && !sd_prepare_control_nets(&ctx, invalid, 65), "Malformed model preparation accepted");
}

struct ProbeVision : FrozenCLIPVisionEmbedder {
    int calls = 0, fail_at = 0;
    bool plus = false, nonfinite = false;
    std::vector<bool> zero_pixels;
    std::vector<float> first_pixels;
    explicit ProbeVision(ggml_backend_t backend) : FrozenCLIPVisionEmbedder(backend) { vision_model.image_size = 4; }
    sd::Tensor<float> compute(int, const sd::Tensor<float>& pixels, bool pooled, int skip) override {
        require(runner_start(), "Vision runner start failed");
        ++calls;
        require(pooled == !plus && skip == (plus ? 2 : -1), "Wrong classic/Plus vision output selected");
        const bool zero = std::all_of(pixels.values().begin(), pixels.values().end(), [](float x) { return x == 0; });
        zero_pixels.push_back(zero);
        first_pixels.push_back(pixels.values().front());
        if (calls == fail_at) return {};
        sd::Tensor<float> result(plus ? std::vector<int64_t>{3, 2} : std::vector<int64_t>{3});
        result.fill_(nonfinite ? std::numeric_limits<float>::infinity() : (zero ? 7.f : 2.f));
        return result;
    }
};
struct ProbeIPProjection : IPAdapter::IPAdapterRunner {
    int calls = 0, fail_at = 0;
    bool bad_shape = false;
    std::vector<float> embeddings;
    std::function<void(int)> after_compute;
    explicit ProbeIPProjection(ggml_backend_t backend) : IPAdapterRunner(backend, {}, "ip_adapter") {
        image_proj.clip_dim = resampler.embed_dim = 3;
        image_proj.ctx_dim = resampler.output_dim = 4;
        num_tokens = 2;
    }
    sd::Tensor<float> compute(int, const sd::Tensor<float>& input) override {
        require(runner_start(), "Projection runner start failed");
        ++calls; embeddings.push_back(input.values().front());
        if (calls == fail_at) return {};
        sd::Tensor<float> output({bad_shape ? 5 : 4, 2, 1}); output.fill_(input.values().front() + 10.f);
        if (after_compute) after_compute(calls);
        return output;
    }
};
void ip_preparation(ggml_backend_t backend) {
    StableDiffusionGGML engine; initialize(engine, backend, false);
    auto vision = std::make_shared<ProbeVision>(backend);
    auto projection = std::make_shared<ProbeIPProjection>(backend);
    engine.clip_vision = vision; engine.ip_adapter = projection;
    std::vector<uint8_t> pixels(4 * 4 * 3, 127);
    const sd_image_t input{4, 4, 3, pixels.data()};
    for (bool plus : {false, true}) {
        vision->plus = projection->is_plus = plus;
        vision->calls = projection->calls = 0;
        vision->zero_pixels.clear(); projection->embeddings.clear();
        require(engine.compute_ip_adapter_tokens(input, .7f), "Valid IP preparation failed");
        require(projection->embeddings == std::vector<float>({2, plus ? 7.f : 0.f}),
            "Plus unconditional encoding used zero embeddings instead of zero normalized pixels");
        require(vision->zero_pixels == (plus ? std::vector<bool>{false, true} : std::vector<bool>{false}),
            "Vision input/number of encodes does not match IP variant");
        require(engine.ip_adapter_tokens.values().front() == 12.f
            && engine.ip_adapter_uncond_tokens.values().front() == (plus ? 17.f : 10.f)
            && engine.ip_adapter_strength == .7f && !vision->runner_started() && !projection->runner_started(),
            "Prepared tokens/strength were lost or runner state leaked");
        for (int failed_stage : {0, 1, 2, 3, 4, 5}) {
            vision->calls = projection->calls = 0;
            vision->fail_at = failed_stage == 0 ? 1 : failed_stage == 2 && plus ? 2 : 0;
            projection->fail_at = failed_stage == 1 ? 1 : failed_stage == 3 ? 2 : 0;
            vision->nonfinite = failed_stage == 4;
            projection->bad_shape = failed_stage == 5;
            if (failed_stage == 2 && !plus) continue;
            require(!engine.compute_ip_adapter_tokens(input, .7f)
                && engine.ip_adapter_tokens.empty() && engine.ip_adapter_uncond_tokens.empty()
                && !vision->runner_started() && !projection->runner_started(),
                "Failed IP preparation retained partial tokens or active runners");
        }
        vision->fail_at = projection->fail_at = 0; vision->nonfinite = projection->bad_shape = false;
        require(engine.compute_ip_adapter_tokens(input, 0), "Zero IP strength skipped valid preparation");
        require(engine.compute_ip_adapter_tokens({}, 1) && engine.ip_adapter_tokens.empty()
            && engine.ip_adapter_uncond_tokens.empty(), "Disable retained IP conditioning");
        require(engine.clip_vision == vision && engine.ip_adapter == projection, "Disable unloaded IP weights");
    }
    for (float invalid : {-1.f, 2.1f, std::numeric_limits<float>::quiet_NaN()})
        require(!engine.compute_ip_adapter_tokens(input, invalid), "Invalid IP strength accepted");
    require(!engine.compute_ip_adapter_tokens({4, 4, 3, nullptr}, 1), "Missing RGB data accepted");
    bool cancelled = true;
    sd_set_abort_callback([](void* value) { return *static_cast<bool*>(value); }, &cancelled);
    const bool succeeded = engine.compute_ip_adapter_tokens(input, 1);
    sd_set_abort_callback(nullptr, nullptr);
    require(!succeeded && engine.ip_adapter_tokens.empty(), "Cancelled IP conditioning committed tokens");
    cancelled = false; projection->calls = 0;
    projection->after_compute = [&](int calls) { if (calls == 2) cancelled = true; };
    sd_set_abort_callback([](void* value) { return *static_cast<bool*>(value); }, &cancelled);
    const bool late_succeeded = engine.compute_ip_adapter_tokens(input, 1);
    sd_set_abort_callback(nullptr, nullptr); projection->after_compute = {};
    require(!late_succeeded && engine.ip_adapter_tokens.empty() && engine.ip_adapter_uncond_tokens.empty()
        && !vision->runner_started() && !projection->runner_started(),
        "Late IP projection cancellation committed partial conditioning");
}
void indexed_ip_preparation(ggml_backend_t backend) {
    StableDiffusionGGML engine; initialize(engine, backend, false);
    auto first_vision = std::make_shared<ProbeVision>(backend);
    auto second_vision = std::make_shared<ProbeVision>(backend);
    auto first = std::make_shared<ProbeIPProjection>(backend);
    auto second = std::make_shared<ProbeIPProjection>(backend);
    second_vision->plus = second->is_plus = true;
    engine.ip_stack.models = {{first_vision, first}, {second_vision, second}};
    sd_ctx_t context{&engine};
    std::vector<uint8_t> pixels(4 * 4 * 3, 127), mask{0, 255};
    sd_ip_adapter_input_t inputs[]{
        {1, {4, 4, 3, pixels.data()}, {2, 1, 1, mask.data()}, .3f},
        {0, {4, 4, 3, pixels.data()}, {}, .7f}};
    const float original = clip_preprocess(sd_image_to_tensor(inputs[0].image), 4, 4).values().front();
    require(sd_set_ip_adapter_inputs(&context, inputs, 2), "Indexed IP inputs rejected");
    std::fill(pixels.begin(), pixels.end(), 0); mask = {255, 0};
    require(engine.ip_stack.prepare(1), "Independent Classic/Plus preparation failed");
    require(first_vision->first_pixels.front() == original && second_vision->first_pixels.front() == original,
        "Indexed IP inputs did not own source RGB pixels");
    const auto& positive = engine.ip_stack.views(false);
    const auto& negative = engine.ip_stack.views(true);
    require(positive.size() == 2 && negative.size() == 2 && positive[0].slot == 1 && positive[1].slot == 0
        && positive[0].scale == .3f && positive[1].scale == .7f
        && positive[0].tokens->values()[0] == 12.f && positive[1].tokens->values()[0] == 12.f
        && negative[0].tokens->values()[0] == 17.f && negative[1].tokens->values()[0] == 10.f,
        "Indexed token branches, order or per-entry strength changed");
    require(positive[0].coverage && *positive[0].coverage(4, 1) == std::vector<float>({0, .25f, .75f, 1})
        && !positive[1].coverage && *negative[0].coverage(4, 1) == *positive[0].coverage(4, 1),
        "Borrowed source mutation changed owned regional masks or unconditional coverage");
    require(!first->runner_started() && !second->runner_started() && !first_vision->runner_started()
        && !second_vision->runner_started(), "Indexed preparation leaked runner scratch");
    first->fail_at = first->calls + 1;
    require(!engine.ip_stack.prepare(1) && engine.ip_stack.views(false).empty() && engine.ip_stack.views(true).empty(),
        "Later adapter failure retained a previous or partially prepared request");
    first->fail_at = 0;
    require(engine.ip_stack.prepare(1), "Indexed preparation did not recover after failure");
    bool abort = false;
    sd_set_abort_callback([](void* p) { return *static_cast<bool*>(p); }, &abort);
    const auto last_call = first->calls + 2;
    first->after_compute = [&](int call) { if (call == last_call) abort = true; };
    const bool cancelled = !engine.ip_stack.prepare(1) && engine.ip_stack.views(false).empty();
    first->after_compute = {}; sd_set_abort_callback(nullptr, nullptr);
    require(cancelled && !first->runner_started() && !second->runner_started(), "Late indexed cancellation committed partial tokens");
    inputs[1].slot = 1;
    require(!sd_set_ip_adapter_inputs(&context, inputs, 2), "Duplicate IP input slot accepted");
    inputs[1].slot = 2;
    require(!sd_set_ip_adapter_inputs(&context, inputs, 2), "Unprepared IP input slot accepted");
    require(sd_set_ip_adapter_inputs(&context, nullptr, 0) && !engine.ip_stack.active()
        && engine.ip_stack.views(false).empty() && engine.ip_stack.models.size() == 2
        && engine.ip_stack.models[0].projection == first, "Disabling IP inputs unloaded runtime resources");
}

void missing_ip_adapter(ggml_backend_t backend) {
    StableDiffusionGGML engine; initialize(engine, backend, false);
    engine.first_stage_model = std::make_shared<ProbeVAE>(backend);
    sd_ctx_t ctx{&engine};
    sd_img_gen_params_t params; sd_img_gen_params_init(&params);
    params.prompt = "positive"; params.negative_prompt = "negative";
    params.width = params.height = 64; params.seed = 42;
    params.sample_params.sample_method = EULER_SAMPLE_METHOD;
    params.sample_params.sample_steps = 1; params.sample_params.guidance.txt_cfg = 2.f;
    std::vector<uint8_t> pixels(16 * 16 * 3, 127);
    params.ip_adapter_image = {16, 16, 3, pixels.data()};
    sd_image_t* images = nullptr; int count = 0;
    const bool succeeded = generate_image(&ctx, &params, &images, &count);
    const bool published = images || count;
    free_sd_images(images, count);
    require(!succeeded && !published, "Missing IP-Adapter silently published an unconditioned image");
}

void generation(ggml_backend_t backend) {
    for (int ip : {0, 1, 2}) for (bool multi : {false, true})
        for (bool hires : {false, true}) for (float switch_at : {0.f, .5f, 1.f}) {
        StableDiffusionGGML base, refiner;
        initialize(base, backend, false); initialize(refiner, backend, true);
        auto base_vae = std::make_shared<ProbeVAE>(backend);
        auto refiner_vae = std::make_shared<ProbeVAE>(backend);
        base.first_stage_model = base_vae; refiner.first_stage_model = refiner_vae;
        auto base_runner = std::static_pointer_cast<ProbeRunner>(base.diffusion_model);
        auto refiner_runner = std::static_pointer_cast<ProbeRunner>(refiner.diffusion_model);
        auto refiner_conditioner = std::static_pointer_cast<ProbeConditioner>(refiner.cond_stage_model);
        sd_ctx_t base_ctx{&base}, refiner_ctx{&refiner};
        auto first = std::make_shared<ProbeControl>(backend, 2.f);
        auto second = std::make_shared<ProbeControl>(backend, 4.f);
        if (multi) {
            base.control_models = {{nullptr, first}, {nullptr, second}};
            std::vector<uint8_t> rgbA(12, 32), rgbB(12, 224);
            sd_control_input_t inputs[]{{{2, 2, 3, rgbA.data()}, {}, .5f}, {{2, 2, 3, rgbB.data()}, {}, 1.5f}};
            require(sd_set_control_net_inputs(&base_ctx, inputs, 2), "Public generation inputs rejected");
            base_runner->expect_multi_control = true;
        }
        sd_img_gen_params_t params; sd_img_gen_params_init(&params);
        std::vector<uint8_t> ip_pixels(4 * 4 * 3, 127);
        if (ip == 1) {
            auto vision = std::make_shared<ProbeVision>(backend);
            auto projection = std::make_shared<ProbeIPProjection>(backend);
            vision->plus = projection->is_plus = hires;
            base.clip_vision = vision; base.ip_adapter = projection;
            params.ip_adapter_image = {4, 4, 3, ip_pixels.data()}; params.ip_adapter_strength = .7f;
            base_runner->expect_ip = true;
            base_runner->expected_ip_unconditional = hires ? 17.f : 10.f;
        }
        if (ip == 2) {
            auto a = std::make_shared<ProbeVision>(backend), b = std::make_shared<ProbeVision>(backend);
            auto pa = std::make_shared<ProbeIPProjection>(backend), pb = std::make_shared<ProbeIPProjection>(backend);
            b->plus = pb->is_plus = true;
            base.ip_stack.models = {{a, pa}, {b, pb}};
            std::vector<uint8_t> mask{0, 255};
            sd_ip_adapter_input_t inputs[]{
                {1, {4, 4, 3, ip_pixels.data()}, {2, 1, 1, mask.data()}, .3f},
                {0, {4, 4, 3, ip_pixels.data()}, {}, .7f}};
            require(sd_set_ip_adapter_inputs(&base_ctx, inputs, 2), "Generation rejected indexed IP inputs");
            base_runner->expect_indexed_ip = true;
        }
        params.prompt = "positive"; params.negative_prompt = "negative";
        params.width = params.height = 64; params.seed = 42; params.batch_count = 2;
        params.sample_params.sample_method = EULER_SAMPLE_METHOD;
        params.sample_params.sample_steps = 4;
        params.sample_params.guidance.txt_cfg = 2.f;
        float sigmas[]{2, 1.5f, 1, .5f, 0};
        params.sample_params.custom_sigmas = sigmas; params.sample_params.custom_sigmas_count = 5;
        params.hires.enabled = hires; params.hires.upscaler = SD_HIRES_UPSCALER_LANCZOS;
        params.hires.target_width = params.hires.target_height = 128;
        params.hires.custom_sigmas = sigmas; params.hires.custom_sigmas_count = 5;
        sd_image_t* images = nullptr; int count = 0;
        require(generate_image_with_refiner(&base_ctx, &refiner_ctx, switch_at, &params, &images, &count, "user negative"),
            "Full native Refiner generation failed");
        require(count == 2 && images, "Native batch output missing");
        for (int i = 0; i < count; ++i)
            require(images[i].data && images[i].channel == 3 && images[i].width == (hires ? 128u : 64u)
                && images[i].height == images[i].width, "Final Refiner output shape changed");
        free_sd_images(images, count);
        const int base_steps = switch_at == 0 ? 0 : switch_at == 1 ? 4 : 2;
        require(base_runner->calls == ((hires ? 4 : 0) + base_steps) * 4
            && refiner_runner->calls == (4 - base_steps) * 4,
            "Refiner applied to the wrong pass or batch steps");
        if (multi) {
            require(first->calls == base_runner->calls && second->calls == first->calls,
                "Multi-ControlNet did not follow Base batch/Refiner/Hires routing");
            if (hires && base_steps)
                require(first->sizes.back() == std::pair<int, int>{128, 128}
                    && second->sizes.back() == first->sizes.back(), "Hires omitted multi hint resize");
        }
        require(base_vae->decodes == (hires ? 4 : 2) && base_vae->encodes == (hires ? 2 : 0) + (multi ? 1 : 0),
            "Unexpected decode/re-encode at Refiner switch");
        require(refiner_vae->decodes == 0 && refiner_vae->encodes == 0,
            "Refiner used an intermediate RGB path");
        require(refiner_conditioner->calls == (switch_at < 1 ? 2 : 0), "Refiner conditioning repeated or unused encoding performed");
        if (switch_at < 1)
            require(refiner_conditioner->texts == std::vector<std::string>{"positive", "user negative"},
                "Refiner did not receive its independent negative conditioning");
        const auto base_conditioner = std::static_pointer_cast<ProbeConditioner>(base.cond_stage_model);
        require(base_conditioner->texts.size() >= 2 && base_conditioner->texts[1] == "negative",
            "Refiner override changed Base negative conditioning");
        for (auto size : refiner_conditioner->sizes)
            require(size == std::pair<int, int>{hires ? 128 : 64, hires ? 128 : 64}, "Wrong Refiner ADM image size");

        params.hires.enabled = false; params.batch_count = 1;
        refiner_runner->fail = true; const auto decodes_before = base_vae->decodes;
        images = nullptr; count = 0;
        require(!generate_image_with_refiner(&base_ctx, &refiner_ctx, .5f, &params, &images, &count, nullptr)
            && !images && count == 0 && base_vae->decodes == decodes_before,
            "Failed Refiner published a Base image");
        refiner_runner->fail = false;

        // Cancellation on the last Refiner compute must stop publication even
        // when no later sampling iteration can observe the abort callback.
        bool abort = false;
        sd_set_abort_callback([](void* value) { return *static_cast<bool*>(value); }, &abort);
        struct ResetAbort { ~ResetAbort() { sd_set_abort_callback(nullptr, nullptr); } } reset;
        refiner_runner->after_compute = [&] { abort = true; };
        float last_sigmas[]{1, 0}; params.sample_params.custom_sigmas = last_sigmas;
        params.sample_params.custom_sigmas_count = 2; params.sample_params.sample_steps = 1;
        params.sample_params.guidance.txt_cfg = 1.f;
        require(!generate_image_with_refiner(&base_ctx, &refiner_ctx, 0.f, &params, &images, &count, nullptr)
            && !images && count == 0 && base_vae->decodes == decodes_before,
            "Late Refiner cancellation published output");
    }
}

int main() {
    try {
        // Exercise the real engine getter, without pretrained weight loading.
        // Metadata recognition alone must not hide an unknown runtime family.
        StableDiffusionGGML krea;
        krea.version = VERSION_KREA2;
        sd_ctx_t krea_context{&krea};
        require(std::string(sd_get_model_family(&krea_context)) == "krea2",
            "Krea2 runtime family was lost after metadata inspection");
        require(std::string(sd_vae_contract(VERSION_KREA2).model_family) == "krea2",
            "Krea2 metadata family disagrees with runtime family");
        auto backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(ggml_backend_cpu_init(), &ggml_backend_free);
        require(bool(backend), "CPU fixture backend");
        sampling(backend.get());
        api(backend.get());
        missing_ip_adapter(backend.get());
        ip_preparation(backend.get());
        indexed_ip_preparation(backend.get());
        multi_control(backend.get());
        generation(backend.get());
        std::cout << "Refiner sampler contracts passed\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
