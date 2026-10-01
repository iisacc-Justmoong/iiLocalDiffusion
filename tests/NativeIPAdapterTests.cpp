#include "model/adapter/ip_adapter.hpp"
#include "model/diffusion/unet.hpp"
#include "runtime/control_region_mask.hpp"
#include <ggml-cpu.h>
#if defined(__APPLE__)
#include <ggml-metal.h>
#endif
#include <iostream>
#include <numeric>
#include <stdexcept>

void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
std::vector<float> normalized(const std::vector<float>& values) {
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double variance = 0;
    for (float value : values) variance += (value - mean) * (value - mean);
    variance /= values.size();
    std::vector<float> result;
    for (float value : values) result.push_back(float((value - mean) / std::sqrt(variance + 1e-5)));
    return result;
}
struct Graph {
    ggml_backend_t backend;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context;
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer{nullptr, ggml_backend_buffer_free};
    GGMLRunnerContext runner;
    std::map<std::string, ggml_tensor*> weights;
    ggml_cgraph* graph = nullptr;
    ggml_tensor* output = nullptr;
    std::map<ggml_tensor*, const void*> inputs;
    explicit Graph(ggml_backend_t device)
        : backend(device), context(ggml_init({8 * 1024 * 1024, nullptr, true}), ggml_free) {
        require(bool(context), "Graph metadata allocation failed");
        runner.backend = backend; runner.ggml_ctx = context.get();
        runner.set_backend_tensor_data = [&](ggml_tensor* tensor, const void* values) { inputs[tensor] = values; };
    }
    void allocate(ggml_tensor* result) {
        output = result; ggml_set_output(output);
        graph = ggml_new_graph_custom(context.get(), 4096, false);
        ggml_build_forward_expand(graph, output);
        buffer.reset(ggml_backend_alloc_ctx_tensors(context.get(), backend));
        require(bool(buffer), "Graph storage allocation failed");
        for (const auto& [name, tensor] : weights) set(tensor, std::vector<float>(ggml_nelements(tensor), 0.f));
        for (const auto& [tensor, values] : inputs) ggml_backend_tensor_set(tensor, values, 0, ggml_nbytes(tensor));
    }
    void set(ggml_tensor* tensor, const std::vector<float>& values) {
        if ((tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16)
            || values.size() != size_t(ggml_nelements(tensor)))
            throw std::runtime_error(std::string("Invalid fixture tensor: ") + ggml_get_name(tensor));
        if (tensor->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> half(values.size());
            ggml_fp32_to_fp16_row(values.data(), half.data(), half.size());
            ggml_backend_tensor_set(tensor, half.data(), 0, half.size() * sizeof(ggml_fp16_t));
        } else ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
    }
    void set(const std::string& name, const std::vector<float>& values) { set(weights.at(name), values); }
    void identity(const std::string& name, int offset = 0) {
        auto* tensor = weights.at(name);
        std::vector<float> values(ggml_nelements(tensor));
        for (int i = 0; i < tensor->ne[0] && i + offset < tensor->ne[1]; ++i)
            values[(i + offset) * tensor->ne[0] + i] = 1.f;
        set(tensor, values);
    }
    std::vector<float> run() {
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "IP-Adapter graph compute failed");
        std::vector<float> result(ggml_nelements(output));
        ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
        return result;
    }
};
void close(const std::vector<float>& actual, const std::vector<float>& expected) {
    require(actual.size() == expected.size(), "IP-Adapter output size mismatch");
    for (size_t i = 0; i < actual.size(); ++i)
        require(std::isfinite(actual[i]) && std::abs(actual[i] - expected[i]) < 3e-4f,
            "IP-Adapter output differs from independent numerical oracle");
}
void classic(ggml_backend_t backend) {
    Graph g(backend);
    IPAdapter::ImageProjModel model(2, 4, 3); model.init(g.context.get(), {}, "classic");
    model.get_param_tensors(g.weights, "classic");
    auto* input = ggml_new_tensor_2d(g.context.get(), GGML_TYPE_F32, 3, 1);
    ggml_set_input(input);
    g.allocate(model.forward(&g.runner, input));
    std::vector<float> matrix(24), bias(8);
    for (int row = 0; row < 8; ++row) {
        bias[row] = row * row * .01f;
        for (int column = 0; column < 3; ++column) matrix[row * 3 + column] = float((row + column * 2) % 5 - 2) * .2f;
    }
    g.set("classic.proj.weight", matrix); g.set("classic.proj.bias", bias);
    const std::vector<float> gamma{1, .5f, 1.5f, 2}, beta{.1f, .2f, .3f, .4f};
    g.set("classic.norm.weight", gamma); g.set("classic.norm.bias", beta);
    for (const std::vector<float> values : {std::vector<float>{.2f, .4f, -.7f}, {-.3f, .8f, .1f}, {0, 0, 0}}) {
        g.set(input, values);
        std::vector<float> expected;
        for (int token = 0; token < 2; ++token) {
            std::vector<float> projected(4);
            for (int channel = 0; channel < 4; ++channel) {
                const int row = token * 4 + channel; projected[channel] = bias[row];
                for (int i = 0; i < 3; ++i) projected[channel] += matrix[row * 3 + i] * values[i];
            }
            projected = normalized(projected);
            for (int i = 0; i < 4; ++i) expected.push_back(projected[i] * gamma[i] + beta[i]);
        }
        close(g.run(), expected);
    }
}
void plus(ggml_backend_t backend) {
    Graph g(backend);
    IPAdapter::Resampler model(64, 1, 2, 3, 4, 128); model.init(g.context.get(), {}, "plus");
    model.get_param_tensors(g.weights, "plus");
    auto* input = ggml_new_tensor_3d(g.context.get(), GGML_TYPE_F32, 3, 2, 1);
    ggml_set_input(input);
    g.allocate(model.forward(&g.runner, input));
    for (const auto& [name, tensor] : g.weights)
        if (name.ends_with(".weight") && name.find("norm") != std::string::npos)
            g.set(tensor, std::vector<float>(ggml_nelements(tensor), 1.f));
    g.identity("plus.proj_in.weight");
    g.identity("plus.layers.0.0.to_kv.weight", 64); // Zero Q/K => uniform attention.
    g.identity("plus.layers.0.0.to_out.weight");
    g.identity("plus.proj_out.weight");
    const std::vector<float> bias{0, .1f, .2f, .3f};
    g.set("plus.proj_out.bias", bias);
    for (const std::vector<float> values : {std::vector<float>{.4f, -.5f, .7f, .9f, .1f, -.2f}, {0, 0, 0, 0, 0, 0}}) {
        g.set(input, values);
        std::vector<float> projected = bias;
        for (int token = 0; token < 2; ++token) {
            std::vector<float> encoded(64);
            for (int i = 0; i < 3; ++i) encoded[i] = values[token * 3 + i];
            encoded = normalized(encoded);
            for (int i = 0; i < 4; ++i) projected[i] += encoded[i] / 4.f; // Two image + two latent tokens.
        }
        projected = normalized(projected);
        auto expected = projected; expected.insert(expected.end(), projected.begin(), projected.end());
        close(g.run(), expected);
    }
}
void attention(ggml_backend_t backend, float strength) {
    Graph g(backend);
    String2TensorStorage metadata;
    metadata["attention.to_k_ip.weight"].ne[0] = 4;
    CrossAttention model(4, 4, 1, 4, true); model.init(g.context.get(), metadata, "attention");
    model.get_param_tensors(g.weights, "attention");
    auto* query = ggml_new_tensor_3d(g.context.get(), GGML_TYPE_F32, 4, 2, 1);
    auto* text = ggml_new_tensor_3d(g.context.get(), GGML_TYPE_F32, 4, 2, 1);
    auto* image = ggml_new_tensor_3d(g.context.get(), GGML_TYPE_F32, 4, 3, 1);
    ggml_set_input(query); ggml_set_input(text); ggml_set_input(image);
    g.runner.ip_context = image; g.runner.ip_scale = strength;
    g.allocate(model.forward(&g.runner, query, text));
    for (const auto* name : {"attention.to_v.weight", "attention.to_v_ip.weight", "attention.to_out.0.weight"}) g.identity(name);
    g.set(query, std::vector<float>(8, .7f));
    g.set(text, {1, 2, 3, 4, 3, 4, 5, 6});
    g.set(image, {2, 4, 6, 8, 4, 6, 8, 10, 6, 8, 10, 12});
    std::vector<float> expected;
    for (int token = 0; token < 2; ++token) for (int channel = 0; channel < 4; ++channel)
        expected.push_back(float(2 + channel) + strength * float(4 + channel * 2));
    close(g.run(), expected); // Decoupled softmax, sum before output projection.
}
void bindings(ggml_backend_t backend) {
    Graph g(backend);
    const auto tensor = [&](int width) { return ggml_new_tensor_2d(g.context.get(), GGML_TYPE_F32, width, 4); };
    std::map<std::string, ggml_tensor*> weights{
        {"unet.input.attn2.to_k.weight", tensor(4)},
        {"unet.input.attn2.to_k_ip.weight", tensor(3)},
        {"unet.input.attn2.to_v_ip.weight", tensor(3)}};
    require(sd_valid_ip_attention_bindings(weights, 3), "Complete IP attention was rejected");
    require(!sd_valid_ip_attention_bindings(weights, 4), "Wrong IP context width was accepted");
    weights.erase("unet.input.attn2.to_v_ip.weight");
    require(!sd_valid_ip_attention_bindings(weights, 3), "Incomplete K/V pair was accepted");
    weights["unet.input.attn2.to_v_ip.weight"] = tensor(3);
    weights["unet.output.attn2.to_k.weight"] = tensor(4);
    require(!sd_valid_ip_attention_bindings(weights, 3), "A partially attached UNet was accepted");
    weights.clear(); weights["ip_adapter.image_proj.proj.weight"] = tensor(3);
    require(!sd_valid_ip_attention_bindings(weights, 3), "Projection-only weights were accepted as an adapter");
}
void multi_attention(ggml_backend_t backend, bool regional, bool shared_batch, float strength) {
    Graph g(backend);
    String2TensorStorage metadata;
    for (int slot = 0; slot < 2; ++slot) for (const auto* role : {"to_k", "to_v"}) {
        auto& storage = metadata["attention.ip_adapters." + std::to_string(slot) + "." + role + ".weight"];
        storage.ne[0] = slot == 0 ? 4 : 3; storage.ne[1] = 4;
    }
    CrossAttention model(4, 4, 1, 4, true); model.init(g.context.get(), metadata, "attention");
    model.get_param_tensors(g.weights, "attention");
    auto* query = ggml_new_tensor_3d(g.context.get(), GGML_TYPE_F32, 4, 6, 2);
    auto* text = ggml_new_tensor_3d(g.context.get(), GGML_TYPE_F32, 4, 2, 2);
    auto* first = ggml_new_tensor_3d(g.context.get(), GGML_TYPE_F32, 4, 2, shared_batch ? 1 : 2);
    auto* second = ggml_new_tensor_3d(g.context.get(), GGML_TYPE_F32, 3, 3, shared_batch ? 1 : 2);
    for (auto* tensor : {query, text, first, second}) ggml_set_input(tensor);
    ControlRegionMask masks[2];
    std::vector<uint8_t> pixels{0, 255, 255, 0};
    require(masks[0].set({2, 2, 1, pixels.data()}), "First attention mask rejected");
    std::reverse(pixels.begin(), pixels.begin() + 2); std::reverse(pixels.begin() + 2, pixels.end());
    require(masks[1].set({2, 2, 1, pixels.data()}), "Second attention mask rejected");
    g.runner.ip_spatial_width = 3; g.runner.ip_spatial_height = 2;
    g.runner.ip_adapters = {{0, first, strength, {}}, {1, second, .25f, {}}};
    if (regional) for (int i = 0; i < 2; ++i)
        g.runner.ip_adapters[i].coverage = [&, i](int w, int h) { return masks[i].coverage(w, h); };
    g.allocate(model.forward(&g.runner, query, text));
    g.identity("attention.to_v.weight"); g.identity("attention.ip_adapters.0.to_v.weight");
    g.set("attention.ip_adapters.1.to_v.weight", {2, 0, 0, 0, 3, 0, 0, 0, 4, 1, 1, 1});
    g.set("attention.to_out.0.weight", {2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2});
    g.set("attention.to_out.0.bias", {.1f, .2f, .3f, .4f});
    g.set(query, std::vector<float>(48, .7f));
    g.set(text, {1, 2, 3, 4, 3, 4, 5, 6, 11, 12, 13, 14, 13, 14, 15, 16});
    std::vector<float> a{2, 4, 6, 8, 4, 6, 8, 10}, b{1, 2, 3, 3, 4, 5, 5, 6, 7};
    if (!shared_batch) {
        for (int i = 0; i < 8; ++i) a.push_back(a[i] + 10);
        for (int i = 0; i < 9; ++i) b.push_back(b[i] + 10);
    }
    g.set(first, a); g.set(second, b);
    const float mask_values[]{0, .5f, 1, 1, .5f, 0};
    std::vector<float> expected;
    for (int batch = 0; batch < 2; ++batch) for (int token = 0; token < 6; ++token) {
        const float shift = shared_batch ? 0 : float(batch * 10);
        const float projected[]{2 * (3 + shift), 3 * (4 + shift), 4 * (5 + shift), 12 + 3 * shift};
        for (int channel = 0; channel < 4; ++channel) {
            const float base = float(2 + channel + batch * 10);
            const float contribution = strength * (3 + 2 * channel + shift) * (regional ? mask_values[token] : 1.f)
                + .25f * projected[channel] * (regional ? 1.f - mask_values[token] : 1.f);
            expected.push_back(2 * (base + contribution) + float(channel + 1) * .1f);
        }
    }
    close(g.run(), expected);
    require(g.inputs.size() == (regional ? 2 : 0), "Attention mask inputs were lost or duplicated");
}
void invalid_attention(ggml_backend_t backend) {
    for (int failure = 0; failure < 19; ++failure) {
        Graph g(backend);
        String2TensorStorage metadata;
        for (const auto* role : {"to_k", "to_v"}) {
            auto& storage = metadata[std::string("attention.ip_adapters.0.") + role + ".weight"];
            storage.ne[0] = 4; storage.ne[1] = 4;
        }
        if (failure == 0) metadata.erase("attention.ip_adapters.0.to_v.weight");
        if (failure == 1) metadata.at("attention.ip_adapters.0.to_v.weight").ne[1] = 3;
        CrossAttention model(4, 4, 1, 4, true); model.init(g.context.get(), metadata, "attention");
        auto* query = ggml_new_tensor_3d(g.context.get(), GGML_TYPE_F32, 4, 6, 2);
        auto* text = ggml_new_tensor_3d(g.context.get(), GGML_TYPE_F32, 4, 2, 2);
        auto* tokens = ggml_new_tensor_4d(g.context.get(), GGML_TYPE_F32,
            failure == 2 ? 3 : 4, 2, failure == 3 ? 3 : 1, failure == 4 ? 2 : 1);
        std::vector<float> coverage(6, .5f);
        g.runner.ip_spatial_width = 3; g.runner.ip_spatial_height = 2;
        g.runner.ip_adapters = {{0, tokens, 0.f, [&](int, int) { return &coverage; }}};
        auto& input = g.runner.ip_adapters.front();
        switch (failure) {
        case 5: input.tokens = nullptr; break;
        case 6: input.slot = 64; break;
        case 7: input.scale = -1.f; break;
        case 8: input.scale = std::numeric_limits<float>::infinity(); break;
        case 9: input.scale = std::numeric_limits<float>::quiet_NaN(); break;
        case 10: input.scale = 2.01f; break;
        case 11: g.runner.ip_spatial_width = 0; break;
        case 12: g.runner.ip_spatial_height = 3; break;
        case 13: g.runner.set_backend_tensor_data = {}; break;
        case 14: coverage.resize(5); break;
        case 15: coverage[0] = -0.1f; break;
        case 16: coverage[5] = std::numeric_limits<float>::quiet_NaN(); break;
        case 17: g.runner.ip_adapters.push_back(input); break;
        case 18: g.runner.ip_context = tokens; break;
        }
        bool rejected = false;
        try { (void)model.forward(&g.runner, query, text); }
        catch (const IPAttentionInputError&) { rejected = true; }
        require(rejected, "Invalid or partial IP attention input was silently accepted, even at strength zero");
    }
}
void nonuniform_attention(ggml_backend_t backend) {
    Graph g(backend);
    String2TensorStorage metadata;
    for (int slot : {0, 7}) for (const auto* role : {"to_k", "to_v"}) {
        auto& storage = metadata["attention.ip_adapters." + std::to_string(slot) + "." + role + ".weight"];
        storage.ne[0] = storage.ne[1] = 4;
    }
    CrossAttention model(4, 4, 1, 4, true); model.init(g.context.get(), metadata, "attention");
    model.get_param_tensors(g.weights, "attention");
    const auto tensor = [&](int count) {
        auto* result = ggml_new_tensor_3d(g.context.get(), GGML_TYPE_F32, 4, count, 1);
        ggml_set_input(result); return result;
    };
    auto* query = tensor(2); auto* text = tensor(2); auto* first = tensor(2); auto* second = tensor(3);
    g.runner.ip_adapters = {{7, second, .25f, {}}, {0, first, .5f, {}}};
    g.allocate(model.forward(&g.runner, query, text));
    for (const auto* name : {"to_q", "to_k", "to_v", "to_out.0", "ip_adapters.0.to_k", "ip_adapters.0.to_v"})
        g.identity(std::string("attention.") + name + ".weight");
    g.set("attention.ip_adapters.7.to_k.weight", {2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2});
    g.set("attention.ip_adapters.7.to_v.weight", {3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3});
    const std::vector<float> a{1, 2, 3, 4, 2, 3, 4, 5}, b{2, 1, 0, -1, 1, 2, 3, 4, -1, 0, 1, 2};
    g.set(query, {1, 0, 0, 0, 0, 1, 0, 0}); g.set(text, a); g.set(first, a); g.set(second, b);
    const auto oracle = [](const std::vector<float>& values, int axis, float key_scale, float value_scale, int channel) {
        double total = 0, output = 0;
        for (size_t i = 0; i < values.size(); i += 4) {
            const double probability = std::exp(double(values[i + axis] * key_scale / 2));
            total += probability; output += probability * values[i + channel] * value_scale;
        }
        return float(output / total);
    };
    std::vector<float> expected;
    for (int query_axis = 0; query_axis < 2; ++query_axis) for (int channel = 0; channel < 4; ++channel)
        expected.push_back(1.5f * oracle(a, query_axis, 1, 1, channel)
            + .25f * oracle(b, query_axis, 2, 3, channel));
    close(g.run(), expected); // Different K and V, sparse slots and reversed request order.
}
void spatial_attention(ggml_backend_t backend, bool linear, bool fail) {
    Graph g(backend);
    String2TensorStorage metadata;
    for (int layer = 0; layer < 2; ++layer) for (const auto* role : {"to_k", "to_v"}) {
        auto& storage = metadata["spatial.transformer_blocks." + std::to_string(layer) + ".attn2.ip_adapters.0." + role + ".weight"];
        storage.ne[0] = 4; storage.ne[1] = 32;
    }
    SpatialTransformer model(32, 1, 32, 2, 4, linear); model.init(g.context.get(), metadata, "spatial");
    model.get_param_tensors(g.weights, "spatial");
    auto* query = ggml_new_tensor_4d(g.context.get(), GGML_TYPE_F32, 3, 2, 32, 2);
    auto* text = ggml_new_tensor_3d(g.context.get(), GGML_TYPE_F32, 4, 2, 2);
    auto* tokens = ggml_new_tensor_3d(g.context.get(), GGML_TYPE_F32, 4, 2, 1);
    for (auto* tensor : {query, text, tokens}) ggml_set_input(tensor);
    const std::vector<float> coverage{0, .25f, .5f, .75f, 1, 0};
    int calls = 0;
    g.runner.ip_adapters = {{0, tokens, .5f, [&](int w, int h) -> const std::vector<float>* {
        ++calls; require(w == 3 && h == 2, "SpatialTransformer inferred a square or transposed the regional mask");
        return fail ? nullptr : &coverage;
    }}};
    g.runner.ip_spatial_width = 11; g.runner.ip_spatial_height = 13;
    ggml_tensor* output = nullptr;
    bool rejected = false;
    try { output = model.forward(&g.runner, query, text); }
    catch (const IPAttentionInputError&) { rejected = true; }
    require(rejected == fail && calls == 1, "Regional coverage was not cached per slot and spatial shape");
    require(g.runner.ip_spatial_width == 11 && g.runner.ip_spatial_height == 13,
        "SpatialTransformer leaked its dimensions on success or exception");
    if (fail) return;
    g.allocate(output);
    for (int layer = 0; layer < 2; ++layer) {
        const auto prefix = "spatial.transformer_blocks." + std::to_string(layer) + ".attn2.";
        std::vector<float> values(128);
        for (int channel = 0; channel < 32; ++channel) values[channel * 4 + channel % 4] = 1;
        g.set(prefix + "to_v.weight", values); g.set(prefix + "ip_adapters.0.to_v.weight", values);
        g.identity(prefix + "to_out.0.weight");
    }
    std::vector<float> projection(1024);
    for (int i = 0; i < 32; ++i) projection[i * 32 + i] = 1;
    g.set("spatial.proj_out.weight", projection);
    g.set(query, std::vector<float>(384, .125f));
    g.set(text, {1, 2, 3, 4, 3, 4, 5, 6, 11, 12, 13, 14, 13, 14, 15, 16});
    g.set(tokens, {2, 4, 6, 8, 4, 6, 8, 10});
    std::vector<float> expected;
    for (int batch = 0; batch < 2; ++batch) for (int channel = 0; channel < 32; ++channel)
        for (int pixel = 0; pixel < 6; ++pixel)
            expected.push_back(.125f + 2 * (2 + channel % 4 + batch * 10)
                + (3 + 2 * (channel % 4)) * coverage[pixel]);
    close(g.run(), expected); // Self-attention must not apply the adapter a second time.
}
// The graph/scheduler/UNet are production code. Only pretrained weight loading
// is replaced with checked, already-resident controlled tensors for this test.
struct ResidentFixture : DeviceResidencyManager {
    bool segmented_compute_enabled() const override { return false; }
    bool prefetch_enabled() const override { return false; }
    void set_workspace_reclaimer(uintptr_t, std::function<bool()>) override {}
    void remove_runtime_owner(uintptr_t) override {}
    bool fits_compute_backend_capacity(const DeviceMemoryRequest&, const std::vector<ggml_tensor*>&) const override { return true; }
    bool assign_compute_backend(const std::vector<ggml_tensor*>&, ggml_backend_t) override { return false; }
    bool prepare_params(const std::vector<ggml_tensor*>& tensors) override {
        return std::all_of(tensors.begin(), tensors.end(), [](const auto* t) { return t && t->buffer && t->data; });
    }
    void release_compute_backend_params(const std::vector<ggml_tensor*>&) override {}
    void evict_compute_backend_params(const std::vector<ggml_tensor*>&) override {}
    WeightResidencyInfo inspect_compute_backend_params(const std::vector<ggml_tensor*>&) const override { return {}; }
    void update_runtime_residency(uintptr_t, ggml_backend_t, size_t) override {}
    bool ensure_compute_backend_capacity(const DeviceMemoryRequest&, const std::vector<ggml_tensor*>&,
        const std::vector<std::vector<ggml_tensor*>>&, const std::vector<ggml_tensor*>&) override { return true; }
    WeightPrefetchResult prefetch_params(uintptr_t, const std::vector<ggml_tensor*>&) override { return WeightPrefetchResult::Unsupported; }
    bool activate_prefetched_params(uintptr_t, const std::vector<ggml_tensor*>&) override { return false; }
    void clear_prefetched_params(uintptr_t) override {}
};
struct SmallUNet : UNetModelRunner {
    using UNetModelRunner::UNetModelRunner;
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> weights{nullptr, ggml_backend_buffer_free};
    ~SmallUNet() { runner_end(); }
    void allocate(ggml_backend_t backend) {
        weights.reset(ggml_backend_alloc_ctx_tensors(params_ctx, backend));
        require(bool(weights), "Controlled UNet weights could not be allocated");
        ggml_backend_buffer_set_usage(weights.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_buffer_clear(weights.get(), 0);
        std::map<std::string, ggml_tensor*> tensors;
        get_param_tensors(tensors, "unet");
        const float bias[]{1, 2, 3, 4};
        ggml_backend_tensor_set(tensors.at("unet.out.2.bias"), bias, 0, sizeof(bias));
    }
};
void unet_inputs(ggml_backend_t backend) {
    String2TensorStorage metadata;
    auto& input = metadata["unet.input_blocks.0.0.weight"];
    input.n_dims = 4; input.ne[0] = input.ne[1] = 3; input.ne[2] = 4; input.ne[3] = 32;
    auto& context = metadata["unet.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight"];
    context.n_dims = 2; context.ne[0] = 4; context.ne[1] = 32;
    {
        UNetModelRunner shape(backend, metadata, "unet");
        std::map<std::string, ggml_tensor*> tensors;
        shape.get_param_tensors(tensors, "unet");
        for (const auto& [name, tensor] : tensors) if (name.ends_with("attn2.to_k.weight")) {
            const auto prefix = name.substr(0, name.size() - std::string("to_k.weight").size());
            for (int slot = 0; slot < 2; ++slot) for (const auto* role : {"to_k", "to_v"}) {
                auto& weight = metadata[prefix + "ip_adapters." + std::to_string(slot) + "." + role + ".weight"];
                weight.n_dims = 2; weight.ne[0] = slot == 0 ? 4 : 3; weight.ne[1] = tensor->ne[1];
            }
        }
    }
    auto resident = std::make_shared<ResidentFixture>();
    SmallUNet model(backend, metadata, "unet", VERSION_SD1, resident);
    std::map<std::string, ggml_tensor*> tensors;
    model.get_param_tensors(tensors, "unet");
    require(sd_valid_ip_attention_bindings(tensors, 4, 0) && sd_valid_ip_attention_bindings(tensors, 3, 1),
        "Indexed adapter weights were not attached to every UNet cross-attention layer");
    require(!sd_valid_ip_attention_bindings(tensors, 3, 0) && !sd_valid_ip_attention_bindings(tensors, 4, 64)
        && !sd_valid_ip_attention_bindings(tensors, 4), "Indexed bindings aliased legacy or another slot");
    model.allocate(backend);
    sd::Tensor<float> x({24, 16, 4, 1}), time({1}, {1}), text({4, 2, 1});
    sd::Tensor<float> first({4, 2, 1}), second({3, 3, 1});
    std::vector<uint8_t> pixels{0, 255, 255, 0};
    ControlRegionMask mask; require(mask.set({2, 2, 1, pixels.data()}), "UNet mask rejected");
    std::map<std::pair<int, int>, int> calls;
    bool fail_mask = false;
    std::vector<IPAdapterInput> adapters{{0, &first, .5f, [&](int w, int h) -> const std::vector<float>* {
        ++calls[{w, h}]; return fail_mask && w == 3 ? nullptr : mask.coverage(w, h);
    }}, {1, &second, .25f, {}}};
    UNetDiffusionExtra extra; extra.ip_adapters = &adapters;
    DiffusionParams params; params.x = &x; params.timesteps = &time; params.context = &text; params.extra = extra;
    const auto success = [&] {
        calls.clear();
        const auto result = model.compute(2, params);
        require(result.shape() == x.shape(), "UNet did not preserve output dimensions with multiple adapters");
        for (size_t i = 0; i < result.values().size(); ++i)
            require(result.values()[i] == float((i / (24 * 16)) % 4 + 1), "Controlled UNet output changed");
        require(calls == std::map<std::pair<int, int>, int>{{{24, 16}, 1}, {{12, 8}, 1}, {{6, 4}, 1}, {{3, 2}, 1}},
            "UNet did not forward adapters through all spatial scales or failed to reuse mask inputs");
    };
    success();
    fail_mask = true;
    require(model.compute(2, params).empty(), "Invalid deep-layer mask escaped fail-closed UNet result handling");
    fail_mask = false;
    first.values()[0] = std::numeric_limits<float>::quiet_NaN();
    require(model.compute(2, params).empty(), "Non-finite CPU tokens reached the UNet graph");
    first.values()[0] = 0;
    adapters[1].slot = 0;
    require(model.compute(2, params).empty(), "Duplicate CPU adapter slots reached the UNet graph");
    adapters[1].slot = 1;
    success(); // Failed graphs must not poison runner scratch, caches or subsequent requests.
    model.runner_end(); require(!model.runner_started(), "UNet runner cleanup failed");
}
int main() {
    try {
        sd_set_log_callback([](sd_log_level_t level, const char* text, void*) {
            if (level >= SD_LOG_WARN) std::cerr << text;
        }, nullptr);
        std::vector<std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>> devices;
        devices.emplace_back(ggml_backend_cpu_init(), ggml_backend_free);
#if defined(__APPLE__)
        devices.emplace_back(ggml_backend_metal_init(), ggml_backend_free);
#endif
        for (const auto& backend : devices) {
            require(bool(backend), "Required IP-Adapter backend unavailable");
            bindings(backend.get()); classic(backend.get()); plus(backend.get());
            for (float strength : {0.f, .5f, 1.f, 2.f}) attention(backend.get(), strength);
            for (bool regional : {false, true}) for (bool shared_batch : {false, true})
                for (float strength : {0.f, .5f, 2.f}) multi_attention(backend.get(), regional, shared_batch, strength);
            invalid_attention(backend.get());
            nonuniform_attention(backend.get());
            for (bool linear : {false, true}) for (bool fail : {false, true}) spatial_attention(backend.get(), linear, fail);
            unet_inputs(backend.get());
            std::cout << "IP-Adapter projection/attention backend=" << ggml_backend_name(backend.get()) << " passed\n";
        }
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
