#include "runtime/ip_adapter_resources.hpp"
#include "model/adapter/ip_adapter.hpp"
#include <ggml-cpu.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void fixture(const std::filesystem::path& path, float value, bool vision = false) {
    std::ostringstream header; header << '{';
    std::vector<float> data;
    bool first = true;
    const auto tensor = [&](const std::string& name, const std::vector<int>& shape) {
        if (!first) header << ','; first = false;
        header << '"' << name << "\":{\"dtype\":\"F32\",\"shape\":[";
        size_t count = 1;
        for (size_t i = 0; i < shape.size(); ++i) { if (i) header << ','; header << shape[i]; count *= shape[i]; }
        const auto offset = data.size() * sizeof(float);
        data.insert(data.end(), count, value);
        header << "],\"data_offsets\":[" << offset << ',' << data.size() * sizeof(float) << "]}";
    };
    if (vision) tensor("vision_model.embeddings.class_embedding", {4});
    else {
        tensor("image_proj.proj.weight", {8, 3}); tensor("image_proj.proj.bias", {8});
        tensor("image_proj.norm.weight", {4}); tensor("image_proj.norm.bias", {4});
        tensor("ip_adapter.1.to_k_ip.weight", {4, 4}); tensor("ip_adapter.1.to_v_ip.weight", {4, 4});
    }
    header << '}'; auto json = header.str(); while (json.size() % 8) json += ' ';
    const uint64_t size = json.size();
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(&size), sizeof(size)); stream.write(json.data(), json.size());
    stream.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    check(bool(stream), "Cannot write IP resource fixture");
}
void imported_resources(const std::filesystem::path& directory) {
    const auto first = directory / "first.safetensors", second = directory / "second.safetensors", vision = directory / "vision.safetensors";
    fixture(first, .25f); fixture(second, .75f); fixture(vision, .5f, true);
    ModelLoader a, b, v, merged;
    check(a.init_from_file_and_convert_name(first.string(), "", VERSION_SD1)
        && b.init_from_file_and_convert_name(second.string(), "", VERSION_SD1)
        && v.init_from_file_and_convert_name(vision.string(), "clip_vision.", VERSION_SD1), "Fixture metadata cannot be loaded");
    const auto adapter_name = [](size_t slot) { return [slot](const std::string& name) { return sd_indexed_ip_weight_name(name, slot); }; };
    check(merged.append_metadata(a, adapter_name(0)) && merged.append_metadata(b, adapter_name(1))
        && merged.append_metadata(v, [](const auto& name) { return sd_indexed_ip_vision_name(name, 0); }),
        "Independent model metadata cannot be imported");
    check(merged.append_metadata(a, adapter_name(2)), "Reusing one model in a different slot failed");
    auto& metadata = merged.get_tensor_storage_map();
    const auto count = metadata.size();
    check(!merged.append_metadata(b, adapter_name(0)) && metadata.size() == count, "Collision partially overwrote an existing adapter");
    check(!merged.append_metadata(a, [](const auto&) { return std::string{}; }) && metadata.size() == count,
        "Empty import changed metadata");
    check(metadata.at("ip_adapter_slots.0.image_proj.proj.weight").file_index
        == metadata.at("ip_adapter_slots.2.image_proj.proj.weight").file_index,
        "Repeated model path created duplicate file storage");
    for (int slot = 0; slot < 3; ++slot)
        check(sd_valid_ip_projection_metadata(metadata, "ip_adapter_slots." + std::to_string(slot)), "Imported projection metadata invalid");
    const auto base = "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.ip_adapters.";
    check(metadata.find(std::string(base) + "1.to_v.weight") != metadata.end(), "Official processor names did not map to indexed UNet weights");
    check(sd_indexed_ip_weight_name("model.diffusion_model.input_blocks.0.0.weight", 0).empty(),
        "An adapter could overwrite unrelated Base tensors");
    merged.set_resident_memory(true); merged.process_model_files(false, false);
    check(!merged.append_metadata(b, adapter_name(3)), "A processed resident loader accepted structural mutation");
    // Delete only files created above: subsequent reads must use anonymous RAM.
    std::filesystem::remove(first); std::filesystem::remove(second); std::filesystem::remove(vision);
    auto ctx = std::unique_ptr<ggml_context, decltype(&ggml_free)>(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    auto cpu = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(ggml_backend_cpu_init(), ggml_backend_free);
    check(bool(ctx) && bool(cpu), "Cannot allocate loader fixture metadata");
    std::map<std::string, ggml_tensor*> tensors;
    for (const auto& [name, storage] : metadata) tensors[name] = ggml_new_tensor(ctx.get(), GGML_TYPE_F32, storage.n_dims, storage.ne);
    auto buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>(
        ggml_backend_alloc_ctx_tensors(ctx.get(), cpu.get()), ggml_backend_buffer_free);
    check(bool(buffer), "Cannot allocate fixture tensors");
    for (int repeat = 0; repeat < 2; ++repeat) {
        ggml_backend_buffer_clear(buffer.get(), 0);
        check(merged.load_tensors(tensors, {}, false), "Resident imported weights tried to reopen removed source files");
        for (const auto& [name, tensor] : tensors) {
            const float expected = name.starts_with("ip_vision_slots.") ? .5f
                : name.starts_with("ip_adapter_slots.1.") || name.find(".ip_adapters.1.") != std::string::npos ? .75f : .25f;
            std::vector<float> values(ggml_nelements(tensor));
            ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
            for (auto value : values) check(value == expected, "Imported weight storage offset, file index or slot was corrupted");
        }
    }
}
void architecture() {
    String2TensorStorage metadata;
    const std::string prefix = "ip_vision_slots.0";
    auto& patch = metadata[prefix + ".vision_model.embeddings.patch_embedding.weight"];
    patch.n_dims = 4; patch.ne[0] = patch.ne[1] = 14; patch.ne[2] = 3;
    auto& projection = metadata[prefix + ".visual_projection.weight"];
    projection.n_dims = 2;
    for (const auto& [width, version, output] : std::vector<std::tuple<int, CLIPVersion, int>>{
            {1024, OPENAI_CLIP_VIT_L_14, 768}, {1280, OPEN_CLIP_VIT_H_14, 1024}, {1664, OPEN_CLIP_VIT_BIGG_14, 1280}}) {
        patch.ne[3] = projection.ne[0] = width; projection.ne[1] = output;
        CLIPVersion detected = OPENAI_CLIP_VIT_L_14; int32_t dim = 0;
        check(sd_ip_vision_architecture(metadata, prefix, detected, dim) && detected == version && dim == output,
            "Vision family or projection dimension was guessed incorrectly");
        CLIPVisionModelProjection model(detected, false, false, dim);
        check(model.hidden_size == width && model.projection_dim == output, "Vision runner ignored selected architecture");
        std::vector<GGMLBlock*> blocks; model.get_all_blocks(blocks);
        auto ctx = std::unique_ptr<ggml_context, decltype(&ggml_free)>(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
        auto cpu = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(ggml_backend_cpu_init(), ggml_backend_free);
        check(bool(ctx) && bool(cpu), "Cannot build vision activation contract graph");
        bool checked = false;
        for (auto* block : blocks) if (auto* mlp = dynamic_cast<CLIPMLP*>(block)) {
            mlp->init(ctx.get(), {}, "vision_mlp");
            GGMLRunnerContext runner; runner.backend = cpu.get(); runner.ggml_ctx = ctx.get();
            auto* input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, 1);
            auto* graph = ggml_new_graph(ctx.get());
            ggml_build_forward_expand(graph, mlp->forward(&runner, input));
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                const auto* node = ggml_graph_node(graph, i);
                if (node->op != GGML_OP_UNARY) continue;
                const auto op = ggml_get_unary_op(node);
                if (op == GGML_UNARY_OP_GELU || op == GGML_UNARY_OP_GELU_QUICK) {
                    check((op == GGML_UNARY_OP_GELU_QUICK) == (version == OPENAI_CLIP_VIT_L_14),
                        "Vision activation was selected from text width instead of model family");
                    checked = true;
                }
            }
            break;
        }
        check(checked, "Vision MLP activation was not found in actual graph");
    }
    CLIPVersion detected; int32_t dim;
    patch.ne[3] = 100;
    check(!sd_ip_vision_architecture(metadata, prefix, detected, dim), "Unknown vision architecture accepted");
    const auto key = "adapter.image_proj.norm.weight";
    metadata[key].n_dims = 1; metadata[key].ne[0] = 0;
    metadata["adapter.image_proj.proj.weight"].n_dims = 2;
    check(!sd_valid_ip_projection_metadata(metadata, "adapter"), "Zero-width projection could divide by zero");
    sd_ctx_params_t params; sd_ctx_params_init(&params);
    sd_ip_adapter_model_t invalid{nullptr, nullptr};
    check(!new_sd_ctx_with_ip_adapters(&params, &invalid, 1)
        && !new_sd_ctx_with_ip_adapters(&params, nullptr, 1)
        && !new_sd_ctx_with_ip_adapters(nullptr, &invalid, 1)
        && !new_sd_ctx_with_ip_adapters(&params, &invalid, 65), "Malformed public resource descriptor accepted");
}
int main(int argc, char** argv) {
    try {
        check(argc == 2, "Expected a build-local fixture directory");
        std::filesystem::create_directories(argv[1]);
        imported_resources(argv[1]); architecture();
        std::cout << "Indexed IP resource import, residency and architecture contracts passed\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
