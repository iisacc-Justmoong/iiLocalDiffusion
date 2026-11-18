#include "Generation/ImageModelRouting.hpp"
#include <model_loader.h>
#include <name_conversion.h>
#include <model/diffusion/anima.hpp>
#include <model_io/gguf_io.h>
#include <stable-diffusion.h>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

// A descriptor-only version of the observed 40-block Anima layout. The
// quantization marker and scales are valid but no learned weights are used.
void writeFixture(const fs::path &path, const std::string &prefix, bool conflict = false) {
    struct Tensor { std::string name, dtype, shape, bytes; };
    const std::string quant = R"({"format":"int8_tensorwise","orig_dtype":"torch.bfloat16","convrot":true,"convrot_groupsize":4,"per_row":true})";
    const std::string stem = prefix + "blocks.39.self_attn.q_proj";
    std::vector<Tensor> tensors{
        {prefix + "llm_adapter.blocks.0.cross_attn.q_proj.weight", "F32", "[1,1]", std::string(4, '\0')},
        {prefix + "blocks.0.self_attn.q_proj.weight", "F32", "[1,1]", std::string(4, '\0')},
        {stem + ".weight", "I8", "[2,4]", std::string(8, '\0')},
        {stem + ".weight_scale", "F32", "[2,1]", std::string(8, '\0')},
        {stem + ".comfy_quant", "U8", "[" + std::to_string(quant.size()) + "]", quant},
    };
    if (conflict) tensors.push_back({"model.diffusion_model.net.blocks.0.self_attn.q_proj.weight",
                                     "F32", "[1,1]", std::string(4, '\0')});
    std::string header = "{";
    std::size_t offset = 0;
    for (const auto &tensor : tensors) {
        if (offset) header += ',';
        header += '"' + tensor.name + "\":{\"dtype\":\"" + tensor.dtype + "\",\"shape\":" + tensor.shape
            + ",\"data_offsets\":[" + std::to_string(offset) + ',' + std::to_string(offset + tensor.bytes.size()) + "]}";
        offset += tensor.bytes.size();
    }
    header += '}';
    while (header.size() % 8) header += ' ';
    std::ofstream file(path, std::ios::binary);
    const auto size = static_cast<std::uint64_t>(header.size());
    for (unsigned shift = 0; shift < 64; shift += 8) file.put(static_cast<char>((size >> shift) & 0xff));
    file << header;
    for (const auto &tensor : tensors) file.write(tensor.bytes.data(), static_cast<std::streamsize>(tensor.bytes.size()));
    require(bool(file), "Could not write fixture");
}

struct EncoderTensor { std::string name; std::vector<std::uint64_t> shape; };

std::vector<EncoderTensor> encoderTensors(bool omitLayer) {
    std::vector<EncoderTensor> tensors{{"model.embed_tokens.weight", {151936, 1024}}, {"model.norm.weight", {1024}}};
    for (int layer = 0; layer < 28; ++layer) {
        const auto p = "model.layers." + std::to_string(layer) + '.';
        tensors.push_back({p + "input_layernorm.weight", {1024}});
        tensors.push_back({p + "post_attention_layernorm.weight", {1024}});
        tensors.push_back({p + "self_attn.q_proj.weight", {2048, 1024}});
        tensors.push_back({p + "self_attn.k_proj.weight", {1024, 1024}});
        tensors.push_back({p + "self_attn.v_proj.weight", {1024, 1024}});
        tensors.push_back({p + "self_attn.o_proj.weight", {1024, 2048}});
        if (!omitLayer || layer != 5) tensors.push_back({p + "self_attn.q_norm.weight", {128}});
        tensors.push_back({p + "self_attn.k_norm.weight", {128}});
        tensors.push_back({p + "mlp.gate_proj.weight", {3072, 1024}});
        tensors.push_back({p + "mlp.up_proj.weight", {3072, 1024}});
        tensors.push_back({p + "mlp.down_proj.weight", {1024, 3072}});
    }
    return tensors;
}

// Optional sparse metadata fixture for Qwen3-0.6B-Base. No weight payload is
// read; offsets describe the public 28-layer architecture on a sparse file.
void writeEncoderFixture(const fs::path &path, bool omitLayer) {
    const auto tensors = encoderTensors(omitLayer);
    std::string header = "{";
    std::uint64_t offset = 0;
    for (const auto &tensor : tensors) {
        if (offset) header += ',';
        std::uint64_t bytes = 2;
        header += '"' + tensor.name + "\":{\"dtype\":\"BF16\",\"shape\":[";
        for (std::size_t i = 0; i < tensor.shape.size(); ++i) {
            if (i) header += ',';
            header += std::to_string(tensor.shape[i]);
            bytes *= tensor.shape[i];
        }
        header += "],\"data_offsets\":[" + std::to_string(offset) + ',' + std::to_string(offset + bytes) + "]}";
        offset += bytes;
    }
    header += '}';
    while (header.size() % 8) header += ' ';
    std::ofstream file(path, std::ios::binary);
    const auto length = static_cast<std::uint64_t>(header.size());
    for (unsigned shift = 0; shift < 64; shift += 8) file.put(static_cast<char>((length >> shift) & 0xff));
    file << header;
    file.seekp(static_cast<std::streamoff>(8 + length + offset - 1));
    file.put('\0');
    require(bool(file), "Could not write sparse encoder fixture");
}

void writeGgufEncoderFixture(const fs::path &path) {
    std::vector<TensorWritePlan> plans;
    for (const auto &tensor : encoderTensors(false)) {
        TensorWritePlan plan;
        plan.name = tensor.name;
        if (plan.name == "model.embed_tokens.weight") plan.name = "token_embd.weight";
        else if (plan.name == "model.norm.weight") plan.name = "output_norm.weight";
        else {
            require(plan.name.starts_with("model.layers."), "Unexpected GGUF encoder fixture tensor");
            plan.name.replace(0, std::string("model.layers.").size(), "blk.");
            for (const auto &[original, gguf] : std::vector<std::pair<std::string, std::string>>{
                {"input_layernorm.", "attn_norm."}, {"post_attention_layernorm.", "ffn_norm."},
                {"self_attn.q_proj.", "attn_q."}, {"self_attn.k_proj.", "attn_k."},
                {"self_attn.v_proj.", "attn_v."}, {"self_attn.o_proj.", "attn_output."},
                {"self_attn.q_norm.", "attn_q_norm."}, {"self_attn.k_norm.", "attn_k_norm."},
                {"mlp.gate_proj.", "ffn_gate."}, {"mlp.up_proj.", "ffn_up."},
                {"mlp.down_proj.", "ffn_down."}}) {
                const auto position = plan.name.find(original);
                if (position != std::string::npos) { plan.name.replace(position, original.size(), gguf); break; }
            }
        }
        plan.type = tensor.shape.size() == 2 ? GGML_TYPE_Q8_0 : GGML_TYPE_F32;
        plan.n_dims = static_cast<int>(tensor.shape.size());
        for (std::size_t i = 0; i < tensor.shape.size(); ++i)
            plan.ne[i] = static_cast<std::int64_t>(tensor.shape[tensor.shape.size() - 1 - i]);
        plans.push_back(std::move(plan));
    }
    GGUFStreamingWriter writer;
    std::string error;
    require(writer.write_metadata(path.string(), plans, &error), "Could not write GGUF encoder metadata: " + error);
    const auto size = writer.file_size();
    writer.close();
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    file.seekp(static_cast<std::streamoff>(size - 1));
    file.put('\0');
    require(bool(file), "Could not extend sparse GGUF encoder fixture");
}
}

int main(int argc, char **argv) try {
    require(argc == 2, "Expected a fixture directory");
    const auto root = fs::absolute(argv[1]);
    fs::create_directories(root);
    const std::vector<std::string> prefixes{"", "net.", "model.diffusion_model.", "model.diffusion_model.net.",
        "diffusion_model.", "diffusion_model.net.", "transformer.", "transformer.net."};
    const std::string canonical = "model.diffusion_model.net.";
    int number = 0;
    for (const auto &prefix : prefixes) {
        for (const auto *suffix : {"llm_adapter.blocks.0.cross_attn.q_proj.weight", "blocks.39.self_attn.q_proj.weight",
                                   "blocks.39.self_attn.q_proj.weight_scale", "x_embedder.proj.1.weight"}) {
            require(convert_tensor_name(prefix + suffix, VERSION_ANIMA) == canonical + suffix,
                    "Anima name was not normalized: " + prefix + suffix);
        }
        const auto file = root / ("anima-" + std::to_string(number++) + ".safetensors");
        writeFixture(file, prefix);
        const auto inspection = iiLocalDiffusion::inspectImageModel(file);
        require(inspection.architecture == iiLocalDiffusion::ImageModelArchitecture::Anima
            && inspection.role == iiLocalDiffusion::ImageModelRole::Checkpoint
            && inspection.encoding == iiLocalDiffusion::ImageWeightEncoding::ComfyInt8,
            "Anima INT8 checkpoint was not recognized: " + prefix);
        require(!inspection.hasTextEncoder && !inspection.hasVae, "Standalone denoiser was reported as complete");
        ModelLoader loader;
        require(loader.init_from_file(file.string()), "Could not read Anima metadata");
        require(loader.get_sd_version() == VERSION_ANIMA, "Anima version was not detected");
        loader.convert_tensors_name();
        const auto &tensors = loader.get_tensor_storage_map();
        const auto item = tensors.find(canonical + "blocks.39.self_attn.q_proj.weight");
        require(item != tensors.end(), "Quantized Anima tensor was not mounted under the canonical name");
        require(item->second.is_int8_tensorwise && item->second.int8_convrot
            && item->second.int8_convrot_group_size == 4, "INT8 ConvRot metadata was lost");
        require(tensors.find(canonical + "blocks.39.self_attn.q_proj.weight_scale") != tensors.end(),
                "Anima per-row scale was not mounted");
        require(Anima::AnimaConfig::detect_from_weights(tensors, "model.diffusion_model.net").num_layers == 40,
                "Anima block count was not inferred after name conversion");
    }
    for (const auto *unrelated : {"backup.net.blocks.0.self_attn.q_proj.weight", "net.unrelated.weight",
                                  "text_encoders.llm.model.embed_tokens.weight", "first_stage_model.encoder.conv1.weight"})
        require(convert_tensor_name(unrelated, VERSION_ANIMA) == unrelated,
                std::string("Unrelated tensor name was changed: ") + unrelated);
    const auto ambiguous = root / "ambiguous-anima.safetensors";
    writeFixture(ambiguous, "net.", true);
    bool rejected = false;
    try { (void)iiLocalDiffusion::inspectImageModel(ambiguous); }
    catch (const std::exception &error) { rejected = std::string(error.what()).find("conflicting") != std::string::npos; }
    require(rejected, "Conflicting Anima aliases must be rejected before model loading");
    require(!sd_model_validate_anima_llm(nullptr) && !sd_model_validate_anima_llm("")
            && !sd_model_validate_anima_llm(ambiguous.string().c_str()),
            "A missing or incomplete Anima text encoder was accepted");
    const auto unconventional = root / "anima-weights.bin";
    writeFixture(unconventional, "net.");
    require(iiLocalDiffusion::inspectImageModel(unconventional).architecture == iiLocalDiffusion::ImageModelArchitecture::Anima,
            "Safetensors metadata was ignored because of its filename extension");
    for (const auto &file : {unconventional, root / "anima-weights.json"}) {
        if (file != unconventional) writeFixture(file, "net.");
        ModelLoader loader;
        require(loader.init_from_file(file.string()) && loader.get_sd_version() == VERSION_ANIMA,
                "Native loading did not detect the safetensors content: " + file.string());
    }
    const auto index = root / "weights.index.json";
    { std::ofstream file(index); file << R"({"weight_map":{"net.blocks.0.self_attn.q_proj.weight":"anima-weights.bin"}})"; }
    ModelLoader sharded;
    require(sharded.init_from_file(index.string()) && sharded.get_sd_version() == VERSION_ANIMA,
            "A genuine safetensors shard index stopped loading");
    const auto uppercaseIndex = root / "weights.SAFETENSORS.Index.JSON";
    { std::ofstream file(uppercaseIndex); file << R"({"weight_map":{"net.blocks.0.self_attn.q_proj.weight":"anima-weights.bin"}})"; }
    ModelLoader uppercaseSharded;
    require(uppercaseSharded.init_from_file(uppercaseIndex.string()) && uppercaseSharded.get_sd_version() == VERSION_ANIMA,
            "The native engine did not load an index with a mixed-case suffix");
    const auto contentIndex = root / "weights.index.data";
    { std::ofstream file(contentIndex); file << R"({"weight_map":{"net.blocks.0.self_attn.q_proj.weight":"anima-weights.bin"}})"; }
    ModelLoader contentSharded;
    require(contentSharded.init_from_file(contentIndex.string()) && contentSharded.get_sd_version() == VERSION_ANIMA,
            "The native engine did not load an index by content after its extension changed");
    const auto verifiedIndex=root/"verified-index.bin";
    { std::ofstream file(verifiedIndex); file << R"({"weight_map":{"net.llm_adapter.blocks.0.cross_attn.q_proj.weight":"anima-weights.bin","net.blocks.0.self_attn.q_proj.weight":"anima-weights.bin","net.blocks.39.self_attn.q_proj.weight":"anima-weights.bin","net.blocks.39.self_attn.q_proj.weight_scale":"anima-weights.bin","net.blocks.39.self_attn.q_proj.comfy_quant":"anima-weights.bin"}})"; }
    const auto verified=iiLocalDiffusion::inspectImageModel(verifiedIndex);
    require(verified.architecture==iiLocalDiffusion::ImageModelArchitecture::Anima &&
            verified.encoding==iiLocalDiffusion::ImageWeightEncoding::ComfyInt8,
            "A renamed shard index did not preserve Anima architecture and quantization metadata");
    const auto collidingHeader=root/"header-starts-with-brace.data";
    {
        std::ifstream in(unconventional,std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(in)),{});
        require(bytes.size()>=8,"The source header fixture is truncated");
        std::uint64_t length=0;
        for (unsigned i=0;i<8;++i) length|=std::uint64_t(static_cast<unsigned char>(bytes[i]))<<(8*i);
        require(length<=bytes.size()-8,"The source header fixture is invalid");
        const auto padding=(123+256-length%256)%256;
        const auto padded=length+padding;
        std::ofstream out(collidingHeader,std::ios::binary);
        for (unsigned i=0;i<8;++i) out.put(static_cast<char>((padded>>(8*i))&255));
        out.write(bytes.data()+8,static_cast<std::streamsize>(length));
        for (std::uint64_t i=0;i<padding;++i) out.put(' ');
        out.write(bytes.data()+8+length,static_cast<std::streamsize>(bytes.size()-8-length));
        require(bool(out),"Could not write a colliding Safetensors header fixture");
    }
    ModelLoader headerCollision;
    require(headerCollision.init_from_file(collidingHeader.string()) && headerCollision.get_sd_version()==VERSION_ANIMA,
            "A Safetensors length byte resembling JSON selected the index reader");
    const auto other = root / "unrelated.gguf";
    { std::ofstream file(other, std::ios::binary); file << "GGUF" << std::string(20, '\0'); }
    require(iiLocalDiffusion::inspectImageModel(other).architecture == iiLocalDiffusion::ImageModelArchitecture::Unknown,
            "An unrelated format was incorrectly classified as Anima");
    if (const char *sparse = std::getenv("IILD_TEST_SPARSE_ANIMA_ENCODER"); sparse && std::string(sparse) == "1") {
        const auto encoder = root / "qwen3-06b-sparse.safetensors";
        const auto incomplete = root / "qwen3-06b-missing.safetensors";
        const auto gguf = root / "qwen3-06b-q8.gguf";
        writeEncoderFixture(encoder, false);
        writeEncoderFixture(incomplete, true);
        writeGgufEncoderFixture(gguf);
        require(sd_model_validate_anima_llm(encoder.string().c_str()), "Complete Qwen3-0.6B metadata was rejected");
        require(!sd_model_validate_anima_llm(incomplete.string().c_str()), "Incomplete Qwen3-0.6B metadata was accepted");
        require(sd_model_validate_anima_llm(gguf.string().c_str()), "Q8 GGUF Qwen3-0.6B metadata was rejected");
        std::cout << "Sparse safetensors and Q8 GGUF encoder contracts verified without reading weights\n";
    }
    if (const char *path = std::getenv("IILD_TEST_ANIMA_LLM")) {
        const fs::path actual(path);
        const auto size = fs::file_size(actual);
        const auto modified = fs::last_write_time(actual);
        require(sd_model_validate_anima_llm(actual.string().c_str()),
                "Actual Qwen3-0.6B encoder metadata did not match the Anima tensor contract");
        require(fs::file_size(actual) == size && fs::last_write_time(actual) == modified,
                "Actual Anima text encoder was modified");
        std::cout << "Actual Qwen3-0.6B encoder metadata verified without loading weights\n";
    }
    if (const char *path = std::getenv("IILD_TEST_ANIMA_HEADER")) {
        const fs::path actual(path);
        const auto size = fs::file_size(actual);
        const auto modified = fs::last_write_time(actual);
        const auto inspection = iiLocalDiffusion::inspectImageModel(actual);
        require(inspection.architecture == iiLocalDiffusion::ImageModelArchitecture::Anima,
                "Actual file was not identified as Anima");
        sd_model_vae_info_t components{};
        bool embeddedEncoder = true;
        require(sd_model_inspect_vae(actual.string().c_str(), &components)
            && std::string(components.model_family) == "anima"
            && std::string(components.vae_family) == "qwen-image"
            && components.state == SD_VAE_MISSING
            && sd_model_has_embedded_llm(actual.string().c_str(), &embeddedEncoder)
            && !embeddedEncoder,
            "Actual Anima component requirements were not identified");
        if (const char *decoder = std::getenv("IILD_TEST_ANIMA_VAE")) {
            const fs::path vae(decoder);
            const auto vaeSize = fs::file_size(vae);
            const auto vaeModified = fs::last_write_time(vae);
            require(sd_model_validate_vae(actual.string().c_str(), vae.string().c_str()),
                    "Actual Qwen Image VAE metadata did not match the Anima tensor contract");
            require(fs::file_size(vae) == vaeSize && fs::last_write_time(vae) == vaeModified,
                    "Actual Anima VAE was modified");
            std::cout << "Actual Qwen Image VAE metadata verified without loading weights\n";
        }
        ModelLoader loader;
        require(loader.init_from_file(actual.string()), "Could not inspect actual Anima header");
        if (const char *encoder = std::getenv("IILD_TEST_ANIMA_LLM")) {
            if (const char *decoder = std::getenv("IILD_TEST_ANIMA_VAE")) {
                ModelLoader combined;
                require(combined.init_from_file(actual.string())
                    && combined.init_from_file(encoder, "text_encoders.llm.")
                    && combined.init_from_file(decoder, "vae.")
                    && combined.get_sd_version() == VERSION_ANIMA,
                    "Combined Anima components changed the detected model architecture");
                combined.convert_tensors_name();
                const auto &combinedStorage = combined.get_tensor_storage_map();
                require(combinedStorage.find(canonical + "llm_adapter.blocks.0.cross_attn.q_proj.weight") != combinedStorage.end()
                    && combinedStorage.find(canonical + "blocks.39.self_attn.q_proj.weight") != combinedStorage.end()
                    && !combined.get_diffusion_model_wtype_stat().empty(),
                    "Combined Anima components did not preserve the diffusion namespace");
                std::cout << "Actual combined model, encoder and VAE namespaces verified\n";
            }
        }
        loader.convert_tensors_name();
        const auto &storage = loader.get_tensor_storage_map();
        require(storage.find(canonical + "llm_adapter.blocks.0.cross_attn.q_proj.weight") != storage.end()
            && storage.find(canonical + "x_embedder.proj.1.weight") != storage.end(),
            "Actual Anima tensors were not mounted under the canonical names");
        const auto layers = Anima::AnimaConfig::detect_from_weights(storage, "model.diffusion_model.net").num_layers;
        require(layers > 0 && fs::file_size(actual) == size && fs::last_write_time(actual) == modified,
                "Actual model metadata was invalid or the source changed");
        std::cout << "Actual Anima header: " << layers << " diffusion blocks; encoding="
                  << static_cast<int>(inspection.encoding) << '\n';
    }
    std::cout << "Anima namespaces, 40-block metadata and INT8 ConvRot descriptors verified\n";
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n'; return 1;
}
