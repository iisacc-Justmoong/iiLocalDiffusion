#include "Generation/NativeDiffusion.hpp"
#include "Generation/ImageModelRouting.hpp"
#include "Generation/ImagePackageBridge.hpp"
#include "ModelPackaging/ModelPackaging.hpp"
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace iiLocalDiffusion;

static void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

static std::string read(const fs::path &path) {
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

static std::string bridgeResult(iild_image_package_result_v1 *result) {
    require(result != nullptr, "image package bridge returned no result");
    const auto *json = iild_image_package_json_v1(result);
    require(json != nullptr, "image package bridge returned no JSON");
    const std::string value(json);
    iild_image_package_free_v1(result);
    return value;
}

static void tensorFile(const fs::path &path, const std::vector<std::string> &keys) {
    fs::create_directories(path.parent_path());
    std::string header = "{";
    std::string payload;
    const std::string quant = R"({"format":"int8_tensorwise","orig_dtype":"torch.bfloat16","convrot":true,"convrot_groupsize":4,"per_row":true})";
    for (std::size_t i = 0; i < keys.size(); ++i) {
        if (i) header += ',';
        const bool marker = keys[i].ends_with(".comfy_quant");
        const bool int8 = keys[i].find("blocks.39.") != std::string::npos && keys[i].ends_with(".weight");
        const bool scale = keys[i].find("blocks.39.") != std::string::npos && keys[i].ends_with(".weight_scale");
        const auto bytes = marker ? quant : std::string(int8 || scale ? 8 : 4, '\0');
        const std::string dtype = marker ? "U8" : int8 ? "I8" : "F32";
        const std::string shape = marker ? "[" + std::to_string(bytes.size()) + "]"
            : int8 ? "[2,4]" : scale ? "[2,1]" : "[1]";
        header += '"' + keys[i] + "\":{\"dtype\":\"" + dtype + "\",\"shape\":" + shape + ",\"data_offsets\":["
            + std::to_string(payload.size()) + ',' + std::to_string(payload.size() + bytes.size()) + "]}";
        payload += bytes;
    }
    header += '}';
    while (header.size() % 8) header += ' ';
    std::ofstream file(path, std::ios::binary);
    const auto length = static_cast<std::uint64_t>(header.size());
    for (unsigned shift = 0; shift < 64; shift += 8) file.put(static_cast<char>((length >> shift) & 0xff));
    file << header << payload;
    require(static_cast<bool>(file), "fixture write failed");
}

static void mossShard(const fs::path &path, bool first) {
    fs::create_directories(path.parent_path());
    std::string header = first
        ? R"({"video_embedder.weight":{"dtype":"F32","shape":[4,2],"data_offsets":[0,32]}})"
        : R"({"final_linear_video.weight":{"dtype":"F32","shape":[2,4],"data_offsets":[0,32]},"layers.0.proj.qkv.weight":{"dtype":"F32","shape":[12,4],"data_offsets":[32,224]}})";
    while (header.size() % 8) header += ' ';
    std::ofstream file(path,std::ios::binary);
    const auto length=static_cast<std::uint64_t>(header.size());
    for (unsigned shift=0;shift<64;shift+=8) file.put(static_cast<char>((length>>shift)&0xff));
    file << header << std::string(first?32:224,'\0');
    require(static_cast<bool>(file), "MOSS shard fixture write failed");
}

static void writeLegacyPackageWithoutGroups(const fs::path &source,const fs::path &target) {
    const auto bytes=read(source);
    require(bytes.size()>=8, "package fixture is truncated");
    std::uint64_t length=0;
    for (unsigned i=0;i<8;++i) length|=std::uint64_t(static_cast<unsigned char>(bytes[i]))<<(8*i);
    require(length<=bytes.size()-8, "package header fixture is truncated");
    auto header=bytes.substr(8,static_cast<std::size_t>(length));
    const std::string needle="\\\"group\\\":\\\"";
    std::size_t removed=0;
    for (auto position=header.find(needle);position!=std::string::npos;position=header.find(needle)) {
        const auto end=header.find("\\\",",position+needle.size());
        require(end!=std::string::npos, "legacy group field is malformed");
        header.erase(position,end+3-position); ++removed;
    }
    require(removed>0, "package fixture has no group fields");
    while (!header.empty() && header.back()==' ') header.pop_back();
    while (header.size()%8) header.push_back(' ');
    std::ofstream output(target,std::ios::binary);
    const auto next=static_cast<std::uint64_t>(header.size());
    for (unsigned i=0;i<8;++i) output.put(static_cast<char>((next>>(8*i))&0xff));
    output << header; output.write(bytes.data()+8+length,static_cast<std::streamsize>(bytes.size()-8-length));
    require(static_cast<bool>(output), "legacy package fixture write failed");
}

int main(int argc, char **argv) {
    try {
        require(argc == 2, "expected workspace");
        fs::create_directories(argv[1]);
        const auto root = fs::canonical(argv[1]);
        {
            const auto direct=root/"standalone-shards";
            mossShard(direct/"part1.safetensors",true);
            mossShard(direct/"part2.safetensors",false);
            const auto index=direct/"model.safetensors.index.json";
            std::ofstream output(index);
            output << R"({"weight_map":{"video_embedder.weight":"part1.safetensors","final_linear_video.weight":"part2.safetensors","layers.0.proj.qkv.weight":"part2.safetensors"}})";
            output.close();
            const auto found=inspectImageModel(index);
            require(found.architecture==ImageModelArchitecture::MossImage && found.role==ImageModelRole::Checkpoint,
                    "Standalone MOSS shards must be classified together before native engine routing");
        }
        const auto source = root / "sources";
        const auto model = source / "Checkpoint/anima.safetensors";
        const auto encoder = source / "LLM/text.safetensors";
        const auto vae = source / "Other/qwen.safetensors";
        tensorFile(model, {"model.diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj.weight",
            "model.diffusion_model.blocks.0.self_attn.q_proj.weight"});
        tensorFile(encoder, {"model.embed_tokens.weight", "model.norm.weight", "model.layers.0.self_attn.q_proj.weight"});
        tensorFile(vae, {"encoder.conv1.weight", "encoder.head.2.weight", "decoder.conv1.weight", "decoder.head.2.weight"});
        const std::vector<iild::ModelPackagingAssignment> assignments{
            {"Checkpoint/anima.safetensors", "model", "main"},
            {"LLM/text.safetensors", "text_encoder", "component"},
            {"Other/qwen.safetensors", "video_vae", "component"}};
        const auto package = root / "anima-pack.safetensors";
        (void)iild::createModelPackageWithAssignments(source, package, assignments);
        require(inspectImageModel(package).encoding == ImageWeightEncoding::Floating,
            "Packaged floating-point Anima encoding was not recognized");
        const auto original = read(package);
        NativeGenerationRequest request; request.modelPath = package; request.prompt = "fixture";
        std::atomic_bool cancelled{false};
        bool preparedProgress = false;
        const auto prepared = prepareNativeImagePackage(request, root / "cache", {}, cancelled,
            [&](const auto &event) { preparedProgress |= event.stage == NativeGenerationStage::Preparing; });
        require(prepared.packaged, "Anima package was not recognized");
        require(prepared.request.modelPath == package, "root transformer was unnecessarily extracted");
        require(read(prepared.components.llm) == read(encoder), "original text encoder was not recovered");
        require(read(prepared.components.vae) == read(vae), "original Qwen VAE was not recovered");
        require(preparedProgress, "component preparation did not report progress");
        require(read(package) == original, "source package changed");
        const auto multiSource = root / "multi-anima-sources";
        tensorFile(multiSource / "one.safetensors", {
            "model.diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj.weight",
            "model.diffusion_model.blocks.0.self_attn.q_proj.weight"});
        tensorFile(multiSource / "two.safetensors", {
            "model.diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj.weight",
            "model.diffusion_model.blocks.0.self_attn.q_proj.weight", "other.weight"});
        const auto multiPackage = root / "multi-anima.safetensors";
        const std::vector<iild::ModelPackagingAssignment> multiAssignments{
            {"one.safetensors", "model", "main"}, {"two.safetensors", "model", "component"}};
        (void)iild::createModelPackageWithAssignments(multiSource, multiPackage, multiAssignments);
        const auto multiInspection = inspectEmbeddedImagePackage(multiPackage);
        require(multiInspection.components.packaged && multiInspection.catalog.groups.size() == 2 &&
            multiInspection.model.architecture == ImageModelArchitecture::Unknown,
            "Ambiguous Anima catalog lost selectable source groups");
        require(std::count_if(multiInspection.catalog.groups.begin(), multiInspection.catalog.groups.end(),
            [](const auto &group) { return group.userSelectedMain && group.path == "one.safetensors"; }) == 1,
            "Ambiguous Anima catalog lost the explicit main source");
        bool multiRejected = false;
        try { (void)inspectImageModel(multiPackage); }
        catch (const std::runtime_error &) { multiRejected = true; }
        require(multiRejected, "Direct native image inspection silently chose an ambiguous model");
        NativeModelComponents overrides; overrides.llm = encoder; overrides.vae = vae;
        const auto explicitComponents = prepareNativeImagePackage(request, root / "unused", overrides, cancelled, {});
        require(explicitComponents.components.llm == encoder && explicitComponents.components.vae == vae,
            "explicit components were overridden");
        require(!fs::exists(root / "unused"), "explicit components caused unnecessary extraction");
        NativeModelComponents onlyVae; onlyVae.vae = vae;
        const auto mixed = prepareNativeImagePackage(request, root / "mixed", onlyVae, cancelled, {});
        require(mixed.components.vae == vae && read(mixed.components.llm) == read(encoder),
            "explicit VAE and embedded encoder were not combined");
        const auto shardSource = root / "sharded-anima-sources";
        tensorFile(shardSource / "Checkpoint/anima-1.safetensors", {
            "model.diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj.weight"});
        tensorFile(shardSource / "Checkpoint/anima-2.safetensors", {
            "model.diffusion_model.blocks.0.self_attn.q_proj.weight"});
        tensorFile(shardSource / "LLM/part-1.safetensors", {"model.embed_tokens.weight"});
        tensorFile(shardSource / "LLM/part-2.safetensors", {"model.norm.weight", "model.layers.0.self_attn.q_proj.weight"});
        {
            std::ofstream index(shardSource / "Checkpoint/model.safetensors.index.json");
            index << R"({"weight_map":{"model.diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj.weight":"anima-1.safetensors","model.diffusion_model.blocks.0.self_attn.q_proj.weight":"anima-2.safetensors"}})";
            require(static_cast<bool>(index), "sharded transformer index write failed");
        }
        {
            std::ofstream index(shardSource / "LLM/model.safetensors.index.json");
            index << R"({"weight_map":{"model.embed_tokens.weight":"part-1.safetensors","model.norm.weight":"part-2.safetensors","model.layers.0.self_attn.q_proj.weight":"part-2.safetensors"}})";
            require(static_cast<bool>(index), "sharded encoder index write failed");
        }
        const std::vector<iild::ModelPackagingAssignment> shardAssignments{
            {"Checkpoint/anima-1.safetensors", "model", "main"},
            {"LLM/part-1.safetensors", "text_encoder", "component"}};
        const auto shardPackage = root / "sharded-anima.safetensors";
        (void)iild::createModelPackageWithAssignments(shardSource, shardPackage, shardAssignments);
        const auto shardAssets = inspectEmbeddedImageComponents(shardPackage);
        require(shardAssets.transformer == "Checkpoint/model.safetensors.index.json"
            && shardAssets.textEncoder == "LLM/model.safetensors.index.json",
            "Anima inspection did not recognize the indexed transformer and text encoder groups");
        auto shardRequest = request; shardRequest.modelPath = shardPackage;
        const auto shardPrepared = prepareNativeImagePackage(shardRequest, root / "sharded-cache", {}, cancelled, {});
        const auto combinedEncoder = read(shardPrepared.components.llm);
        const auto combinedModel = read(shardPrepared.request.modelPath);
        require(shardPrepared.packaged && shardPrepared.request.modelPath != shardPackage
            && combinedModel.find("model.diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj.weight") != std::string::npos
            && combinedModel.find("model.diffusion_model.blocks.0.self_attn.q_proj.weight") != std::string::npos
            && combinedEncoder.find("model.embed_tokens.weight") != std::string::npos
            && combinedEncoder.find("model.norm.weight") != std::string::npos
            && combinedEncoder.find("model.layers.0.self_attn.q_proj.weight") != std::string::npos,
            "Anima did not prepare all indexed text encoder shards");
        const auto renamedShardSource=root/"renamed-sharded-anima-sources";
        fs::copy(shardSource,renamedShardSource,fs::copy_options::recursive);
        fs::rename(renamedShardSource/"Checkpoint/model.safetensors.index.json",renamedShardSource/"Checkpoint/model-index.bin");
        fs::rename(renamedShardSource/"LLM/model.safetensors.index.json",renamedShardSource/"LLM/encoder-index.gguf");
        const auto renamedShardPackage=root/"renamed-sharded-anima.safetensors";
        (void)iild::createModelPackageWithAssignments(renamedShardSource,renamedShardPackage,shardAssignments);
        const auto renamedShardAssets=inspectEmbeddedImageComponents(renamedShardPackage);
        require(renamedShardAssets.transformer=="Checkpoint/model-index.bin" &&
                renamedShardAssets.textEncoder=="LLM/encoder-index.gguf",
                "A renamed index was not recognized as an Anima component group");
        auto renamedRequest=request;renamedRequest.modelPath=renamedShardPackage;
        const auto renamedPrepared=prepareNativeImagePackage(renamedRequest,root/"renamed-sharded-cache",{},cancelled,{});
        require(renamedPrepared.packaged && renamedPrepared.request.modelPath!=renamedShardPackage &&
                read(renamedPrepared.request.modelPath).find("model.diffusion_model.blocks.0.self_attn.q_proj.weight")!=std::string::npos &&
                read(renamedPrepared.components.llm).find("model.layers.0.self_attn.q_proj.weight")!=std::string::npos,
                "A renamed index did not materialize its Anima model and encoder shards");
        const auto legacyShardPackage=root / "legacy-sharded-anima.safetensors";
        writeLegacyPackageWithoutGroups(shardPackage,legacyShardPackage);
        const auto legacyAssets=inspectEmbeddedImageComponents(legacyShardPackage);
        require(legacyAssets.transformer=="Checkpoint/model.safetensors.index.json" &&
            legacyAssets.textEncoder=="LLM/model.safetensors.index.json", "Legacy indexed groups were not recovered");
        auto legacyRequest=request; legacyRequest.modelPath=legacyShardPackage;
        const auto legacyPrepared=prepareNativeImagePackage(legacyRequest,root / "legacy-sharded-cache",{},cancelled,{});
        require(legacyPrepared.packaged && legacyPrepared.request.modelPath!=legacyShardPackage &&
            read(legacyPrepared.components.llm).find("model.layers.0.self_attn.q_proj.weight")!=std::string::npos,
            "Legacy indexed encoder group was not verified and prepared");
        auto damagedLegacy=read(legacyShardPackage);
        const std::string indexText=R"({"weight_map":{"model.embed_tokens.weight":"part-1.safetensors","model.norm.weight":"part-2.safetensors","model.layers.0.self_attn.q_proj.weight":"part-2.safetensors"}})";
        const auto damagedPosition=damagedLegacy.find(indexText);
        require(damagedPosition!=std::string::npos, "legacy index payload not found");
        damagedLegacy[damagedPosition+1]^=1;
        const auto damagedPackage=root / "damaged-legacy-shards.safetensors";
        { std::ofstream output(damagedPackage,std::ios::binary); output << damagedLegacy; }
        bool rejectedIndex=false;
        try { (void)inspectEmbeddedImageComponents(damagedPackage); }
        catch (const std::exception &) { rejectedIndex=true; }
        require(rejectedIndex, "Damaged legacy shard index was not rejected");
        // A package may deduplicate a whole component shard against tensors
        // already embedded in the main model. Its index still refers to the
        // original shard and the remaining shards must not be inspected alone.
        const auto duplicateSource = root / "sharded-duplicate-vae-sources";
        tensorFile(duplicateSource / "Checkpoint/anima.safetensors", {
            "model.diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj.weight",
            "model.diffusion_model.blocks.0.self_attn.q_proj.weight", "vae.encoder.conv1.weight"});
        tensorFile(duplicateSource / "VAE/part-1.safetensors", {"encoder.conv1.weight"});
        tensorFile(duplicateSource / "VAE/part-2.safetensors", {
            "encoder.head.2.weight", "decoder.conv1.weight", "decoder.head.2.weight"});
        {
            std::ofstream index(duplicateSource / "VAE/model.safetensors.index.json");
            index << R"({"weight_map":{"encoder.conv1.weight":"part-1.safetensors","encoder.head.2.weight":"part-2.safetensors","decoder.conv1.weight":"part-2.safetensors","decoder.head.2.weight":"part-2.safetensors"}})";
            require(static_cast<bool>(index), "duplicate VAE index write failed");
        }
        const auto duplicatePackage = root / "sharded-duplicate-vae.safetensors";
        const std::vector<iild::ModelPackagingAssignment> duplicateAssignments{
            {"Checkpoint/anima.safetensors", "model", "main"},
            {"VAE/part-1.safetensors", "video_vae", "component"}};
        (void)iild::createModelPackageWithAssignments(duplicateSource, duplicatePackage, duplicateAssignments);
        const auto duplicateAssets = inspectEmbeddedImageComponents(duplicatePackage);
        require(duplicateAssets.vae == "VAE/model.safetensors.index.json",
            "Indexed VAE containing an embedded duplicate shard was not recognized");
        const auto duplicateVae = iild::materializeModelPackageAssetByPath(
            duplicatePackage, root / "duplicate-vae-cache", duplicateAssets.vae);
        const auto duplicateVaeBytes = read(duplicateVae);
        require(duplicateVaeBytes.find("encoder.conv1.weight") != std::string::npos &&
                duplicateVaeBytes.find("decoder.head.2.weight") != std::string::npos,
            "Indexed VAE with a duplicate shard was not reconstructed completely");
        const auto duplicateLegacy = root / "legacy-sharded-duplicate-vae.safetensors";
        writeLegacyPackageWithoutGroups(duplicatePackage, duplicateLegacy);
        require(inspectEmbeddedImageComponents(duplicateLegacy).vae == "VAE/model.safetensors.index.json",
            "Legacy indexed VAE containing a duplicate shard was not recognized");
        const auto mossSource = root / "sharded-moss-sources";
        mossShard(mossSource / "Checkpoint/moss-1.safetensors",true);
        mossShard(mossSource / "Checkpoint/moss-2.safetensors",false);
        tensorFile(mossSource / "LLM/part-1.safetensors", {"model.language_model.embed_tokens.weight"});
        tensorFile(mossSource / "LLM/part-2.safetensors", {"model.language_model.layers.0.weight"});
        {
            std::ofstream index(mossSource / "Checkpoint/model.safetensors.index.json");
            index << R"({"weight_map":{"video_embedder.weight":"moss-1.safetensors","final_linear_video.weight":"moss-2.safetensors","layers.0.proj.qkv.weight":"moss-2.safetensors"}})";
            require(static_cast<bool>(index), "MOSS transformer index write failed");
        }
        {
            std::ofstream index(mossSource / "LLM/model.safetensors.index.json");
            index << R"({"weight_map":{"model.language_model.embed_tokens.weight":"part-1.safetensors","model.language_model.layers.0.weight":"part-2.safetensors"}})";
            require(static_cast<bool>(index), "MOSS encoder index write failed");
        }
        const auto mossPackage=root / "sharded-moss.safetensors";
        const std::vector<iild::ModelPackagingAssignment> mossAssignments{
            {"Checkpoint/moss-1.safetensors","model","main"}, {"LLM/part-1.safetensors","text_encoder","component"}};
        (void)iild::createModelPackageWithAssignments(mossSource,mossPackage,mossAssignments);
        const auto mossInspection=inspectImageModel(mossPackage);
        const auto mossAssets=inspectEmbeddedImageComponents(mossPackage);
        require(mossInspection.architecture==ImageModelArchitecture::MossImage &&
            mossInspection.encoding==ImageWeightEncoding::Floating &&
            mossAssets.transformer=="Checkpoint/model.safetensors.index.json" &&
            mossAssets.textEncoder=="LLM/model.safetensors.index.json", "MOSS indexed groups were not recognized");
        const auto mossModel=iild::materializeModelPackageAssetByPath(mossPackage,root / "moss-sharded-cache",mossAssets.transformer);
        const auto mossEncoder=iild::materializeModelPackageAssetByPath(mossPackage,root / "moss-sharded-cache",mossAssets.textEncoder);
        require(read(mossModel).find("layers.0.proj.qkv.weight")!=std::string::npos &&
            read(mossEncoder).find("model.language_model.layers.0.weight")!=std::string::npos,
            "MOSS indexed groups were not prepared");
        const auto legacyMossPackage=root / "legacy-sharded-moss.safetensors";
        writeLegacyPackageWithoutGroups(mossPackage,legacyMossPackage);
        const auto legacyMoss=inspectImageModel(legacyMossPackage);
        const auto legacyMossAssets=inspectEmbeddedImageComponents(legacyMossPackage);
        require(legacyMoss.architecture==ImageModelArchitecture::MossImage &&
            legacyMossAssets.transformer=="Checkpoint/model.safetensors.index.json" &&
            legacyMossAssets.textEncoder=="LLM/model.safetensors.index.json",
            "Legacy MOSS indexed groups were not recovered");
        const auto legacyCatalog=inspectEmbeddedImageCatalog(legacyMossPackage);
        const auto modelGroup=std::find_if(legacyCatalog.groups.begin(),legacyCatalog.groups.end(),[](const auto &group) {
            return group.path=="Checkpoint/model.safetensors.index.json";
        });
        require(legacyCatalog.packaged && modelGroup!=legacyCatalog.groups.end() && modelGroup->userSelectedMain &&
            std::any_of(modelGroup->tensorMappings.begin(),modelGroup->tensorMappings.end(),[](const auto &mapping) {
                return mapping.first=="layers.0.proj.qkv.weight" && !mapping.second.empty();
            }), "Legacy catalog did not preserve indexed original tensor mappings");
        const auto bridgeInspection=bridgeResult(iild_image_package_inspect_v1(legacyMossPackage.string().c_str()));
        if (bridgeInspection.find("\"ok\":true")==std::string::npos ||
            bridgeInspection.find("\"architecture\":\"moss-image\"")==std::string::npos ||
            bridgeInspection.find("\"transformer\":\"Checkpoint")==std::string::npos ||
            bridgeInspection.find("\"text_encoder\":\"LLM")==std::string::npos ||
            bridgeInspection.find("model.safetensors.index.json")==std::string::npos)
            throw std::runtime_error("C package bridge did not preserve the verified MOSS route: "+bridgeInspection);
        const auto bridgeCatalog=bridgeResult(iild_image_package_catalog_v1(legacyMossPackage.string().c_str()));
        require(bridgeCatalog.find("\"ok\":true")!=std::string::npos &&
            bridgeCatalog.find("\"groups\"")!=std::string::npos &&
            bridgeCatalog.find("\"user_selected_main\":true")!=std::string::npos &&
            bridgeCatalog.find("layers.0.proj.qkv.weight")!=std::string::npos,
            "C package catalog did not expose original indexed tensor mappings");
        int progressCalls=0;
        const auto observe=[](const char *,std::uint64_t,std::uint64_t,void *user) -> int {
            ++*static_cast<int *>(user);return 0;
        };
        const auto bridgePrepared=bridgeResult(iild_image_package_materialize_v1(
            legacyMossPackage.string().c_str(),(root / "bridge-cache").string().c_str(),
            legacyMossAssets.transformer.c_str(),observe,&progressCalls));
        require(bridgePrepared.find("\"ok\":true")!=std::string::npos && progressCalls>0,
            "C package bridge did not materialize or report progress");
        const auto cancel=[](const char *,std::uint64_t,std::uint64_t,void *) -> int {return 1;};
        const auto canceledBridge=bridgeResult(iild_image_package_materialize_v1(
            legacyMossPackage.string().c_str(),(root / "canceled-bridge-cache").string().c_str(),
            legacyMossAssets.transformer.c_str(),cancel,nullptr));
        require(canceledBridge.find("\"ok\":false")!=std::string::npos &&
            !fs::exists(root / "canceled-bridge-cache"), "C package bridge ignored cancellation");
        require(bridgeResult(iild_image_package_inspect_v1(nullptr)).find("\"ok\":false")!=std::string::npos,
            "C package bridge accepted a missing path");
        const auto nestedPackage = root / "nested-anima.safetensors";
        const std::vector<iild::ModelPackagingAssignment> nestedAssignments{
            {"Checkpoint/anima.safetensors", "model", "component"},
            {"LLM/text.safetensors", "model", "main"},
            {"Other/qwen.safetensors", "video_vae", "component"}};
        (void)iild::createModelPackageWithAssignments(source, nestedPackage, nestedAssignments);
        auto nestedRequest = request; nestedRequest.modelPath = nestedPackage;
        const auto nested = prepareNativeImagePackage(nestedRequest, root / "nested", {}, cancelled, {});
        require(nested.packaged && nested.request.modelPath != nestedPackage
            && read(nested.request.modelPath) == read(model) && read(nested.components.llm) == read(encoder)
            && read(nested.components.vae) == read(vae), "nested transformer was not reconstructed");
        bool stoppedDuringCopy = false;
        try {
            (void)prepareNativeImagePackage(request, root / "cancel-midway", {}, cancelled,
                [&](const auto &) { cancelled = true; });
        } catch (const std::exception &) { stoppedDuringCopy = true; }
        require(stoppedDuringCopy, "preparation did not stop when cancellation arrived");
        cancelled = false;
        const auto ordinary = prepareNativeImagePackage(NativeGenerationRequest{model}, root / "ordinary", {}, cancelled, {});
        require(!ordinary.packaged && ordinary.request.modelPath == model && !fs::exists(root / "ordinary"),
            "a standalone checkpoint was incorrectly treated as an embedded package");
        const auto netSource = root / "net-sources";
        const auto netModel = netSource / "Checkpoint/anima-net.safetensors";
        tensorFile(netModel, {"net.llm_adapter.blocks.0.cross_attn.q_proj.weight",
                              "net.blocks.0.self_attn.q_proj.weight", "net.blocks.39.self_attn.q_proj.weight",
                              "net.blocks.39.self_attn.q_proj.weight_scale", "net.blocks.39.self_attn.q_proj.comfy_quant"});
        tensorFile(netSource / "LLM/text.safetensors", {"model.embed_tokens.weight", "model.norm.weight",
                   "model.layers.0.self_attn.q_proj.weight"});
        tensorFile(netSource / "Other/qwen.safetensors", {"encoder.conv1.weight", "encoder.head.2.weight",
                   "decoder.conv1.weight", "decoder.head.2.weight"});
        const auto netPackage = root / "anima-net-pack.safetensors";
        const std::vector<iild::ModelPackagingAssignment> netAssignments{
            {"Checkpoint/anima-net.safetensors", "model", "main"},
            {"LLM/text.safetensors", "text_encoder", "component"},
            {"Other/qwen.safetensors", "video_vae", "component"}};
        (void)iild::createModelPackageWithAssignments(netSource, netPackage, netAssignments);
        require(inspectImageModel(netPackage).encoding == ImageWeightEncoding::ComfyInt8,
            "Packaged Anima lost its INT8 transformer encoding");
        auto netRequest = request; netRequest.modelPath = netPackage;
        const auto netPrepared = prepareNativeImagePackage(netRequest, root / "net-cache", {}, cancelled, {});
        require(netPrepared.packaged && read(netPrepared.components.llm) == read(encoder)
            && read(netPrepared.components.vae) == read(vae), "Bare net Anima package components were not recognized");
        const auto conflictSource = root / "conflicting-sources";
        tensorFile(conflictSource / "anima.safetensors", {"net.llm_adapter.blocks.0.cross_attn.q_proj.weight",
            "net.blocks.0.self_attn.q_proj.weight", "model.diffusion_model.net.blocks.0.self_attn.q_proj.weight"});
        const auto conflictPackage = root / "conflicting-anima-pack.safetensors";
        const std::vector<iild::ModelPackagingAssignment> conflictAssignments{{"anima.safetensors", "model", "main"}};
        (void)iild::createModelPackageWithAssignments(conflictSource, conflictPackage, conflictAssignments);
        bool conflictRejected = false;
        try { (void)inspectImageModel(conflictPackage); }
        catch (const std::exception &error) { conflictRejected = std::string(error.what()).find("conflicting") != std::string::npos; }
        require(conflictRejected, "Conflicting aliases in an Anima package were not rejected");
        auto control = std::make_shared<NativeExecutionControl>(); control->setPaused(true);
        auto paused = std::async(std::launch::async, [&] {
            try { (void)prepareNativeImagePackage(request, root / "paused", {}, cancelled, {}, control); return false; }
            catch (const std::exception &) { return true; }
        });
        for (int i = 0; i < 100 && !control->isWaiting(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        const auto wasWaiting = control->isWaiting(); cancelled = true;
        require(paused.get() && wasWaiting && !fs::exists(root / "paused"), "paused preparation did not cancel safely");
        bool interrupted = false;
        try { (void)prepareNativeImagePackage(request, root / "cancelled", {}, cancelled, {}); }
        catch (const std::exception &) { interrupted = true; }
        require(interrupted && !fs::exists(root / "cancelled"), "cancellation was not respected");
        if (const auto *actual = std::getenv("IILD_TEST_ANIMA_PACKAGE")) {
            const auto actualPath = fs::canonical(actual);
            const auto inspection = inspectImageModel(actualPath);
            const auto assets = inspectEmbeddedImageComponents(actualPath);
            require(inspection.architecture == ImageModelArchitecture::Anima && assets.packaged
                && !assets.transformer.empty() && !assets.textEncoder.empty() && !assets.vae.empty(),
                "actual package metadata did not identify all Anima components");
        }
        std::cout << "Anima package component binding verified\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
