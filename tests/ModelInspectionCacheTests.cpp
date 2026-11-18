#include <ModelPackaging/ModelInspection.hpp>
#include <ModelPackaging/ModelPackaging.hpp>
#include <iiFileProviderNative.h>
#include <iiFileProviderPresentation.h>
#include <array>
#include <filesystem>
#include <iostream>
#include <stdexcept>

void require(bool condition, const char *message) { if (!condition) throw std::runtime_error(message); }
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    namespace fs = std::filesystem;
    using iiFileProvider::native::File;
    try {
        const fs::path root = fs::absolute(argv[1]);
        File::createDirectories(root);
        const auto model = root / "model.safetensors";
        const std::string header = R"({"__metadata__":{"config":"{\"_class_name\":\"LTX2Pipeline\"}"}})";
        std::string bytes(8, '\0');
        for (unsigned i = 0; i < 8; ++i) bytes[i] = static_cast<char>((header.size() >> (i * 8)) & 255);
        File::write(model, bytes + header);
        const auto renamed = root / "weights.bin";
        File::write(renamed, bytes + header);
        const auto gguf = root / "model.data";
        File::write(gguf, "GGUF-test");
        const auto opaque = root / "tokenizer.bin";
        File::write(opaque, "not model weights");
        require(iild::detectModelContainer(renamed) == "safetensors", "Renamed Safetensors prefix was not detected");
        require(iild::detectModelContainer(gguf) == "gguf", "Renamed GGUF prefix was not detected");
        require(iild::detectModelContainer(opaque) == "unknown", "Opaque asset was identified as a model");
        require(iild::detectModelContainer(root / "missing") == "unknown", "Missing file was identified as a model");
        require(iiFileProvider::native::Presentation::waitForIdle(std::chrono::seconds(5)), "Fixture publication must finish");
        auto first = iild::inspectModel(model);
        require(first.recognized && !first.cacheHit && first.metadataBytesRead > 0, "First inspection must read architecture metadata");
        auto cached = iild::inspectModel(model);
        require(cached.cacheHit && cached.metadataBytesRead == 0, "Repeated polling must not reread a large header");
        iild::setModelModalities(model, {"image", "video"});
        auto manual = iild::inspectModel(model);
        require(!manual.cacheHit && manual.userOverride && manual.modalities == std::vector<std::string>{"image", "video"}, "Sidecar edits must invalidate discovery");
        iild::setModelModalities(model, {});
        auto automatic = iild::inspectModel(model);
        require(!automatic.userOverride && automatic.modalities == first.modalities, "Automatic inspection must restore actual output modalities");
        const auto zipCandidate=[&](const fs::path &path,const std::string &entry) {
            std::string header(30,'\0');
            header[0]='P';header[1]='K';header[2]=3;header[3]=4;
            header[26]=static_cast<char>(entry.size()&255);
            header[27]=static_cast<char>((entry.size()>>8)&255);
            File::write(path,header+entry+"fixture");
        };
        const auto legacyZip=root/"torch-with-unexpected-name.data";
        zipCandidate(legacyZip,"model/data.pkl");
        require(iild::detectModelContainer(legacyZip)=="pytorch","PyTorch ZIP candidate was not detected by content");
        const auto legacyPickle=root/"torch-with-safe-extension.safetensors";
        File::write(legacyPickle,std::string("\x80\x02\x8a\x0a\x6c\xfc\x9c\x46\xf9\x20\x6a\xa8\x50\x19\x2e",15)+"fixture");
        require(iild::detectModelContainer(legacyPickle)=="pytorch","Pickle candidate did not take precedence over filename");
        auto unclassified=iild::inspectModel(legacyZip);
        require(!unclassified.recognized && unclassified.modalities.empty()
                && unclassified.diagnostic.find("PyTorch")!=std::string::npos,
                "PyTorch discovery must defer architecture inspection to the safe worker");
        iild::setModelModalities(legacyZip,{"video"});
        require(iild::inspectModel(legacyZip).modalities==std::vector<std::string>{"video"},
                "PyTorch candidate ignored its explicit Society modality");
        const auto otherZip=root/"unrelated.zip";
        zipCandidate(otherZip,"archive/readme.txt");
        require(iild::detectModelContainer(otherZip)=="unknown","An unrelated ZIP was labelled as PyTorch");
        const auto otherPickle=root/"unrelated-pickle.data";
        File::write(otherPickle,std::string("\x80\x04",2)+"fixture");
        require(iild::detectModelContainer(otherPickle)=="unknown","An unrelated pickle was labelled as a PyTorch checkpoint");
        const auto index = root / "model_index.json";
        File::write(index, R"({"_class_name":"LTXPipeline"})");
        require(iild::inspectModel(root).modalities == std::vector<std::string>{"video"}, "Legacy pipeline discovery");
        File::write(index, R"({"_class_name":"StableDiffusionPipeline"})");
        require(iild::inspectModel(root).modalities == std::vector<std::string>{"image"}, "Directory configuration edits invalidate cached classification");
        const auto shards=root/"shards";
        File::createDirectories(shards);
        const auto makeShard=[&](const fs::path &path,const std::string &name) {
            const auto text="{\""+name+"\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
            std::string encoded(8,'\0');
            for (unsigned i=0;i<8;++i) encoded[i]=static_cast<char>((text.size()>>(8*i))&255);
            File::write(path,encoded+text+std::string(4,'\0'));
        };
        makeShard(shards/"first.safetensors","audio_vae.decoder.weight");
        makeShard(shards/"second.safetensors","model.diffusion_model.transformer_blocks.0.weight");
        makeShard(shards/"third.safetensors","text_embedding_projection.audio_aggregate_embed.weight");
        const auto sharded=shards/"model.safetensors.index.json";
        File::write(sharded,R"({"weight_map":{"audio_vae.decoder.weight":"first.safetensors","model.diffusion_model.transformer_blocks.0.weight":"second.safetensors","text_embedding_projection.audio_aggregate_embed.weight":"third.safetensors"}})");
        const auto inspected=iild::inspectSafetensorsIndex(sharded);
        require(inspected.shards.size()==3 && inspected.tensors.size()==3,"Index did not retain its exact shard catalog");
        constexpr std::array<std::string_view,2> required{"audio_vae.decoder.weight","model.diffusion_model.transformer_blocks.0.weight"};
        constexpr std::array<std::string_view,2> absent{"audio_vae.decoder.weight","missing.weight"};
        require(iild::modelContainsTensorNames(sharded,required),"Tensor probe must combine all indexed shards");
        require(!iild::modelContainsTensorNames(sharded,absent),"Tensor probe accepted a missing name");
        require(iild::findModelTensorNames(sharded,absent)==std::vector<std::string>{"audio_vae.decoder.weight"},
                "Tensor probe must return the matching subset");
        require(iild::detectModelContainer(sharded)=="safetensors-index","Valid shard index was not detected");
        const auto mixedCase=shards/"mixed.SAFETENSORS.Index.JSON";
        File::write(mixedCase,File::read(sharded,65536));
        require(iild::detectModelContainer(mixedCase)=="safetensors-index","Mixed-case shard index was not detected");
        require(iild::modelContainsTensorNames(mixedCase,required),"Mixed-case shard index tensor probe failed");
        require(iild::inspectModel(mixedCase).modalities==std::vector<std::string>({"audio","video"}),
                "Mixed-case index was not classified from its shards");
        require(iild::inspectModel(mixedCase).cacheHit,"Unchanged mixed-case index did not reuse its inspection");
        const auto renamedIndex=shards/"weights.data";
        File::write(renamedIndex,File::read(sharded,65536));
        require(iild::detectModelContainer(renamedIndex)=="safetensors-index","Content-based index detection depended on its filename");
        require(iild::modelContainsTensorNames(renamedIndex,required),"Renamed index tensor probe failed");
        require(iild::inspectModel(renamedIndex).modalities==std::vector<std::string>({"audio","video"}),
                "Renamed index was not classified from its shards");
        require(iild::inspectModel(renamedIndex).cacheHit,"Unchanged renamed index did not reuse its inspection");
        const auto torchShard=shards/"torch-shard.bin",torchShard2=shards/"torch-shard2.bin",torchIndex=shards/"torch-index.bin";
        const auto torchFixture=std::string("\x80\x02\x8a\x0a\x6c\xfc\x9c\x46\xf9\x20\x6a\xa8\x50\x19\x2e",15)+"fixture";
        File::write(torchShard,torchFixture);File::write(torchShard2,torchFixture);
        File::write(torchIndex,R"({"weight_map":{"audio_vae.decoder.weight":"torch-shard.bin","model.diffusion_model.transformer_blocks.0.weight":"torch-shard2.bin","text_embedding_projection.audio_aggregate_embed.weight":"torch-shard2.bin"}})");
        require(iild::detectModelContainer(torchIndex)=="pytorch-index","A PyTorch shard index was not detected by content");
        const auto torchGroup=iild::inspectPyTorchIndex(torchIndex);
        require(torchGroup.shards.size()==2 && torchGroup.tensors.size()==3 && iild::verifyPyTorchIndex(torchGroup),
                "PyTorch shard index did not bind its exact logical members");
        require(iild::inspectModel(torchIndex).modalities==std::vector<std::string>({"audio","video"}),
                "PyTorch index tensor names were not used for modality discovery");
        const auto misleadingTorchIndex=shards/"torch.safetensors.index.json";
        File::write(misleadingTorchIndex,File::read(torchIndex,65536));
        require(iild::detectModelContainer(misleadingTorchIndex)=="pytorch-index",
                "A Safetensors index suffix overrode the actual PyTorch shards");
        const auto mixedIndex=shards/"mixed-index.data";
        File::write(mixedIndex,R"({"weight_map":{"weight":"torch-shard.bin","other":"first.safetensors"}})");
        require(iild::detectModelContainer(mixedIndex)=="unknown","Mixed-format shards were accepted as one group");
        File::write(torchShard,torchFixture+"changed");
        require(!iild::verifyPyTorchIndex(torchGroup),"A changed PyTorch shard retained its captured identity");
        File::write(torchShard,torchFixture);
        const auto invalidIndex=shards/"traversal.data";
        File::write(invalidIndex,R"({"weight_map":{"weight":"../outside.safetensors"}})");
        require(iild::detectModelContainer(invalidIndex)=="unknown","An unsafe index was identified as a usable model");
        const auto shardedVideo=iild::inspectModel(sharded);
        require(shardedVideo.recognized && shardedVideo.modalities==std::vector<std::string>({"audio","video"}),
                "The index must classify tensor signatures across all shards");
        require(iild::inspectModel(sharded).cacheHit,"Unchanged shard model did not reuse its inspection");
        makeShard(shards/"first.safetensors","different.weight");
        const auto mixedChanged=iild::inspectModel(mixedCase);
        require(!mixedChanged.cacheHit && !mixedChanged.diagnostic.empty(),"A changed shard reused a mixed-case index inspection");
        const auto renamedChanged=iild::inspectModel(renamedIndex);
        require(!renamedChanged.cacheHit && !renamedChanged.diagnostic.empty(),"A changed shard reused a renamed index inspection");
        const auto changed=iild::inspectModel(sharded);
        require(!changed.cacheHit && !changed.diagnostic.empty(),"A changed shard reused a stale inspection");
        bool invalid=false;
        try {(void)iild::inspectSafetensorsIndex(sharded);} catch (const std::exception &) {invalid=true;}
        require(invalid,"A shard missing its indexed tensor was accepted");
        makeShard(shards/"target-one.safetensors","linked.weight");
        makeShard(shards/"target-two.safetensors","linked.weight");
        std::error_code linkError;
        fs::create_symlink("target-one.safetensors",shards/"linked.safetensors",linkError);
        if (!linkError) {
            File::write(sharded,R"({"weight_map":{"linked.weight":"linked.safetensors"}})");
            const auto linked=iild::inspectSafetensorsIndex(sharded);
            require(iild::verifySafetensorsIndex(linked),"Valid linked shard was rejected");
            fs::remove(shards/"linked.safetensors");
            fs::create_symlink("target-two.safetensors",shards/"linked.safetensors");
            require(!iild::verifySafetensorsIndex(linked),"A retargeted shard link retained its identity");
            const auto snapshot=root/"snapshot",blobs=root/"blobs";
            File::createDirectories(snapshot);File::createDirectories(blobs);
            const auto blob=blobs/"index-hash",blob2=blobs/"replacement-hash";
            File::write(blob,R"({"weight_map":{"linked.weight":"linked.safetensors"}})");
            File::write(blob2,File::read(blob,65536));
            const auto redirected=snapshot/"model.safetensors.index.json";
            fs::create_symlink(blob,redirected);
            fs::create_symlink(shards/"target-one.safetensors",snapshot/"linked.safetensors");
            const auto linkedIndex=iild::inspectSafetensorsIndex(redirected);
            require(linkedIndex.sourcePath==redirected && linkedIndex.resolvedPath==fs::canonical(blob) &&
                    linkedIndex.shards.front().resolvedPath==fs::canonical(shards/"target-one.safetensors"),
                    "Linked index must resolve shards relative to its selected location");
            require(iild::safetensorsIndexShardPaths(redirected)==std::vector<fs::path>{snapshot/"linked.safetensors"},
                    "Shard paths must retain the index's logical base");
            require(iild::verifySafetensorsIndex(linkedIndex),"Linked index or shard snapshot was rejected");
            fs::remove(snapshot/"linked.safetensors");
            fs::create_symlink(shards/"target-two.safetensors",snapshot/"linked.safetensors");
            require(!iild::verifySafetensorsIndex(linkedIndex),"A linked index retained a retargeted shard");
            const auto refreshed=iild::inspectSafetensorsIndex(redirected);
            fs::remove(redirected);fs::create_symlink(blob2,redirected);
            require(!iild::verifySafetensorsIndex(refreshed),"A retargeted index retained its old identity");
        }
        const auto wan=root/"wan-directory";
        for (const auto &name:{"transformer","transformer_2","text_encoder","vae"})
            File::createDirectories(wan/name);
        File::write(wan/"model_index.json",R"({"_class_name":"WanPipeline","scheduler":["diffusers","UniPCMultistepScheduler"],"text_encoder":["transformers","UMT5EncoderModel"],"tokenizer":["transformers","T5TokenizerFast"],"transformer":["diffusers","WanTransformer3DModel"],"transformer_2":["diffusers","WanTransformer3DModel"],"vae":["diffusers","AutoencoderKLWan"],"boundary_ratio":0.875})");
        makeShard(wan/"transformer/high.bin","patch_embedding.weight");
        makeShard(wan/"transformer_2/low.safetensors","patch_embedding.weight");
        makeShard(wan/"text_encoder/text.safetensors","shared.weight");
        makeShard(wan/"vae/vae.safetensors","decoder.weight");
        const auto route=iild::inspectWanDirectory(wan);
        require(route.model.resolvedPath==fs::canonical(wan/"transformer_2/low.safetensors") && route.highNoiseModel &&
                route.highNoiseModel->resolvedPath==fs::canonical(wan/"transformer/high.bin") && route.moeBoundary &&
                *route.moeBoundary==0.875,"Wan directory expert roles or boundary were not preserved");
        require(route.textEncoder.resolvedPath==fs::canonical(wan/"text_encoder/text.safetensors") &&
                route.videoVae.resolvedPath==fs::canonical(wan/"vae/vae.safetensors"),"Wan directory lost its components");
        File::createDirectories(wan/"image_encoder");
        makeShard(wan/"image_encoder/vision.safetensors","vision_model.embeddings.patch_embedding.weight");
        File::write(wan/"model_index.json",R"({"_class_name":"WanImageToVideoPipeline","scheduler":["diffusers","UniPCMultistepScheduler"],"text_encoder":["transformers","UMT5EncoderModel"],"tokenizer":["transformers","T5TokenizerFast"],"transformer":["diffusers","WanTransformer3DModel"],"transformer_2":["diffusers","WanTransformer3DModel"],"vae":["diffusers","AutoencoderKLWan"],"image_encoder":["transformers","CLIPVisionModelWithProjection"],"image_processor":["transformers","CLIPProcessor"],"boundary_ratio":0.875})");
        const auto imageRoute=iild::inspectWanPipelineDirectory(wan);
        require(imageRoute.pipeline=="WanImageToVideoPipeline" && imageRoute.visionEncoder &&
                imageRoute.visionEncoder->resolvedPath==fs::canonical(wan/"image_encoder/vision.safetensors"),
                "The additive Wan pipeline API lost its image encoder");
        invalid=false;
        try {(void)iild::inspectWanDirectory(wan);} catch (const std::exception &) {invalid=true;}
        require(invalid,"The legacy text-only Wan API changed its existing contract");
        File::write(wan/"model_index.json",R"({"_class_name":"WanPipeline","scheduler":["diffusers","UniPCMultistepScheduler"],"text_encoder":["transformers","UMT5EncoderModel"],"tokenizer":["transformers","T5TokenizerFast"],"transformer":["diffusers","WanTransformer3DModel"],"transformer_2":["diffusers","WanTransformer3DModel"],"vae":["diffusers","AutoencoderKLWan"],"boundary_ratio":0.875})");
        makeShard(wan/"vae/second.safetensors","decoder.other.weight");
        invalid=false;
        try {(void)iild::inspectWanDirectory(wan);} catch (const std::exception &) {invalid=true;}
        require(invalid,"An ambiguous Wan component was accepted");
        std::cout << "Model inspection cache and metadata invalidation passed\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
