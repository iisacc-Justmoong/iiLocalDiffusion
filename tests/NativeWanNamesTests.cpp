#include <ModelPackaging/ModelInspection.hpp>
#include <ModelPackaging/ModelPackaging.hpp>
#include <model_loader.h>
#include <name_conversion.h>
#include <stable-diffusion.h>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs=std::filesystem;
void require(bool condition,const char *message) {if (!condition) throw std::runtime_error(message);}

int main(int argc,char **argv) {
    if (argc!=2) return 2;
    try {
        const auto root=fs::absolute(argv[1]);fs::create_directories(root);
        const auto model=root/"wan.safetensors";
        const std::string header=R"({"model.diffusion_model.patch_embedding.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"model.diffusion_model.blocks.0.cross_attn.norm_k.weight":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}})";
        {
            std::ofstream file(model,std::ios::binary);
            const auto length=static_cast<std::uint64_t>(header.size());
            for (unsigned shift=0;shift<64;shift+=8) file.put(static_cast<char>((length>>shift)&0xff));
            file << header << std::string(8,'\0');
            require(bool(file),"Cannot write synthetic Wan metadata");
        }
        const auto inspected=iild::inspectModel(model);
        require(inspected.recognized && inspected.family=="Wan" && inspected.modalities==std::vector<std::string>{"video"},
                "The catalogue did not recognize the Wan tensor signature");
        const auto unknownType=root/"standalone-future.safetensors";
        const std::string unknownHeader=R"({"tensor":{"dtype":"FUTURE_TYPE","shape":[1],"data_offsets":[0,1]}})";
        {std::ofstream file(unknownType,std::ios::binary);const auto length=static_cast<std::uint64_t>(unknownHeader.size());
         for (unsigned shift=0;shift<64;shift+=8) file.put(static_cast<char>((length>>shift)&0xff));
         file<<unknownHeader<<'x';}
        require(iild::detail::inspectIncludedPackageAssets(unknownType).roles.empty(),
            "Package metadata inspection must not claim or reject an ordinary model's tensor dtype");
        ModelLoader loader;
        require(loader.init_from_file(model.string()),"Native engine could not inspect the synthetic Wan file");
        loader.convert_tensors_name();
        require(loader.get_sd_version()==VERSION_WAN2,"Native engine did not recognize the same Wan signature");
        constexpr auto prefix="text_encoders.t5xxl.transformer.";
        require(convert_tensor_name(std::string(prefix)+"token_embd.weight",VERSION_WAN2)==
                    std::string(prefix)+"shared.weight", "Wan T5 token embedding conversion differs");
        require(convert_tensor_name(std::string(prefix)+"enc.output_norm.weight",VERSION_WAN2)==
                    std::string(prefix)+"encoder.final_layer_norm.weight", "Wan T5 output normalization conversion differs");
        require(convert_tensor_name(std::string(prefix)+"enc.blk.0.attn_q.weight",VERSION_WAN2)==
                    std::string(prefix)+"encoder.block.0.layer.0.SelfAttention.q.weight", "Wan T5 attention conversion differs");
        require(convert_tensor_name("vae.decoder.conv_in.weight",VERSION_WAN2)==
                    "first_stage_model.decoder.conv1.weight", "Diffusers Wan VAE input conversion differs");
        require(convert_tensor_name("vae.decoder.up_blocks.0.resnets.0.conv1.weight",VERSION_WAN2)==
                    "first_stage_model.decoder.upsamples.0.residual.2.weight", "Diffusers Wan VAE residual conversion differs");
        const auto input=root/"package-input";
        fs::create_directories(input);
        fs::copy_file(model,input/"model.safetensors",fs::copy_options::overwrite_existing);
        const std::string vaeHeader=R"({"decoder.conv1.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})";
        {
            std::ofstream file(input/"video_vae.safetensors",std::ios::binary);
            const auto length=static_cast<std::uint64_t>(vaeHeader.size());
            for (unsigned shift=0;shift<64;shift+=8) file.put(static_cast<char>((length>>shift)&0xff));
            file << vaeHeader << std::string(4,'\0');
            require(bool(file),"Cannot write synthetic Wan VAE metadata");
        }
        const auto package=root/"wan-packaged.safetensors";
        (void)iild::createModelPackage(input,package);
        const auto extracted=iild::materializeModelPackageAsset(package,root/"cache","video_vae");
        ModelLoader packageLoader;
        require(packageLoader.init_from_file(package.string()),"Native loader could not inspect the Wan package");
        require(packageLoader.init_from_file(extracted.string(),"vae."),"Native loader could not inspect the extracted VAE");
        packageLoader.convert_tensors_name();
        require(packageLoader.get_sd_version()==VERSION_WAN2,"Extracting the VAE must preserve the Wan model family");
        const auto converted=convert_tensor_name("vae.decoder.conv1.weight",VERSION_WAN2);
        const auto &tensors=packageLoader.get_tensor_storage_map();
        require(tensors.find(converted)!=tensors.end(),"The package VAE must reach the engine's converted tensor namespace");
        const auto ggufInput=root/"gguf-package-input";
        fs::create_directories(ggufInput);
        std::string gguf="GGUF";
        const auto number=[&](std::uint64_t value,unsigned count) {
            for (unsigned i=0;i<count;++i) gguf.push_back(static_cast<char>((value>>(8*i))&255));
        };
        const auto string=[&](const std::string &value) {number(value.size(),8);gguf+=value;};
        number(3,4);number(3,8);number(1,8);
        string("general.architecture");number(8,4);string("wan");
        const std::vector<std::string> names{"patch_embedding.weight","blocks.0.cross_attn.norm_k.weight",
            "text_encoders.t5xxl.transformer.shared.weight"};
        for (std::size_t i=0;i<names.size();++i) {
            string(names[i]);number(1,4);number(1,8);number(0,4);number(i*32,8);
        }
        gguf.append((32-gguf.size()%32)%32,'\0');gguf.append(68,'\0');
        {std::ofstream file(ggufInput/"model.gguf",std::ios::binary);file.write(gguf.data(),gguf.size());
         require(bool(file),"Cannot write synthetic GGUF metadata");}
        fs::copy_file(input/"video_vae.safetensors",ggufInput/"video_vae.safetensors",fs::copy_options::overwrite_existing);
        const auto ggufPackage=root/"wan-gguf-packaged.safetensors";
        (void)iild::createModelPackage(ggufInput,ggufPackage);
        const auto restoredModel=iild::materializeModelPackageAssetByPath(ggufPackage,root/"gguf-cache","model.gguf");
        const auto restoredVae=iild::materializeModelPackageAsset(ggufPackage,root/"gguf-cache","video_vae");
        ModelLoader ggufLoader;
        require(ggufLoader.init_from_file(restoredModel.string()),"Native loader could not read the restored main GGUF");
        require(ggufLoader.init_from_file(restoredModel.string(),"model.diffusion_model."),
            "Native loader could not mount bare diffusion tensor names");
        require(ggufLoader.init_from_file(restoredVae.string(),"vae."),"Native loader could not read the GGUF package VAE");
        ggufLoader.convert_tensors_name();
        require(ggufLoader.get_sd_version()==VERSION_WAN2 && ggufLoader.get_tensor_storage_map().find(converted)!=ggufLoader.get_tensor_storage_map().end(),
            "The restored main GGUF and VAE must share the engine's Wan tensor namespace");
        require(ggufLoader.get_tensor_storage_map().find("text_encoders.t5xxl.transformer.shared.weight")!=ggufLoader.get_tensor_storage_map().end(),
            "Mounting the bare denoiser must preserve an in-file text encoder");
        const auto highPrefix=std::string("model.high_noise_diffusion_model.");
        const std::vector<std::string> sourcePrefixes{"", "model.diffusion_model.", "diffusion_model.",
                                                       "transformer.", highPrefix};
        for (std::size_t i=0;i<sourcePrefixes.size();++i) {
            const auto name=sourcePrefixes[i]+"patch_embedding.weight";
            const auto attention=sourcePrefixes[i]+"blocks.0.cross_attn.norm_k.weight";
            const std::string sourceHeader="{\""+name+"\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},\""+
                attention+"\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,8]},"
                "\"text_encoders.t5xxl.transformer.shared.weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[8,12]}}";
            const auto filePath=root/("high-"+std::to_string(i)+".safetensors");
            {std::ofstream file(filePath,std::ios::binary);const auto length=static_cast<std::uint64_t>(sourceHeader.size());
             for (unsigned shift=0;shift<64;shift+=8) file.put(static_cast<char>((length>>shift)&0xff));
             file<<sourceHeader<<std::string(12,'\0');require(bool(file),"Cannot write synthetic high-noise weights");}
            ModelLoader expert;
            require(expert.init_from_file(filePath.string(),highPrefix),"Native loader could not mount the high-noise expert");
            const auto &highTensors=expert.get_tensor_storage_map();
            require(highTensors.find(highPrefix+"patch_embedding.weight")!=highTensors.end() &&
                highTensors.find(highPrefix+"blocks.0.cross_attn.norm_k.weight")!=highTensors.end(),
                "High-noise tensor prefixes must normalize to one native expert namespace");
            require(highTensors.find(highPrefix+"text_encoders.t5xxl.transformer.shared.weight")==highTensors.end(),
                "Embedded auxiliary encoders must not be mounted as high-noise diffusion weights");
        }
        ModelLoader ggufExpert;
        require(ggufExpert.init_from_file(restoredModel.string(),highPrefix),"Native loader could not mount GGUF expert weights");
        require(ggufExpert.get_tensor_storage_map().find(highPrefix+"patch_embedding.weight")!=ggufExpert.get_tensor_storage_map().end() &&
            ggufExpert.get_tensor_storage_map().find(highPrefix+"text_encoders.t5xxl.transformer.shared.weight")==ggufExpert.get_tensor_storage_map().end(),
            "GGUF expert mounting must retain only the expert tensor namespace");
        const auto extra=root/"high-extra.safetensors";
        const std::string extraHeader=R"({"model.diffusion_model.other.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})";
        {std::ofstream file(extra,std::ios::binary);const auto length=static_cast<std::uint64_t>(extraHeader.size());
         for (unsigned shift=0;shift<64;shift+=8) file.put(static_cast<char>((length>>shift)&0xff));
         file<<extraHeader<<std::string(4,'\0');}
        const auto indexPath=root/"high.safetensors.index.json";
        {std::ofstream file(indexPath);file<<R"({"weight_map":{"model.diffusion_model.patch_embedding.weight":"high-1.safetensors","model.diffusion_model.blocks.0.cross_attn.norm_k.weight":"high-1.safetensors","text_encoders.t5xxl.transformer.shared.weight":"high-1.safetensors","model.diffusion_model.other.weight":"high-extra.safetensors"}})";}
        ModelLoader shardedExpert;
        require(shardedExpert.init_from_file(indexPath.string(),highPrefix) &&
            shardedExpert.get_tensor_storage_map().find(highPrefix+"other.weight")!=shardedExpert.get_tensor_storage_map().end(),
            "Indexed high-noise shards must share the normalized expert namespace");
        const auto collision=root/"high-collision.safetensors";
        const std::string collisionHeader=R"({"patch_embedding.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"model.diffusion_model.patch_embedding.weight":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}})";
        {std::ofstream file(collision,std::ios::binary);const auto length=static_cast<std::uint64_t>(collisionHeader.size());
         for (unsigned shift=0;shift<64;shift+=8) file.put(static_cast<char>((length>>shift)&0xff));
         file<<collisionHeader<<std::string(8,'\0');}
        ModelLoader conflictingExpert;
        require(!conflictingExpert.init_from_file(collision.string(),highPrefix) && conflictingExpert.get_tensor_storage_map().empty(),
            "Colliding high-noise tensor aliases must be rejected without a partial mount");
        const auto diffusers=root/"diffusers-expert.safetensors";
        const std::vector<std::string> diffusersNames{"patch_embedding.weight", "blocks.0.attn1.to_q.weight",
            "blocks.0.attn2.norm_k.weight", "condition_embedder.time_embedder.linear_1.weight", "blocks.0.norm2.weight",
            "blocks.0.ffn.net.0.proj.weight", "blocks.0.scale_shift_table", "scale_shift_table", "proj_out.weight"};
        std::string diffusersHeader="{";
        for (std::size_t i=0;i<diffusersNames.size();++i) {
            if (i) diffusersHeader+=",";
            diffusersHeader+="\""+diffusersNames[i]+"\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":["+
                std::to_string(i*4)+","+std::to_string((i+1)*4)+"]}";
        }
        diffusersHeader+="}";
        {std::ofstream file(diffusers,std::ios::binary);const auto length=static_cast<std::uint64_t>(diffusersHeader.size());
         for (unsigned shift=0;shift<64;shift+=8) file.put(static_cast<char>((length>>shift)&0xff));
         file<<diffusersHeader<<std::string(diffusersNames.size()*4,'\0');}
        ModelLoader diffusersExpert;
        require(diffusersExpert.init_from_file(diffusers.string(),highPrefix),"Diffusers expert mount failed");
        for (const auto &name : {"blocks.0.self_attn.q.weight","blocks.0.cross_attn.norm_k.weight","time_embedding.0.weight",
                                 "blocks.0.norm3.weight","blocks.0.ffn.0.weight","blocks.0.modulation","head.modulation","head.head.weight"})
            require(diffusersExpert.get_tensor_storage_map().find(highPrefix+name)!=diffusersExpert.get_tensor_storage_map().end(),
                "Diffusers expert names must map to native Wan tensor names");
        const auto writeSynthetic=[&](const fs::path &path,const std::vector<std::string> &names) {
            std::string json="{";
            for (std::size_t i=0;i<names.size();++i) {
                if (i) json+=",";
                json+="\""+names[i]+"\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":["+
                    std::to_string(i*4)+","+std::to_string((i+1)*4)+"]}";
            }
            json+="}";
            std::ofstream file(path,std::ios::binary);const auto length=static_cast<std::uint64_t>(json.size());
            for (unsigned shift=0;shift<64;shift+=8) file.put(static_cast<char>((length>>shift)&0xff));
            file<<json<<std::string(names.size()*4,'\0');require(bool(file),"Cannot write standalone Diffusers metadata");
        };
        for (std::size_t i=0;i<4;++i) {
            std::vector<std::string> names;
            for (const auto &name:diffusersNames) names.push_back(sourcePrefixes[i]+name);
            names.push_back("text_encoders.t5xxl.transformer.shared.weight");
            const auto source=root/("diffusers-main-"+std::to_string(i)+".safetensors");
            writeSynthetic(source,names);
            ModelLoader standalone;
            require(standalone.init_from_file(source.string()) && standalone.init_from_file(source.string(),"model.diffusion_model."),
                "Standalone Diffusers Wan metadata must mount as a diffusion model");
            standalone.convert_tensors_name();
            const auto &mapped=standalone.get_tensor_storage_map();
            require(standalone.get_sd_version()==VERSION_WAN2 &&
                mapped.find("model.diffusion_model.blocks.0.cross_attn.norm_k.weight")!=mapped.end() &&
                mapped.find("model.diffusion_model.blocks.0.self_attn.q.weight")!=mapped.end() &&
                mapped.find("text_encoders.t5xxl.transformer.shared.weight")!=mapped.end(),
                "Standalone Wan tensors must be normalized while preserving embedded text encoders");
        }
        const auto shardDir=root/"diffusers-shards";fs::create_directories(shardDir);
        writeSynthetic(shardDir/"a.safetensors",{diffusersNames[0],diffusersNames[1]});
        writeSynthetic(shardDir/"b.safetensors",{diffusersNames[2],diffusersNames[3]});
        const auto shardIndex=shardDir/"model.safetensors.index.json";
        {std::ofstream file(shardIndex);file<<R"({"weight_map":{"patch_embedding.weight":"a.safetensors","blocks.0.attn1.to_q.weight":"a.safetensors","blocks.0.attn2.norm_k.weight":"b.safetensors","condition_embedder.time_embedder.linear_1.weight":"b.safetensors"}})";}
        ModelLoader standaloneShards;
        require(standaloneShards.init_from_file(shardIndex.string()) && standaloneShards.init_from_file(shardIndex.string(),"model.diffusion_model."),
            "Standalone sharded Diffusers Wan metadata must mount");
        standaloneShards.convert_tensors_name();
        require(standaloneShards.get_sd_version()==VERSION_WAN2,"Standalone Diffusers Wan shards must be identified by the native engine");
        const auto snapshot=root/"linked-snapshot",blobDir=root/"linked-blobs";
        fs::create_directories(snapshot);fs::create_directories(blobDir);
        fs::copy_file(shardIndex,blobDir/"index-blob");
        fs::copy_file(shardDir/"a.safetensors",blobDir/"a-blob");
        fs::copy_file(shardDir/"b.safetensors",blobDir/"b-blob");
        const auto linkedIndex=snapshot/"model.safetensors.index.json";
        std::error_code linkError;fs::create_symlink(blobDir/"index-blob",linkedIndex,linkError);
        if (!linkError) {
            fs::create_symlink(blobDir/"a-blob",snapshot/"a.safetensors");
            fs::create_symlink(blobDir/"b-blob",snapshot/"b.safetensors");
            ModelLoader linkedShards;
            require(linkedShards.init_from_file(linkedIndex.string()) &&
                    linkedShards.init_from_file(linkedIndex.string(),"model.diffusion_model."),
                    "The real native loader must use the logical linked-index directory");
            linkedShards.convert_tensors_name();
            require(linkedShards.get_sd_version()==VERSION_WAN2,
                    "The real native loader must classify linked Wan shards");
        }
        const auto visionPath=root/"wan-image-encoder.safetensors";
        writeSynthetic(visionPath,{"vision_model.embeddings.patch_embedding.weight","visual_projection.weight"});
        ModelLoader visionLoader;
        require(visionLoader.init_from_file(visionPath.string(),"clip_vision."),
                "The native loader must mount a Diffusers CLIP vision encoder");
        visionLoader.convert_tensors_name();
        require(visionLoader.get_tensor_storage_map().contains(
                    "cond_stage_model.transformer.vision_model.embeddings.patch_embedding.weight") &&
                visionLoader.get_tensor_storage_map().contains("cond_stage_model.transformer.visual_projection.weight"),
                "The native loader must map Diffusers CLIP vision tensors to the Wan conditioning namespace");
        std::string standaloneGguf="GGUF";
        const auto ggufNumber=[&](std::uint64_t value,unsigned count) {
            for (unsigned i=0;i<count;++i) standaloneGguf.push_back(static_cast<char>((value>>(8*i))&255));
        };
        const auto ggufString=[&](const std::string &value) {ggufNumber(value.size(),8);standaloneGguf+=value;};
        ggufNumber(3,4);ggufNumber(diffusersNames.size(),8);ggufNumber(1,8);
        ggufString("general.architecture");ggufNumber(8,4);ggufString("wan");
        for (std::size_t i=0;i<diffusersNames.size();++i) {
            ggufString(diffusersNames[i]);ggufNumber(1,4);ggufNumber(1,8);ggufNumber(0,4);ggufNumber(i*32,8);
        }
        standaloneGguf.append((32-standaloneGguf.size()%32)%32,'\0');
        standaloneGguf.append((diffusersNames.size()-1)*32+4,'\0');
        const auto ggufDiffusersPath=root/"diffusers-main.gguf";
        {std::ofstream file(ggufDiffusersPath,std::ios::binary);file.write(standaloneGguf.data(),standaloneGguf.size());}
        ModelLoader standaloneGgufLoader;
        require(standaloneGgufLoader.init_from_file(ggufDiffusersPath.string()) &&
            standaloneGgufLoader.init_from_file(ggufDiffusersPath.string(),"model.diffusion_model."),
            "Standalone Diffusers GGUF Wan metadata must mount");
        standaloneGgufLoader.convert_tensors_name();
        require(standaloneGgufLoader.get_sd_version()==VERSION_WAN2,
            "Standalone Diffusers GGUF Wan weights must be identified by the native engine");
        auto conflictingNames=diffusersNames;
        conflictingNames.push_back("model.diffusion_model.blocks.0.cross_attn.norm_k.weight");
        const auto conflictingMain=root/"diffusers-main-collision.safetensors";
        writeSynthetic(conflictingMain,conflictingNames);
        ModelLoader conflictingWan;
        require(conflictingWan.init_from_file(conflictingMain.string()),"Collision fixture must be readable");
        const auto beforeCollision=conflictingWan.get_tensor_storage_map().size();
        require(!conflictingWan.init_from_file(conflictingMain.string(),"model.diffusion_model.") &&
            conflictingWan.get_tensor_storage_map().size()==beforeCollision,
            "Standalone Diffusers tensor alias collisions must not partially mount");
        std::string engineLog;
        sd_set_log_callback([](sd_log_level_t,const char *message,void *user) {
            if (message) static_cast<std::string *>(user)->append(message);
        },&engineLog);
        sd_ctx_params_t parameters;sd_ctx_params_init(&parameters);
        const auto conflictingPath=conflictingMain.string();
        parameters.model_path=conflictingPath.c_str();parameters.diffusion_model_path=conflictingPath.c_str();
        parameters.backend="cpu";parameters.params_backend="cpu";parameters.n_threads=1;
        auto *context=new_sd_ctx(&parameters);
        sd_set_log_callback(nullptr,nullptr);
        if (context) free_sd_ctx(context);
        require(!context && engineLog.find("failed to mount the requested diffusion model")!=std::string::npos,
            "The native context must fail and report a conflicting same-file diffusion mount");
        const auto diffusersInput=root/"diffusers-package-input";
        fs::create_directories(diffusersInput/"transformer");
        fs::create_directories(diffusersInput/"transformer_2");
        fs::copy_file(diffusers,diffusersInput/"transformer"/"weights.safetensors",fs::copy_options::overwrite_existing);
        fs::copy_file(diffusers,diffusersInput/"transformer_2"/"weights.safetensors",fs::copy_options::overwrite_existing);
        {std::ofstream file(diffusersInput/"model_index.json");
         file<<R"({"_class_name":"WanPipeline","boundary_ratio":0.875,"transformer":["diffusers","WanTransformer3DModel"],"transformer_2":["diffusers","WanTransformer3DModel"]})";}
        const auto diffusersPackage=root/"diffusers-package.safetensors";
        (void)iild::createModelPackage(diffusersInput,diffusersPackage);
        const auto extractedExpert=iild::materializeModelPackageAsset(diffusersPackage,root/"diffusers-cache","high_noise_model");
        ModelLoader fullWan;
        require(fullWan.init_from_file(diffusersPackage.string()) && fullWan.init_from_file(extractedExpert.string(),highPrefix),
            "Native loader must inspect both packaged Wan experts");
        fullWan.convert_tensors_name();
        require(fullWan.get_sd_version()==VERSION_WAN2 &&
            fullWan.get_tensor_storage_map().find("model.diffusion_model.blocks.0.cross_attn.norm_k.weight")!=fullWan.get_tensor_storage_map().end() &&
            fullWan.get_tensor_storage_map().find(highPrefix+"blocks.0.cross_attn.norm_k.weight")!=fullWan.get_tensor_storage_map().end(),
            "Both Diffusers experts must reach the actual native Wan namespaces");
        std::cout << "Native Wan metadata and T5 name conversion passed\n";
        return 0;
    } catch (const std::exception &error) {std::cerr << error.what() << '\n';return 1;}
}
