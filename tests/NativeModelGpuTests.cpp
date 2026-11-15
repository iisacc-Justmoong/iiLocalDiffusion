#include "Generation/NativeUpscaler.hpp"
#include "NativeModelUpscalerFixtures.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>

using namespace iiLocalDiffusion;
using namespace model_fixtures;

int main(int argc,char **argv) {
    try {
        checkFixture(argc==3,"Expected an isolated resource workspace and GPU backend");
        const std::string backend=argv[2];
        const auto gpuDevice=backend=="metal" ? UpscaleDevice::Metal : backend=="directx" ? UpscaleDevice::DirectX :
            backend=="vulkan" ? UpscaleDevice::Vulkan : throw std::invalid_argument("Unknown GPU test backend");
        for(bool swin : {false,true}) {
            const auto path=std::filesystem::path(argv[1])/(swin ? "gpu-swinir.safetensors" : "gpu-srvgg.safetensors");
            auto tensors=swin ? swinFixture() : srvggFixture();
            if(swin) {
                const std::string original="layers.0.residual_group.blocks.0.";
                std::vector<Tensor> shifted;
                for(const auto &tensor:tensors) if(tensor.name.starts_with(original)) {
                    auto copy=tensor; copy.name.replace(0,original.size(),"layers.0.residual_group.blocks.1.");
                    shifted.push_back(std::move(copy));
                }
                tensors.insert(tensors.end(),shifted.begin(),shifted.end());
            } else for(auto &tensor:tensors) if(tensor.name=="body.0.weight") {
                for(std::size_t i=0;i<tensor.data.size();++i) tensor.data[i]=.002f*float(i%9+1);
            }
            writeModel(path,tensors);
            UpscaleImage input{5,4,3,std::vector<std::uint8_t>(5*4*3)};
            for(std::size_t i=0;i<input.pixels.size();++i) input.pixels[i]=std::uint8_t((i*17+7)%200);
            NativeModelUpscalerOptions options; options.model=path; options.device=UpscaleDevice::Cpu; options.tileSize=2;
            const auto cpu=upscaleImage(input,*makeNativeModelUpscaler(options),UpscaleOptions{});
            options.device=gpuDevice;
            const auto gpu=upscaleImage(input,*makeNativeModelUpscaler(options),UpscaleOptions{});
            checkFixture(gpu.details.at("actual_device")==backend, "The requested GPU was not selected");
            checkFixture(std::stoull(gpu.details.at("gpu_convolutions"))>0,"No GPU convolution was executed");
            if(swin) checkFixture(std::stoull(gpu.details.at("gpu_projections"))>0,"No GPU SwinIR projection was executed");
            checkFixture(cpu.image.pixels.size()==gpu.image.pixels.size(),"GPU output size differs");
            for(std::size_t i=0;i<cpu.image.pixels.size();++i)
                if(std::abs(int(cpu.image.pixels[i])-int(gpu.image.pixels[i]))>1)
                    throw std::runtime_error("GPU result differs from the CPU model");
            if(backend=="metal") {
                options.device=UpscaleDevice::DirectX;
                bool rejected=false;
                try { (void)upscaleImage(input,*makeNativeModelUpscaler(options),UpscaleOptions{}); }
                catch(const std::runtime_error &) { rejected=true; }
                checkFixture(rejected,"Unavailable DirectX silently fell back to a different device");
            }
        }
        std::cout << "Synthetic SRVGG and SwinIR " << backend << " tests passed\n";
    } catch(const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
