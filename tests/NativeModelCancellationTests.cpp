#include "Generation/NativeUpscaler.hpp"
#include "Generation/NativeUpscalerInternal.hpp"
#include "NativeModelUpscalerFixtures.hpp"
#include <atomic>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace iiLocalDiffusion;
int main(int argc,char **argv) {
    try {
        if(argc!=2) throw std::runtime_error("Expected resource workspace");
        const auto path=std::filesystem::path(argv[1])/"cancel-model.safetensors";
        model_fixtures::writeModel(path,model_fixtures::srvggFixture());
        NativeModelUpscalerOptions options; options.model=path; options.threads=2;
        auto model=makeNativeModelUpscaler(options);
        const UpscaleImage input{16,16,3,std::vector<std::uint8_t>(16*16*3,80)};
        std::atomic<int> checks{0}; bool cancelled=false;
        {
            native_detail::UpscaleStopProbe probe=[&] { return ++checks>30; };
            native_detail::UpscaleStopScope scope(&probe);
            try { (void)upscaleImage(input,*model,UpscaleOptions{}); }
            catch(const UpscaleCancelled &) { cancelled=true; }
        }
        if(!cancelled || checks<=30) throw std::runtime_error("Model did not observe a cancellation during active work");
        const auto output=upscaleImage(input,*model,UpscaleOptions{});
        if(output.image.width!=32 || output.image.height!=32) throw std::runtime_error("Cancelled worker session did not recover");
        options.tileSize=4; auto tiled=makeNativeModelUpscaler(options);
        checks=0; cancelled=false;
        {
            native_detail::UpscaleStopProbe probe=[&] { return ++checks>120; };
            native_detail::UpscaleStopScope scope(&probe);
            try { (void)upscaleImage(input,*tiled,UpscaleOptions{}); }
            catch(const UpscaleCancelled &) { cancelled=true; }
        }
        if(!cancelled || checks<=120) throw std::runtime_error("Tiled processing did not observe cancellation");
        const auto recovered=upscaleImage(input,*tiled,UpscaleOptions{});
        if(recovered.image.pixels!=output.image.pixels) throw std::runtime_error("Cancelled tiled session did not recover");
        std::cout << "Native model cancellation and worker cleanup passed\n";
    } catch(const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
