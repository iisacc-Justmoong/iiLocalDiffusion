#include "Generation/NativeUpscaler.hpp"
#include "NativeModelUpscalerFixtures.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
using namespace iiLocalDiffusion;
namespace {
void require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
}
using namespace model_fixtures;
int main(int argc,char **argv) {
    try {
        require(argc==2,"Expected isolated resource workspace");
        const auto path=std::filesystem::path(argv[1])/"srvgg.safetensors";
        writeModel(path,srvggFixture());
        NativeModelUpscalerOptions config; config.model=path; config.architecture=UpscaleModelArchitecture::Automatic;
        config.device=UpscaleDevice::Cpu; config.threads=2;
        auto model=makeNativeModelUpscaler(config);
        auto result=upscaleImage({1,1,3,{51,102,153}},*model,UpscaleOptions{});
        require(result.image.width==2 && result.image.height==2 && result.appliedScale==2,"Wrong SRVGG shape or inferred scale");
        require(result.upscaler=="native-model" && result.details.at("architecture")=="srvgg" && result.details.at("actual_device")=="cpu","Missing native SRVGG backend report");
        const int input[]{51,102,153};
        for(int y=0;y<2;++y) for(int x=0;x<2;++x) for(int c=0;c<3;++c) {
            const int o=c*4+y*2+x;
            const float v=float(input[c])/255.f + ((o%2) ? .2f : -.02f) + float(o)*.01f;
            const int expected=int(std::lround(std::clamp(v,0.f,1.f)*255));
            require(std::abs(int(result.image.pixels[std::size_t((y*2+x)*3+c)])-expected)<=1,
                    "SRVGG convolution, PReLU, pixel shuffle or residual differs");
        }
        auto withoutActivationWeights=srvggFixture();
        std::erase_if(withoutActivationWeights,[](const Tensor &tensor){return tensor.name=="body.1.weight";});
        const auto activationPath=std::filesystem::path(argv[1])/"activations.safetensors";
        writeModel(activationPath,withoutActivationWeights); config.model=activationPath;
        for(const auto &[activation,expected] : std::vector<std::pair<SrvggActivation,int>>{
                {SrvggActivation::Relu,51},{SrvggActivation::LeakyRelu,46}}) {
            config.srvggActivation=activation;
            const auto value=upscaleImage({1,1,3,{51,102,153}},*makeNativeModelUpscaler(config),UpscaleOptions{});
            require(std::abs(int(value.image.pixels[0])-expected)<=1,"SRVGG activation selection differs");
        }
        config.srvggActivation=SrvggActivation::PRelu;
        auto halfWeights=srvggFixture();
        for(auto &t:halfWeights) {
            t.name="params_ema.module."+t.name;
            if(t.name.ends_with("body.0.bias")) t.data={-.25f,.25f};
            else if(t.name.ends_with("body.1.weight")) t.data={.5f,.5f};
            else if(t.name.ends_with("body.2.bias")) std::fill(t.data.begin(),t.data.end(),.25f);
        }
        for(bool bf16:{false,true}) {
            const auto halfPath=std::filesystem::path(argv[1])/(bf16 ? "bf16.safetensors" : "f16.safetensors");
            writeHalfModel(halfPath,halfWeights,bf16); config.model=halfPath;
            auto half=makeNativeModelUpscaler(config);
            const auto output=upscaleImage({1,1,3,{51,102,153}},*half,UpscaleOptions{});
            require(std::abs(int(output.image.pixels.front())-83)<=1,"Half-precision or nested tensor prefix loading differs");
        }
        auto gapped=srvggFixture();
        gapped.push_back({"body.6.weight",{12,2,3,3},std::vector<float>(12*2*9,0)});
        const auto gappedPath=std::filesystem::path(argv[1])/"gapped.safetensors";
        writeModel(gappedPath,gapped); config.model=gappedPath;
        bool gappedRejected=false;
        try { (void)upscaleImage({1,1,3,{1,2,3}},*makeNativeModelUpscaler(config),UpscaleOptions{}); }
        catch(const std::invalid_argument &) { gappedRejected=true; }
        require(gappedRejected,"Non-contiguous SRVGG layers were silently ignored");
        config.model=path;
        config.maxWorkingBytes=8;
        bool limited=false;
        try { auto provider=makeNativeModelUpscaler(config); (void)upscaleImage({1,1,3,{1,2,3}},*provider,UpscaleOptions{}); }
        catch(const std::exception &) { limited=true; }
        require(limited,"Model working memory limit was ignored");
        config.maxWorkingBytes=512ULL*1024*1024;
        config.maxWeightBytes=16; limited=false;
        try { auto provider=makeNativeModelUpscaler(config); (void)upscaleImage({1,1,3,{1,2,3}},*provider,UpscaleOptions{}); }
        catch(const std::exception &) { limited=true; }
        require(limited,"Model weight memory limit was ignored");
        config.maxWeightBytes=1024ULL*1024*1024;
        auto badConfig=config; badConfig.model="forbidden.onnx"; bool rejected=false;
        try { (void)makeNativeModelUpscaler(badConfig); } catch(const std::invalid_argument &) { rejected=true; }
        require(rejected,"ONNX model was accepted");
        badConfig=config; badConfig.tileSize=-1; rejected=false;
        try { (void)makeNativeModelUpscaler(badConfig); } catch(const std::invalid_argument &) { rejected=true; }
        require(rejected,"Negative tile size was accepted");
        auto spatialSrvgg=srvggFixture();
        for(auto &tensor:spatialSrvgg) if(tensor.name=="body.0.weight") {
            for(std::size_t oc=0;oc<2;++oc) for(std::size_t ic=0;ic<3;++ic)
                for(std::size_t k=0;k<9;++k) tensor.data[(oc*3+ic)*9+k]=.005f*float(k+1);
        }
        const auto spatialPath=std::filesystem::path(argv[1])/"srvgg-spatial.safetensors";
        writeModel(spatialPath,spatialSrvgg);
        std::vector<std::uint8_t> spatialPixels(37*29*3);
        for(std::size_t i=0;i<spatialPixels.size();++i) spatialPixels[i]=std::uint8_t((i*29+i/11)%180);
        const UpscaleImage spatial{37,29,3,spatialPixels};
        auto spatialConfig=config; spatialConfig.model=spatialPath; spatialConfig.tileSize=0;
        const auto wholeSrvgg=upscaleImage(spatial,*makeNativeModelUpscaler(spatialConfig),UpscaleOptions{});
        spatialConfig.tileSize=7;
        const auto tiledSrvgg=upscaleImage(spatial,*makeNativeModelUpscaler(spatialConfig),UpscaleOptions{});
        require(tiledSrvgg.image.pixels==wholeSrvgg.image.pixels && tiledSrvgg.details.at("tile_size")=="7",
                "SRVGG tiling changed pixels at a tile edge");
        const auto boundedPath=std::filesystem::path(argv[1])/"srvgg-tile-memory.safetensors";
        writeModel(boundedPath,{{"body.0.weight",{64,3,3,3},std::vector<float>(64*3*9)},
            {"body.0.bias",{64},std::vector<float>(64)}, {"body.1.weight",{64},std::vector<float>(64,1)},
            {"body.2.weight",{12,64,3,3},std::vector<float>(12*64*9)}, {"body.2.bias",{12},std::vector<float>(12)}});
        auto boundedConfig=config; boundedConfig.model=boundedPath; boundedConfig.maxWorkingBytes=200000;
        const UpscaleImage boundedInput{64,64,3,std::vector<std::uint8_t>(64*64*3,73)};
        bool wholeLimited=false;
        try { (void)upscaleImage(boundedInput,*makeNativeModelUpscaler(boundedConfig),UpscaleOptions{}); }
        catch(const std::runtime_error &) { wholeLimited=true; }
        require(wholeLimited,"The full model should exceed the requested working tensor limit");
        boundedConfig.tileSize=16;
        const auto bounded=upscaleImage(boundedInput,*makeNativeModelUpscaler(boundedConfig),UpscaleOptions{});
        require(bounded.image.width==128 && bounded.image.height==128
                && std::all_of(bounded.image.pixels.begin(),bounded.image.pixels.end(),[](auto pixel){return pixel==73;}),
                "Tiling failed to keep model working tensors within the requested limit");
        const auto swinPath=std::filesystem::path(argv[1])/"swinir.safetensors";
        writeModel(swinPath,swinFixture()); config.model=swinPath; config.threads=1;
        auto swin=makeNativeModelUpscaler(config);
        std::vector<std::uint8_t> pixels; for(int i=0;i<9;++i) pixels.insert(pixels.end(),{51,204,102});
        auto restored=upscaleImage({3,3,3,pixels},*swin,UpscaleOptions{});
        require(restored.image.width==6 && restored.image.height==6 && restored.details.at("architecture")=="swinir",
                "SwinIR model architecture or output dimensions differ");
        const double a=51.0/255.0-.4488, b=204.0/255.0-.4371;
        const double n=(a-b)/std::sqrt((a-b)*(a-b)+4e-5);
        const double ga=2*a-n, gb=2*b+n;
        const double n2=(ga-gb)/std::sqrt((ga-gb)*(ga-gb)+4e-5);
        const int expectedR=int(std::lround((.4488+.1*(a+n2))*255));
        const int expectedG=int(std::lround((.4371+.1*(b-n2))*255));
        const int expectedB=int(std::lround(.4040*255));
        for(std::size_t i=0;i<restored.image.pixels.size();i+=3) {
            require(std::abs(int(restored.image.pixels[i])-expectedR)<=1
                    && std::abs(int(restored.image.pixels[i+1])-expectedG)<=1
                    && std::abs(int(restored.image.pixels[i+2])-expectedB)<=1,
                    "SwinIR attention, layer norm, residual or pixel shuffle differs");
        }
        auto rangeConfig=config; rangeConfig.model=swinPath; rangeConfig.swinIRImageRange=2;
        const auto ranged=upscaleImage({3,3,3,pixels},*makeNativeModelUpscaler(rangeConfig),UpscaleOptions{});
        const double ra=a*2, rb=b*2;
        const double rn=(ra-rb)/std::sqrt((ra-rb)*(ra-rb)+4e-5);
        const double rga=2*ra-rn, rgb=2*rb+rn;
        const double rn2=(rga-rgb)/std::sqrt((rga-rgb)*(rga-rgb)+4e-5);
        require(std::abs(int(ranged.image.pixels[0])-int(std::lround((.4488+.1*(ra+rn2)/2)*255)))<=1
                && std::stof(ranged.details.at("image_range"))==2.f, "Explicit SwinIR image range was not applied");
        auto mlpTensors=swinFixture();
        for(auto &tensor:mlpTensors) {
            if(tensor.name.ends_with("mlp.fc1.bias")) tensor.data={1,-1};
            if(tensor.name.ends_with("mlp.fc2.weight")) tensor.data={-4,0,0,-4};
        }
        const auto mlpPath=std::filesystem::path(argv[1])/"mlp.safetensors";
        writeModel(mlpPath,mlpTensors); config.model=mlpPath;
        const auto mlpOutput=upscaleImage({3,3,3,pixels},*makeNativeModelUpscaler(config),UpscaleOptions{});
        const auto gelu=[](double v){return .5*v*(1+std::erf(v/std::sqrt(2.0)));};
        const double ma=ga-4*gelu(1), mb=gb-4*gelu(-1);
        const double mn=(ma-mb)/std::sqrt((ma-mb)*(ma-mb)+4e-5);
        require(std::abs(int(mlpOutput.image.pixels[0])-int(std::lround((.4488+.1*(a+mn))*255)))<=1
                && std::abs(int(mlpOutput.image.pixels[1])-int(std::lround((.4371+.1*(b-mn))*255)))<=1,
                "SwinIR MLP or GELU differs");
        auto normTensors=swinFixture();
        normTensors.push_back({"patch_embed.norm.weight",{2},{1,1}});
        normTensors.push_back({"patch_embed.norm.bias",{2},{0,0}});
        const auto normPath=std::filesystem::path(argv[1])/"patchnorm.safetensors";
        writeModel(normPath,normTensors); config.model=normPath;
        const auto normOutput=upscaleImage({3,3,3,pixels},*makeNativeModelUpscaler(config),UpscaleOptions{});
        const double p0=n,p1=-n;
        const double pn=(p0-p1)/std::sqrt((p0-p1)*(p0-p1)+4e-5);
        const double pg0=2*p0-pn, pg1=2*p1+pn;
        const double pfinal=(pg0-pg1)/std::sqrt((pg0-pg1)*(pg0-pg1)+4e-5);
        require(std::abs(int(normOutput.image.pixels[0])-int(std::lround((.4488+.1*(a+pfinal))*255)))<=1
                && std::abs(int(normOutput.image.pixels[1])-int(std::lround((.4371+.1*(b-pfinal))*255)))<=1,
                "SwinIR patch embedding normalization differs");
        for(const auto &[nearest,four] : std::vector<std::pair<bool,bool>>{{false,false},{true,false},{true,true}}) {
            const auto headPath=std::filesystem::path(argv[1])/(nearest ? (four ? "nearest4.safetensors" : "nearest2.safetensors") : "pixelshuffle.safetensors");
            writeModel(headPath,swinHeadFixture(nearest,four)); config.model=headPath;
            auto head=makeNativeModelUpscaler(config);
            const auto output=upscaleImage({3,3,3,pixels},*head,UpscaleOptions{});
            const int factor=four ? 4 : 2;
            require(output.image.width==std::uint32_t(3*factor) && output.image.height==std::uint32_t(3*factor),
                    "SwinIR reconstruction head scale differs");
            const double featureR=a+n2, featureG=b-n2;
            const double gSlope=nearest ? (four ? .00008 : .0004) : .01;
            const int headR=int(std::lround((.4488+.1*featureR)*255));
            const int headG=int(std::lround((.4371+.1*featureG*gSlope)*255));
            for(std::size_t i=0;i<output.image.pixels.size();i+=3)
                require(std::abs(int(output.image.pixels[i])-headR)<=1 && std::abs(int(output.image.pixels[i+1])-headG)<=1
                        && std::abs(int(output.image.pixels[i+2])-expectedB)<=1,
                        "SwinIR reconstruction head numerical output differs");
        }
        const auto residualPath=std::filesystem::path(argv[1])/"residual3.safetensors";
        writeModel(residualPath,swinResidual3Fixture()); config.model=residualPath;
        const auto residual=upscaleImage({3,3,3,pixels},*makeNativeModelUpscaler(config),UpscaleOptions{});
        for(std::size_t i=0;i<residual.image.pixels.size();i+=3)
            require(std::abs(int(residual.image.pixels[i])-51)<=1 && std::abs(int(residual.image.pixels[i+1])-204)<=1
                    && std::abs(int(residual.image.pixels[i+2])-102)<=1, "SwinIR 3-convolution residual differs");
        auto unsupported=swinFixture(); unsupported.push_back({"absolute_pos_embed",{1,1,2},{0,0}});
        const auto unsupportedPath=std::filesystem::path(argv[1])/"unsupported-swin.safetensors";
        writeModel(unsupportedPath,unsupported); config.model=unsupportedPath;
        bool unsupportedRejected=false;
        try { (void)upscaleImage({3,3,3,pixels},*makeNativeModelUpscaler(config),UpscaleOptions{}); }
        catch(const std::invalid_argument &) {unsupportedRejected=true;}
        require(unsupportedRejected,"Unsupported SwinIR absolute embedding was silently ignored");
        auto missingBlock=swinFixture();
        missingBlock.push_back({"layers.0.residual_group.blocks.2.norm1.weight",{2},{1,1}});
        const auto missingBlockPath=std::filesystem::path(argv[1])/"gapped-swin.safetensors";
        writeModel(missingBlockPath,missingBlock); config.model=missingBlockPath;
        bool missingRejected=false;
        try { (void)upscaleImage({3,3,3,pixels},*makeNativeModelUpscaler(config),UpscaleOptions{}); }
        catch(const std::invalid_argument &) {missingRejected=true;}
        require(missingRejected,"Non-contiguous SwinIR blocks were silently ignored");
        auto shiftedTensors=swinFixture();
        const std::string firstPrefix="layers.0.residual_group.blocks.0.";
        std::vector<Tensor> shiftedCopies;
        for(const auto &tensor:shiftedTensors) if(tensor.name.starts_with(firstPrefix)) {
            auto copy=tensor; copy.name.replace(0,firstPrefix.size(),"layers.0.residual_group.blocks.1.");
            shiftedCopies.push_back(std::move(copy));
        }
        for(auto &tensor:shiftedTensors)
            if(tensor.name==firstPrefix+"attn.qkv.weight") std::fill(tensor.data.begin(),tensor.data.end(),0);
        for(auto &tensor:shiftedCopies) {
            if(tensor.name.ends_with("attn.relative_position_bias_table")) {
                tensor.shape={9,2}; tensor.data.assign(18,0);
                tensor.data[8]=tensor.data[9]=std::log(3.f);
            } else if(tensor.name.ends_with("attn.qkv.weight")) {
                tensor.data[0]=tensor.data[3]=tensor.data[4]=tensor.data[7]=1;
            }
        }
        shiftedTensors.insert(shiftedTensors.end(),shiftedCopies.begin(),shiftedCopies.end());
        const auto shiftedPath=std::filesystem::path(argv[1])/"shifted.safetensors";
        writeModel(shiftedPath,shiftedTensors); config.model=shiftedPath;
        auto shifted=makeNativeModelUpscaler(config);
        std::vector<std::uint8_t> checker;
        for(int y=0;y<4;++y) for(int x=0;x<4;++x) {
            const bool selected=(x==1 && y==1)||(x==2 && y==1)||(x==1 && y==2);
            if(selected) checker.insert(checker.end(),{51,204,102}); else checker.insert(checker.end(),{204,51,102});
        }
        const auto shiftedResult=upscaleImage({4,4,3,checker},*shifted,UpscaleOptions{});
        const double aa=51.0/255-.4488, ab=204.0/255-.4371;
        const double ba=204.0/255-.4488, bb=51.0/255-.4371;
        const auto normalize=[](double x,double y) {return (x-y)/std::sqrt((x-y)*(x-y)+4e-5);};
        const double na=normalize(aa,ab), nb=normalize(ba,bb);
        const auto checkAnalytic=[&](int x,int y,double first,double second,double mean) {
            const double finalNorm=normalize(2*first-mean,2*second+mean);
            const int r=int(std::lround((.4488+.1*(first+finalNorm))*255));
            const int g=int(std::lround((.4371+.1*(second-finalNorm))*255));
            const auto index=std::size_t((y*2*8+x*2)*3);
            require(std::abs(int(shiftedResult.image.pixels[index])-r)<=1
                    && std::abs(int(shiftedResult.image.pixels[index+1])-g)<=1,
                    "Shifted SwinIR attention or boundary mask differs");
        };
        const auto aWeight=5*std::exp(na*na), bWeight=std::exp(na*nb);
        checkAnalytic(1,1,aa,ab,(aWeight*na+bWeight*nb)/(aWeight+bWeight));
        checkAnalytic(0,0,ba,bb,nb);
        auto qkConfig=config; qkConfig.model=shiftedPath; qkConfig.swinIRQkScale=.5f;
        const auto qkResult=upscaleImage({4,4,3,checker},*makeNativeModelUpscaler(qkConfig),UpscaleOptions{});
        const double wa=5*std::exp(.5*na*na), wb=std::exp(.5*na*nb);
        const double qMean=(wa*na+wb*nb)/(wa+wb);
        const double qNorm=normalize(2*aa-qMean,2*ab+qMean);
        const auto qIndex=std::size_t((2*8+2)*3);
        require(std::abs(int(qkResult.image.pixels[qIndex])-int(std::lround((.4488+.1*(aa+qNorm))*255)))<=1
                && std::stof(qkResult.details.at("qk_scales"))==.5f,
                "Explicit SwinIR QK scale was not applied or reported");
        qkConfig.swinIRQkScale=0.0000001f;
        const auto tiny=upscaleImage({4,4,3,checker},*makeNativeModelUpscaler(qkConfig),UpscaleOptions{});
        require(std::stof(tiny.details.at("qk_scales"))==qkConfig.swinIRQkScale,
                "Reported SwinIR QK scale lost float precision");
        std::vector<std::uint8_t> widePixels(45*37*3);
        for(std::size_t i=0;i<widePixels.size();++i) widePixels[i]=std::uint8_t((i*13+i/7)%220);
        const UpscaleImage wide{45,37,3,widePixels};
        auto tiledConfig=config; tiledConfig.model=shiftedPath; tiledConfig.tileSize=0;
        const auto wholeSwin=upscaleImage(wide,*makeNativeModelUpscaler(tiledConfig),UpscaleOptions{});
        tiledConfig.tileSize=7;
        const auto tiledSwin=upscaleImage(wide,*makeNativeModelUpscaler(tiledConfig),UpscaleOptions{});
        require(tiledSwin.image.pixels==wholeSwin.image.pixels && tiledSwin.details.at("tile_size")=="7",
                "Shifted SwinIR tiling changed pixels or lost alignment at a tile edge");
        auto indexed=swinFixture();
        std::vector<float> relativeIndex;
        for(int q=0;q<4;++q) for(int k=0;k<4;++k)
            relativeIndex.push_back(float((q/2-k/2+1)*3+q%2-k%2+1));
        indexed.push_back({"layers.0.residual_group.blocks.0.attn.relative_position_index",{4,4},relativeIndex,"I64"});
        const auto indexedPath=std::filesystem::path(argv[1])/"swin-index.safetensors";
        writeModel(indexedPath,indexed); auto indexConfig=config; indexConfig.model=indexedPath;
        const auto indexResult=upscaleImage({3,3,3,pixels},*makeNativeModelUpscaler(indexConfig),UpscaleOptions{});
        require(indexResult.image.pixels==restored.image.pixels,"A valid stored relative index changed output");
        indexed.back().data[0]=0; writeModel(indexedPath,indexed);
        bool rejectedIndex=false;
        try { (void)upscaleImage({3,3,3,pixels},*makeNativeModelUpscaler(indexConfig),UpscaleOptions{}); }
        catch(const std::exception &) { rejectedIndex=true; }
        require(rejectedIndex,"Nonstandard SwinIR relative positions were silently ignored");
        indexed.back().shape={9,9}; indexed.back().data.clear();
        for(int q=0;q<9;++q) for(int k=0;k<9;++k)
            indexed.back().data.push_back(float((q/3-k/3+2)*5+q%3-k%3+2));
        writeModel(indexedPath,indexed); rejectedIndex=false;
        try { (void)upscaleImage({3,3,3,pixels},*makeNativeModelUpscaler(indexConfig),UpscaleOptions{}); }
        catch(const std::exception &) { rejectedIndex=true; }
        require(rejectedIndex,"An index incompatible with its SwinIR bias table was accepted");
        std::cout << "Native model upscaler tests passed\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
