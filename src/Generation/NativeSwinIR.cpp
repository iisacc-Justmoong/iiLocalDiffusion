#include "NativeModelUpscalerInternal.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <charconv>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace iiLocalDiffusion::native_detail {
namespace {
void stopIfRequested(std::stop_token stop) {
    if(upscaleStopRequested(stop,activeUpscaleStopProbe())) throw UpscaleCancelled();
}
std::size_t squareRoot(std::size_t value) {
    const auto n=std::size_t(std::llround(std::sqrt(double(value))));
    if(!n || n>std::numeric_limits<std::size_t>::max()/n || n*n!=value)
        throw std::invalid_argument("Invalid square model dimension.");
    return n;
}
std::string exactFloat(float value) {
    std::array<char,64> buffer{};
    const auto result=std::to_chars(buffer.data(),buffer.data()+buffer.size(),value,
                                    std::chars_format::general,std::numeric_limits<float>::max_digits10);
    if(result.ec!=std::errc{}) throw std::runtime_error("Cannot report SwinIR float setting.");
    return {buffer.data(),result.ptr};
}
void add(CpuImage &a,const CpuImage &b) {
    if(a.width!=b.width || a.height!=b.height || a.channels!=b.channels)
        throw std::invalid_argument("Incompatible residual shapes.");
    for(std::size_t i=0;i<a.values.size();++i) a.values[i]+=b.values[i];
}
void leaky(CpuImage &a,float slope) { for(auto &v:a.values) if(v<0) v*=slope; }
CpuImage norm(const CpuImage &input,const ModelTensor &gain,const ModelTensor &bias,
              std::uint64_t limit,std::stop_token stop) {
    if(gain.shape!=std::vector<std::size_t>{input.channels} || bias.shape!=gain.shape)
        throw std::invalid_argument("Invalid layer-normalization parameters.");
    auto output=image(input.width,input.height,input.channels,limit);
    for(std::size_t y=0;y<input.height;++y) {
        stopIfRequested(stop);
        for(std::size_t x=0;x<input.width;++x) {
            double mean=0, variance=0;
            for(std::size_t c=0;c<input.channels;++c) mean+=input.at(x,y,c);
            mean/=double(input.channels);
            for(std::size_t c=0;c<input.channels;++c) { const auto d=double(input.at(x,y,c))-mean; variance+=d*d; }
            variance/=double(input.channels);
            const auto inv=1.0/std::sqrt(variance+1e-5);
            for(std::size_t c=0;c<input.channels;++c)
                output.at(x,y,c)=float((double(input.at(x,y,c))-mean)*inv*gain.values[c]+bias.values[c]);
        }
    }
    return output;
}
CpuImage linear(const CpuImage &input,const ModelTensor &matrix,const ModelTensor *bias,
                CpuExecutor &executor,std::uint64_t limit,std::stop_token stop) {
    if(matrix.shape.size()!=2 || matrix.shape[1]!=input.channels
       || (bias && bias->shape!=std::vector<std::size_t>{matrix.shape[0]}))
        throw std::invalid_argument("Invalid linear tensor shapes.");
    if(executor.gpu()) return executor.gpu()->linear(input,matrix,bias,limit,stop);
    auto output=image(input.width,input.height,matrix.shape[0],limit);
    for(std::size_t y=0;y<input.height;++y) {
        stopIfRequested(stop);
        for(std::size_t x=0;x<input.width;++x) for(std::size_t out=0;out<output.channels;++out) {
            float v=bias ? bias->values[out] : 0;
            for(std::size_t in=0;in<input.channels;++in) v+=matrix.values[out*input.channels+in]*input.at(x,y,in);
            output.at(x,y,out)=v;
        }
    }
    return output;
}
CpuImage nearest2(const CpuImage &input,std::uint64_t limit,std::stop_token stop) {
    if(input.width>std::numeric_limits<std::size_t>::max()/2 || input.height>std::numeric_limits<std::size_t>::max()/2)
        throw std::overflow_error("Upsampled tensor dimensions overflow.");
    auto out=image(input.width*2,input.height*2,input.channels,limit);
    for(std::size_t y=0;y<out.height;++y) {
        stopIfRequested(stop);
        for(std::size_t x=0;x<out.width;++x) for(std::size_t c=0;c<out.channels;++c)
            out.at(x,y,c)=input.at(x/2,y/2,c);
    }
    return out;
}
struct Block { std::string prefix; std::size_t window=0,heads=0,shift=0; };
struct Group { std::string prefix; std::vector<Block> blocks; };
class SwinIRModel final : public CpuUpscaleModel {
    ModelWeights weights_;
    NativeModelUpscalerOptions options_;
    std::shared_ptr<CpuExecutor> executor_;
    std::vector<Group> groups_;
    std::size_t channels_=0,window_=0,scale_=0,upChannels_=0;
    enum class Head { Direct, PixelShuffle, Nearest } head_{};
    std::vector<std::pair<std::string,std::size_t>> upsample_;
    const ModelTensor *optional(const std::string &name) const {
        auto it=weights_.find(name); return it==weights_.end() ? nullptr : &it->second;
    }
    void checkNorm(const std::string &prefix,std::size_t c) const {
        const std::array shape{c}; weight(weights_,prefix+"weight",shape); weight(weights_,prefix+"bias",shape);
    }
    void checkLinear(const std::string &prefix,std::size_t in,std::size_t out,bool biasRequired=true) const {
        const std::array shape{out,in}; weight(weights_,prefix+"weight",shape);
        const std::array bshape{out};
        if(biasRequired || weights_.contains(prefix+"bias")) weight(weights_,prefix+"bias",bshape);
    }
    void checkConv(const std::string &prefix,std::size_t in,std::size_t out,std::size_t kernel=3) const {
        const std::array shape{out,in,kernel,kernel}; const std::array bshape{out};
        weight(weights_,prefix+"weight",shape); weight(weights_,prefix+"bias",bshape);
    }
    void checkResidual(const std::string &prefix) const {
        if(weights_.contains(prefix+"weight")) {
            if(weights_.contains(prefix+"0.weight")) throw std::invalid_argument("Ambiguous SwinIR residual convolution.");
            checkConv(prefix,channels_,channels_);
        } else {
            if(channels_%4) throw std::invalid_argument("SwinIR 3-convolution residual requires divisible feature count.");
            const auto mid=channels_/4;
            checkConv(prefix+"0.",channels_,mid); checkConv(prefix+"2.",mid,mid,1); checkConv(prefix+"4.",mid,channels_);
        }
    }
    CpuImage conv(const CpuImage &input,const std::string &prefix,std::stop_token stop) const {
        return convolve(input,weight(weights_,prefix+"weight"),weight(weights_,prefix+"bias"),
                        *executor_,options_.maxWorkingBytes,stop);
    }
    CpuImage normalized(const CpuImage &input,const std::string &prefix,std::stop_token stop) const {
        return norm(input,weight(weights_,prefix+"weight"),weight(weights_,prefix+"bias"),options_.maxWorkingBytes,stop);
    }
    CpuImage projected(const CpuImage &input,const std::string &prefix,std::stop_token stop) const {
        return linear(input,weight(weights_,prefix+"weight"),optional(prefix+"bias"),*executor_,options_.maxWorkingBytes,stop);
    }
    CpuImage residualConv(const CpuImage &input,const std::string &prefix,std::stop_token stop) const {
        if(weights_.contains(prefix+"weight")) return conv(input,prefix,stop);
        auto a=conv(input,prefix+"0.",stop); leaky(a,.2f);
        auto b=conv(a,prefix+"2.",stop); leaky(b,.2f);
        return conv(b,prefix+"4.",stop);
    }
    CpuImage attention(const CpuImage &input,const Block &block,std::stop_token stop) const {
        const auto &p=block.prefix;
        auto normalizedInput=normalized(input,p+"norm1.",stop);
        auto qkv=projected(normalizedInput,p+"attn.qkv.",stop);
        auto attended=image(input.width,input.height,input.channels,options_.maxWorkingBytes);
        const auto &relative=weight(weights_,p+"attn.relative_position_bias_table");
        const auto ws=block.window,n=ws*ws,headDim=channels_/block.heads,shift=block.shift;
        if(n>options_.maxWorkingBytes/sizeof(float)) throw std::runtime_error("Attention buffer exceeds the requested byte limit.");
        std::vector<float> scores(n);
        const auto region=[&](std::size_t pos,std::size_t dimension) {
            if(!shift || pos<dimension-ws) return 0;
            return pos<dimension-shift ? 1 : 2;
        };
        const float scale=options_.swinIRQkScale>0 ? options_.swinIRQkScale : 1.f/std::sqrt(float(headDim));
        for(std::size_t wy=0;wy<input.height;wy+=ws) for(std::size_t wx=0;wx<input.width;wx+=ws) {
            stopIfRequested(stop);
            for(std::size_t head=0;head<block.heads;++head) for(std::size_t qi=0;qi<n;++qi) {
                stopIfRequested(stop);
                const auto qy=wy+qi/ws,qx=wx+qi%ws;
                const auto qsy=(qy+shift)%input.height,qsx=(qx+shift)%input.width;
                float maximum=-std::numeric_limits<float>::infinity();
                for(std::size_t ki=0;ki<n;++ki) {
                    const auto ky=wy+ki/ws,kx=wx+ki%ws;
                    const auto ksy=(ky+shift)%input.height,ksx=(kx+shift)%input.width;
                    float value=0;
                    for(std::size_t d=0;d<headDim;++d) {
                        const auto channel=head*headDim+d;
                        value+=qkv.at(qsx,qsy,channel)*qkv.at(ksx,ksy,channels_+channel);
                    }
                    const auto ry=int(qi/ws)-int(ki/ws)+int(ws)-1;
                    const auto rx=int(qi%ws)-int(ki%ws)+int(ws)-1;
                    const auto relativeIndex=std::size_t(ry)*std::size_t(2*ws-1)+std::size_t(rx);
                    value=value*scale+relative.values[relativeIndex*block.heads+head];
                    if(shift && (region(qy,input.height)!=region(ky,input.height) || region(qx,input.width)!=region(kx,input.width))) value-=100.f;
                    scores[ki]=value; maximum=std::max(maximum,value);
                }
                double denominator=0;
                for(auto &value:scores) { value=std::exp(value-maximum); denominator+=value; }
                if(!(denominator>0) || !std::isfinite(denominator)) throw std::runtime_error("Invalid SwinIR attention scores.");
                for(std::size_t d=0;d<headDim;++d) {
                    const auto channel=head*headDim+d;
                    double value=0;
                    for(std::size_t ki=0;ki<n;++ki) {
                        const auto ky=(wy+ki/ws+shift)%input.height,kx=(wx+ki%ws+shift)%input.width;
                        value+=double(scores[ki])*qkv.at(kx,ky,2*channels_+channel);
                    }
                    attended.at(qsx,qsy,channel)=float(value/denominator);
                }
            }
        }
        return projected(attended,p+"attn.proj.",stop);
    }
    CpuImage runBlock(const CpuImage &input,const Block &block,std::stop_token stop) const {
        auto result=input;
        auto att=attention(input,block,stop); add(result,att);
        auto n=normalized(result,block.prefix+"norm2.",stop);
        auto hidden=projected(n,block.prefix+"mlp.fc1.",stop);
        constexpr float inverseSqrt2=.7071067811865475f;
        for(auto &v:hidden.values) v=.5f*v*(1.f+std::erf(v*inverseSqrt2));
        auto mlp=projected(hidden,block.prefix+"mlp.fc2.",stop); add(result,mlp);
        return result;
    }
public:
    SwinIRModel(ModelWeights weights,NativeModelUpscalerOptions options,std::shared_ptr<CpuExecutor> executor)
        : weights_(std::move(weights)),options_(std::move(options)),executor_(std::move(executor)) {
        if(weights_.contains("absolute_pos_embed")) throw std::invalid_argument("SwinIR absolute position embeddings are not supported.");
        const auto &first=weight(weights_,"conv_first.weight");
        if(first.shape.size()!=4 || first.shape[1]!=3 || first.shape[2]!=3 || first.shape[3]!=3)
            throw std::invalid_argument("Invalid SwinIR input convolution.");
        channels_=first.shape[0]; checkConv("conv_first.",3,channels_);
        if(weights_.contains("patch_embed.norm.weight") || weights_.contains("patch_embed.norm.bias")) checkNorm("patch_embed.norm.",channels_);
        for(std::size_t group=0;group<256;++group) {
            Group g; g.prefix="layers."+std::to_string(group)+".";
            if(!weights_.contains(g.prefix+"residual_group.blocks.0.attn.qkv.weight")) break;
            for(std::size_t index=0;index<256;++index) {
                Block block; block.prefix=g.prefix+"residual_group.blocks."+std::to_string(index)+".";
                if(!weights_.contains(block.prefix+"attn.qkv.weight")) break;
                checkNorm(block.prefix+"norm1.",channels_); checkNorm(block.prefix+"norm2.",channels_);
                checkLinear(block.prefix+"attn.qkv.",channels_,3*channels_,false);
                checkLinear(block.prefix+"attn.proj.",channels_,channels_);
                const auto &relative=weight(weights_,block.prefix+"attn.relative_position_bias_table");
                if(relative.shape.size()!=2 || !relative.shape[1]) throw std::invalid_argument("Invalid SwinIR relative-position bias.");
                const auto side=squareRoot(relative.shape[0]);
                if(side%2!=1) throw std::invalid_argument("Invalid SwinIR window size.");
                block.window=(side+1)/2; block.heads=relative.shape[1];
                if(block.window<2 || block.window>32 || channels_%block.heads) throw std::invalid_argument("Unsupported SwinIR attention shape.");
                if(!window_) window_=block.window;
                if(window_!=block.window) throw std::invalid_argument("Mixed SwinIR window sizes are unsupported.");
                block.shift=index%2 ? block.window/2 : 0;
                const auto &fc=weight(weights_,block.prefix+"mlp.fc1.weight");
                if(fc.shape.size()!=2 || fc.shape[1]!=channels_) throw std::invalid_argument("Invalid SwinIR MLP.");
                checkLinear(block.prefix+"mlp.fc1.",channels_,fc.shape[0]);
                checkLinear(block.prefix+"mlp.fc2.",fc.shape[0],channels_);
                g.blocks.push_back(std::move(block));
            }
            checkResidual(g.prefix+"conv."); groups_.push_back(std::move(g));
        }
        if(groups_.empty()) throw std::invalid_argument("SwinIR transformer blocks are missing.");
        for(const auto &[name,tensor] : weights_) {
            (void)tensor;
            if(!name.starts_with("layers.")) continue;
            const auto dot=name.find('.',7);
            if(dot==std::string::npos) throw std::invalid_argument("Invalid SwinIR group name: "+name);
            std::size_t group=0;
            const auto result=std::from_chars(name.data()+7,name.data()+dot,group);
            if(result.ec!=std::errc{} || result.ptr!=name.data()+dot || group>=groups_.size())
                throw std::invalid_argument("Non-contiguous SwinIR group: "+name);
            const std::string_view rest(name.data()+dot+1,name.size()-dot-1);
            constexpr std::string_view blocks="residual_group.blocks.";
            if(rest.starts_with("residual_group.downsample.")) throw std::invalid_argument("SwinIR downsampling layers are not supported.");
            if(rest.starts_with(blocks)) {
                const auto end=rest.find('.',blocks.size());
                if(end==std::string_view::npos) throw std::invalid_argument("Invalid SwinIR block name: "+name);
                std::size_t block=0;
                const auto blockResult=std::from_chars(rest.data()+blocks.size(),rest.data()+end,block);
                if(blockResult.ec!=std::errc{} || blockResult.ptr!=rest.data()+end || block>=groups_[group].blocks.size())
                    throw std::invalid_argument("Non-contiguous SwinIR block: "+name);
            }
        }
        checkNorm("norm.",channels_); checkResidual("conv_after_body.");
        if(!weights_.contains("conv_before_upsample.0.weight")) {
            head_=Head::Direct;
            const auto &u=weight(weights_,"upsample.0.weight");
            if(u.shape.size()!=4 || u.shape[0]%3) throw std::invalid_argument("Invalid direct pixel-shuffle head.");
            scale_=squareRoot(u.shape[0]/3); if(scale_<2 || scale_>8) throw std::invalid_argument("Unsupported SwinIR output scale.");
            checkConv("upsample.0.",channels_,3*scale_*scale_);
        } else {
            const auto &before=weight(weights_,"conv_before_upsample.0.weight");
            if(before.shape.size()!=4) throw std::invalid_argument("Invalid SwinIR reconstruction head.");
            upChannels_=before.shape[0]; checkConv("conv_before_upsample.0.",channels_,upChannels_);
            if(weights_.contains("conv_up1.weight")) {
                if(weights_.contains("upsample.0.weight")) throw std::invalid_argument("Ambiguous SwinIR reconstruction head.");
                head_=Head::Nearest; scale_=weights_.contains("conv_up2.weight") ? 4 : 2;
                checkConv("conv_up1.",upChannels_,upChannels_);
                if(scale_==4) checkConv("conv_up2.",upChannels_,upChannels_);
                checkConv("conv_hr.",upChannels_,upChannels_);
            } else {
                head_=Head::PixelShuffle; scale_=1;
                for(std::size_t i=0;i<8;i+=2) {
                    const auto prefix="upsample."+std::to_string(i)+".";
                    auto it=weights_.find(prefix+"weight"); if(it==weights_.end()) break;
                    if(it->second.shape.size()!=4 || it->second.shape[0]%upChannels_) throw std::invalid_argument("Invalid SwinIR pixel-shuffle layer.");
                    const auto factor=squareRoot(it->second.shape[0]/upChannels_);
                    if((factor!=2 && factor!=3) || scale_>8/factor) throw std::invalid_argument("Unsupported SwinIR pixel-shuffle scale.");
                    checkConv(prefix,upChannels_,upChannels_*factor*factor); upsample_.push_back({prefix,factor}); scale_*=factor;
                }
                if(upsample_.empty()) throw std::invalid_argument("SwinIR upsampling layers are missing.");
            }
            checkConv("conv_last.",upChannels_,3);
        }
    }
    std::string name() const override { return "swinir"; }
    int scale() const override { return int(scale_); }
    std::size_t tileAlignment() const override { return window_; }
    std::size_t tileHalo() const override {
        std::size_t halo=1; // Initial feature convolution.
        for(const auto &group:groups_)
            halo+=group.blocks.size()*(window_-1)+(weights_.contains(group.prefix+"conv.weight") ? 1 : 2);
        halo+=weights_.contains("conv_after_body.weight") ? 1 : 2;
        halo+=head_==Head::Direct ? 1 : head_==Head::PixelShuffle ? 2+upsample_.size() : 6;
        return halo+window_; // Keep shifted-window wrap and masks outside the copied core.
    }
    std::map<std::string,std::string> details() const override {
        std::string scales;
        for(const auto &group:groups_) for(const auto &block:group.blocks) {
            if(!scales.empty()) scales+=',';
            scales+=exactFloat(options_.swinIRQkScale>0 ? options_.swinIRQkScale :
                               1.f/std::sqrt(float(channels_/block.heads)));
        }
        return {{"image_range",exactFloat(options_.swinIRImageRange)}, {"qk_scales",std::move(scales)}};
    }
    CpuImage run(const CpuImage &input,std::stop_token stop) const override {
        const auto padW=(window_-input.width%window_)%window_,padH=(window_-input.height%window_)%window_;
        if((padW && padW>=input.width) || (padH && padH>=input.height))
            throw std::invalid_argument("SwinIR input is too small for reflect padding.");
        auto padded=image(input.width+padW,input.height+padH,3,options_.maxWorkingBytes);
        constexpr float means[]{.4488f,.4371f,.4040f};
        for(std::size_t y=0;y<padded.height;++y) {
            stopIfRequested(stop);
            for(std::size_t x=0;x<padded.width;++x) for(std::size_t c=0;c<3;++c) {
                const auto sourceX=x<input.width ? x : 2*input.width-2-x;
                const auto sourceY=y<input.height ? y : 2*input.height-2-y;
                padded.at(x,y,c)=(input.at(sourceX,sourceY,c)-means[c])*options_.swinIRImageRange;
            }
        }
        auto shallow=conv(padded,"conv_first.",stop);
        auto current=weights_.contains("patch_embed.norm.weight") ? normalized(shallow,"patch_embed.norm.",stop) : shallow;
        for(const auto &group:groups_) {
            const auto residual=current;
            for(const auto &block:group.blocks) current=runBlock(current,block,stop);
            current=residualConv(current,group.prefix+"conv.",stop); add(current,residual);
        }
        current=normalized(current,"norm.",stop);
        current=residualConv(current,"conv_after_body.",stop); add(current,shallow);
        if(head_==Head::Direct) current=pixelShuffle(conv(current,"upsample.0.",stop),scale_,options_.maxWorkingBytes,stop);
        else {
            current=conv(current,"conv_before_upsample.0.",stop); leaky(current,.01f);
            if(head_==Head::Nearest) {
                current=conv(nearest2(current,options_.maxWorkingBytes,stop),"conv_up1.",stop); leaky(current,.2f);
                if(scale_==4) { current=conv(nearest2(current,options_.maxWorkingBytes,stop),"conv_up2.",stop); leaky(current,.2f); }
                current=conv(current,"conv_hr.",stop); leaky(current,.2f);
            } else for(const auto &[prefix,factor]:upsample_)
                current=pixelShuffle(conv(current,prefix,stop),factor,options_.maxWorkingBytes,stop);
            current=conv(current,"conv_last.",stop);
        }
        auto output=image(input.width*scale_,input.height*scale_,3,options_.maxWorkingBytes);
        for(std::size_t y=0;y<output.height;++y) {
            stopIfRequested(stop);
            for(std::size_t x=0;x<output.width;++x) for(std::size_t c=0;c<3;++c) output.at(x,y,c)=current.at(x,y,c)/options_.swinIRImageRange+means[c];
        }
        return output;
    }
};
}
std::unique_ptr<CpuUpscaleModel> makeSwinIRModel(ModelWeights weights,const NativeModelUpscalerOptions &options,std::shared_ptr<CpuExecutor> executor) {
    return std::make_unique<SwinIRModel>(std::move(weights),options,std::move(executor));
}
}
