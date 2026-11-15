#include "NativeModelUpscalerInternal.hpp"
#include <json-c/json.h>
#include <algorithm>
#include <atomic>
#include <array>
#include <bit>
#include <cmath>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <fstream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <thread>
#include <utility>

namespace iiLocalDiffusion::native_detail {
std::size_t safeElements(std::span<const std::size_t> dims) {
    std::size_t n=1;
    for (auto d:dims) {
        if (!d || n>std::numeric_limits<std::size_t>::max()/d) throw std::invalid_argument("Invalid or overflowing tensor dimensions.");
        n*=d;
    }
    return n;
}
const ModelTensor &weight(const ModelWeights &weights,const std::string &key,std::span<const std::size_t> expected) {
    auto found=weights.find(key);
    if(found==weights.end()) throw std::invalid_argument("Missing model tensor: "+key);
    if(!expected.empty() && (found->second.shape.size()!=expected.size()
       || !std::equal(expected.begin(),expected.end(),found->second.shape.begin())))
        throw std::invalid_argument("Unexpected tensor shape: "+key);
    return found->second;
}
CpuImage image(std::size_t w,std::size_t h,std::size_t c,std::uint64_t limit) {
    const std::array dims{w,h,c}; const auto n=safeElements(dims);
    if(n>limit/sizeof(float)) throw std::runtime_error("Model working tensor exceeds the requested byte limit.");
    return {w,h,c,std::vector<float>(n)};
}
struct CpuExecutor::Impl {
    std::mutex mutex, serial;
    std::condition_variable available, finished;
    std::vector<std::thread> workers;
    std::unique_ptr<ModelGpu> accelerator;
    std::function<void(std::size_t,std::size_t)> task;
    std::size_t count=0, generation=0, pending=0;
    bool closing=false;
    std::exception_ptr error;
    explicit Impl(int threads,std::unique_ptr<ModelGpu> gpu) : accelerator(std::move(gpu)) {
        try {
            for(int index=1;index<threads;++index) workers.emplace_back([this,index,threads] {
                std::size_t seen=0;
                std::unique_lock lock(mutex);
                for(;;) {
                    available.wait(lock,[&]{ return closing || generation!=seen; });
                    if(closing) return;
                    seen=generation; const auto n=count; const auto *fn=&task;
                    lock.unlock(); std::exception_ptr failure;
                    try { (*fn)(n*std::size_t(index)/std::size_t(threads),n*std::size_t(index+1)/std::size_t(threads)); }
                    catch (...) { failure=std::current_exception(); }
                    lock.lock(); if(failure && !error) error=failure;
                    if(--pending==0) finished.notify_one();
                }
            });
        } catch (...) {
            { const std::lock_guard lock(mutex); closing=true; }
            available.notify_all(); for(auto &worker:workers) if(worker.joinable()) worker.join();
            throw;
        }
    }
    ~Impl() {
        { const std::lock_guard lock(mutex); closing=true; }
        available.notify_all(); for(auto &worker:workers) if(worker.joinable()) worker.join();
    }
};
CpuExecutor::CpuExecutor(int threads,std::unique_ptr<ModelGpu> gpu) : impl_(std::make_unique<Impl>(threads,std::move(gpu))) {}
CpuExecutor::~CpuExecutor() = default;
ModelGpu *CpuExecutor::gpu() const { return impl_->accelerator.get(); }
void CpuExecutor::run(std::size_t count,const std::function<void(std::size_t,std::size_t)> &work) {
    auto &state=*impl_; const std::lock_guard serial(state.serial);
    const auto threads=state.workers.size()+1;
    if(threads==1) { work(0,count); return; }
    {
        const std::lock_guard lock(state.mutex);
        state.count=count; state.task=work; state.pending=state.workers.size(); state.error=nullptr; ++state.generation;
    }
    state.available.notify_all();
    std::exception_ptr failure;
    try { work(0,count/threads); } catch (...) { failure=std::current_exception(); }
    {
        std::unique_lock lock(state.mutex);
        state.finished.wait(lock,[&]{return state.pending==0;});
        if(!failure) failure=state.error;
        state.task={};
    }
    if(failure) std::rethrow_exception(failure);
}
CpuImage convolve(const CpuImage &in,const ModelTensor &kernel,const ModelTensor &bias,CpuExecutor &executor,
                  std::uint64_t limit,std::stop_token stop) {
    if(kernel.shape.size()!=4 || kernel.shape[1]!=in.channels || kernel.shape[2]!=kernel.shape[3]
       || (kernel.shape[2]!=1 && kernel.shape[2]!=3) || bias.shape!=std::vector<std::size_t>{kernel.shape[0]})
        throw std::invalid_argument("Incompatible convolution tensor shapes.");
    if(executor.gpu()) return executor.gpu()->convolve(in,kernel,bias,limit,stop);
    const auto outChannels=kernel.shape[0], k=kernel.shape[2];
    auto out=image(in.width,in.height,outChannels,limit);
    const auto *probe=activeUpscaleStopProbe();
    std::atomic_bool interrupted{false};
    const auto work=[&](std::size_t first,std::size_t end) {
        for(std::size_t oc=first;oc<end;++oc) for(std::size_t y=0;y<in.height;++y) {
            if(upscaleStopRequested(stop,probe)) { interrupted=true; return; }
            for(std::size_t x=0;x<in.width;++x) {
                float sum=bias.values[oc];
                for(std::size_t ic=0;ic<in.channels;++ic) for(std::size_t ky=0;ky<k;++ky) for(std::size_t kx=0;kx<k;++kx) {
                    const auto ix=static_cast<std::int64_t>(x)+static_cast<std::int64_t>(kx)-static_cast<std::int64_t>(k/2);
                    const auto iy=static_cast<std::int64_t>(y)+static_cast<std::int64_t>(ky)-static_cast<std::int64_t>(k/2);
                    if(ix>=0 && iy>=0 && std::size_t(ix)<in.width && std::size_t(iy)<in.height)
                        sum+=kernel.values[((oc*in.channels+ic)*k+ky)*k+kx]*in.at(std::size_t(ix),std::size_t(iy),ic);
                }
                out.at(x,y,oc)=sum;
            }
        }
    };
    executor.run(outChannels,work);
    if(interrupted || upscaleStopRequested(stop,probe)) throw UpscaleCancelled();
    return out;
}
CpuImage pixelShuffle(const CpuImage &in,std::size_t scale,std::uint64_t limit,std::stop_token stop) {
    if(scale<1 || scale>8 || in.channels%(scale*scale)) throw std::invalid_argument("Invalid pixel-shuffle channels.");
    if(in.width>std::numeric_limits<std::size_t>::max()/scale || in.height>std::numeric_limits<std::size_t>::max()/scale)
        throw std::overflow_error("Pixel shuffle dimensions overflow.");
    auto out=image(in.width*scale,in.height*scale,in.channels/(scale*scale),limit);
    for(std::size_t c=0;c<out.channels;++c) for(std::size_t y=0;y<out.height;++y) {
        if(upscaleStopRequested(stop,activeUpscaleStopProbe())) throw UpscaleCancelled();
        for(std::size_t x=0;x<out.width;++x) {
            const auto channel=c*scale*scale+(y%scale)*scale+(x%scale);
            out.at(x,y,c)=in.at(x/scale,y/scale,channel);
        }
    }
    return out;
}
namespace {
float halfToFloat(std::uint16_t bits) {
    const int exponent=int((bits>>10)&31), mantissa=bits&1023;
    float value;
    if(exponent==0) value=std::ldexp(float(mantissa),-24);
    else if(exponent==31) value=mantissa ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
    else value=std::ldexp(float(1024+mantissa),exponent-25);
    return bits&0x8000 ? -value : value;
}
std::uint64_t little64(const std::array<unsigned char,8> &bytes) {
    std::uint64_t n=0; for(unsigned i=0;i<8;++i) n|=std::uint64_t(bytes[i])<<(i*8); return n;
}
struct TensorInfo {
    std::string name,dtype; std::vector<std::size_t> shape; std::uint64_t begin=0,end=0;
};
std::string normalizedTensorName(std::string name) {
    bool removed=true;
    while(removed) {
        removed=false;
        for(const std::string_view prefix : {"params_ema.","params.","module.","net_g.","model."})
            if(name.starts_with(prefix)) { name.erase(0,prefix.size()); removed=true; break; }
    }
    return name;
}
ModelWeights readWeights(const NativeModelUpscalerOptions &options) {
    const auto fileSize=std::filesystem::file_size(options.model);
    if(fileSize<10 || fileSize>options.maxWeightBytes) throw std::runtime_error("Model file exceeds the weight byte limit or is truncated.");
    std::ifstream in(options.model,std::ios::binary); if(!in) throw std::runtime_error("Cannot open model weights.");
    std::array<unsigned char,8> prefix{}; in.read(reinterpret_cast<char*>(prefix.data()),8);
    const auto headerBytes=little64(prefix);
    if(!in || headerBytes<2 || headerBytes>64ULL*1024*1024 || headerBytes>fileSize-8)
        throw std::invalid_argument("Invalid safetensors header length.");
    std::string header(std::size_t(headerBytes),'\0'); in.read(header.data(),std::streamsize(header.size()));
    if(!in) throw std::invalid_argument("Truncated safetensors header.");
    auto tok=std::unique_ptr<json_tokener,decltype(&json_tokener_free)>(json_tokener_new(),json_tokener_free);
    if(!tok) throw std::bad_alloc();
    auto root=std::unique_ptr<json_object,decltype(&json_object_put)>(json_tokener_parse_ex(tok.get(),header.data(),int(header.size())),json_object_put);
    if(json_tokener_get_error(tok.get())!=json_tokener_success || !root || !json_object_is_type(root.get(),json_type_object))
        throw std::invalid_argument("Invalid safetensors JSON header.");
    for(std::size_t i=json_tokener_get_parse_end(tok.get());i<header.size();++i)
        if(!std::isspace(static_cast<unsigned char>(header[i]))) throw std::invalid_argument("Trailing data in safetensors JSON header.");
    std::vector<TensorInfo> info;
    json_object_object_foreach(root.get(), key, entry) {
        const std::string name(key); if(name=="__metadata__") continue;
        if(info.size()>=100000 || !json_object_is_type(entry,json_type_object)) throw std::invalid_argument("Invalid tensor entry.");
        json_object *dtype=nullptr,*shape=nullptr,*offset=nullptr;
        if(!json_object_object_get_ex(entry,"dtype",&dtype) || !json_object_is_type(dtype,json_type_string)
           || !json_object_object_get_ex(entry,"shape",&shape) || !json_object_is_type(shape,json_type_array)
           || !json_object_object_get_ex(entry,"data_offsets",&offset) || !json_object_is_type(offset,json_type_array)
           || json_object_array_length(offset)!=2 || json_object_array_length(shape)==0 || json_object_array_length(shape)>8)
            throw std::invalid_argument("Invalid safetensors metadata for "+name);
        TensorInfo t; t.name=name; t.dtype=json_object_get_string(dtype);
        for(std::size_t i=0;i<json_object_array_length(shape);++i) {
            auto *d=json_object_array_get_idx(shape,i);
            if(!d || !json_object_is_type(d,json_type_int) || json_object_get_int64(d)<=0)
                throw std::invalid_argument("Invalid tensor dimension: "+name);
            const auto dimension=std::uint64_t(json_object_get_int64(d));
            if(dimension>std::numeric_limits<std::size_t>::max()) throw std::invalid_argument("Tensor dimension exceeds address space.");
            t.shape.push_back(std::size_t(dimension));
        }
        auto *a=json_object_array_get_idx(offset,0),*b=json_object_array_get_idx(offset,1);
        if(!a || !b || !json_object_is_type(a,json_type_int) || !json_object_is_type(b,json_type_int)
           || json_object_get_int64(a)<0 || json_object_get_int64(b)<0)
            throw std::invalid_argument("Invalid tensor offsets: "+name);
        t.begin=std::uint64_t(json_object_get_int64(a)); t.end=std::uint64_t(json_object_get_int64(b));
        const auto count=safeElements(t.shape);
        std::size_t item=0;
        if(t.dtype=="F32") item=4; else if(t.dtype=="F16" || t.dtype=="BF16") item=2;
        else if(t.dtype=="I64" && (name.ends_with("relative_position_index") || name.ends_with("attn_mask"))) item=8;
        else throw std::invalid_argument("Unsupported model tensor type: "+name);
        if(count>std::numeric_limits<std::uint64_t>::max()/item || t.end<t.begin || t.end-t.begin!=count*item
           || t.end>fileSize-8-headerBytes) throw std::invalid_argument("Invalid tensor range: "+name);
        info.push_back(std::move(t));
    }
    if(info.empty()) throw std::invalid_argument("Model has no tensors.");
    std::sort(info.begin(),info.end(),[](const auto &a,const auto &b){return a.begin<b.begin;});
    std::uint64_t end=0;
    for(const auto &t:info) { if(t.begin!=end) throw std::invalid_argument("Overlapping or discontinuous model tensor ranges."); end=t.end; }
    if(end!=fileSize-8-headerBytes) throw std::invalid_argument("Model tensor data has trailing bytes.");
    ModelWeights weights;
    std::vector<std::pair<std::string,std::size_t>> relativeIndices;
    const auto dataStart=8+headerBytes;
    std::uint64_t decodedBytes=0;
    for(const auto &t:info) {
        if(upscaleStopRequested({},activeUpscaleStopProbe())) throw UpscaleCancelled();
        if(t.name.ends_with("relative_position_index")) {
            if(t.dtype!="I64" || t.shape.size()!=2 || t.shape[0]!=t.shape[1] || t.shape[0]>1024)
                throw std::invalid_argument("Unsupported SwinIR relative-position index: "+t.name);
            const auto tokens=t.shape[0],window=std::size_t(std::llround(std::sqrt(double(tokens))));
            if(window<2 || window*window!=tokens)
                throw std::invalid_argument("Invalid SwinIR relative-position index shape: "+t.name);
            std::vector<unsigned char> data(std::size_t(t.end-t.begin));
            in.seekg(std::streamoff(dataStart+t.begin)); in.read(reinterpret_cast<char*>(data.data()),std::streamsize(data.size()));
            if(!in) throw std::runtime_error("Truncated SwinIR relative-position index: "+t.name);
            for(std::size_t q=0;q<tokens;++q) {
                if(upscaleStopRequested({},activeUpscaleStopProbe())) throw UpscaleCancelled();
                for(std::size_t k=0;k<tokens;++k) {
                    std::uint64_t actual=0; const auto at=(q*tokens+k)*8;
                    for(unsigned i=0;i<8;++i) actual|=std::uint64_t(data[at+i])<<(8*i);
                    const auto row=int(q/window)-int(k/window)+int(window)-1;
                    const auto col=int(q%window)-int(k%window)+int(window)-1;
                    const auto expected=std::uint64_t(row*(2*int(window)-1)+col);
                    if(actual!=expected) throw std::invalid_argument("Nonstandard SwinIR relative-position index: "+t.name);
                }
            }
            auto biasName=normalizedTensorName(t.name);
            biasName.replace(biasName.size()-std::string_view("relative_position_index").size(),
                             std::string_view("relative_position_index").size(),"relative_position_bias_table");
            relativeIndices.emplace_back(std::move(biasName),window);
            continue;
        }
        if(t.dtype=="I64" || t.name.ends_with("attn_mask")) continue;
        const auto count=safeElements(t.shape), bytes=std::size_t(t.end-t.begin);
        if(count>options.maxWeightBytes/sizeof(float) || decodedBytes>options.maxWeightBytes-count*sizeof(float))
            throw std::runtime_error("Decoded model weights exceed the requested weight byte limit.");
        decodedBytes+=count*sizeof(float);
        std::vector<unsigned char> data(bytes);
        in.seekg(std::streamoff(dataStart+t.begin)); in.read(reinterpret_cast<char*>(data.data()),std::streamsize(bytes));
        if(!in) throw std::runtime_error("Truncated model tensor: "+t.name);
        ModelTensor tensor{t.shape,std::vector<float>(count)};
        for(std::size_t i=0;i<count;++i) {
            float f;
            if(t.dtype=="F32") {
                const auto j=i*4; const auto u=std::uint32_t(data[j])|(std::uint32_t(data[j+1])<<8)|(std::uint32_t(data[j+2])<<16)|(std::uint32_t(data[j+3])<<24);
                f=std::bit_cast<float>(u);
            } else {
                const auto j=i*2; const auto u=std::uint16_t(std::uint16_t(data[j]) | std::uint16_t(std::uint16_t(data[j+1])<<8));
                f=t.dtype=="F16" ? halfToFloat(u) : std::bit_cast<float>(std::uint32_t(u)<<16);
            }
            if(!std::isfinite(f)) throw std::invalid_argument("Model contains a non-finite tensor: "+t.name);
            tensor.values[i]=f;
        }
        auto name=normalizedTensorName(t.name);
        if(!weights.emplace(name,std::move(tensor)).second) throw std::invalid_argument("Ambiguous model tensor name: "+name);
    }
    for(const auto &[name,window]:relativeIndices) {
        const auto &bias=weight(weights,name);
        if(bias.shape.size()!=2 || bias.shape[0]!=(2*window-1)*(2*window-1))
            throw std::invalid_argument("Relative-position index does not match SwinIR bias: "+name);
    }
    return weights;
}
class SrvggModel final : public CpuUpscaleModel {
    ModelWeights weights_; NativeModelUpscalerOptions options_; std::shared_ptr<CpuExecutor> executor_;
    std::vector<std::string> convs_; int scale_ = 0;
public:
    SrvggModel(ModelWeights weights,NativeModelUpscalerOptions options,std::shared_ptr<CpuExecutor> executor)
        : weights_(std::move(weights)),options_(std::move(options)),executor_(std::move(executor)) {
        std::size_t channels=3;
        for(std::size_t index=0; index<2048; index+=2) {
            const auto prefix="body."+std::to_string(index);
            auto it=weights_.find(prefix+".weight"); if(it==weights_.end()) break;
            const auto &w=it->second;
            if(w.shape.size()!=4 || w.shape[1]!=channels || w.shape[2]!=3 || w.shape[3]!=3)
                throw std::invalid_argument("Incompatible SRVGG convolution: "+prefix);
            weight(weights_,prefix+".bias",std::array<std::size_t,1>{w.shape[0]});
            convs_.push_back(prefix); channels=w.shape[0];
            const auto next="body."+std::to_string(index+2)+".weight";
            if(weights_.contains(next) && options_.srvggActivation==SrvggActivation::PRelu)
                weight(weights_,"body."+std::to_string(index+1)+".weight",std::array<std::size_t,1>{channels});
        }
        for(const auto &[name,tensor] : weights_) {
            (void)tensor;
            if(!name.starts_with("body.")) continue;
            const auto dot=name.find('.',5);
            if(dot==std::string::npos) throw std::invalid_argument("Invalid SRVGG tensor name: "+name);
            std::size_t index=0;
            const auto result=std::from_chars(name.data()+5,name.data()+dot,index);
            if(result.ec!=std::errc{} || result.ptr!=name.data()+dot) throw std::invalid_argument("Invalid SRVGG layer index: "+name);
            const auto suffix=name.substr(dot+1);
            if((index%2==0 && (index/2>=convs_.size() || (suffix!="weight" && suffix!="bias")))
               || (index%2==1 && (index/2+1>=convs_.size() || suffix!="weight" || options_.srvggActivation!=SrvggActivation::PRelu)))
                throw std::invalid_argument("Unsupported or non-contiguous SRVGG layer: "+name);
        }
        if(convs_.size()<2 || channels%3) throw std::invalid_argument("Model is not a supported SRVGG network.");
        const auto square=channels/3; const auto factor=std::size_t(std::llround(std::sqrt(double(square))));
        if(factor<2 || factor>4 || factor*factor!=square) throw std::invalid_argument("Unsupported SRVGG output scale.");
        scale_=int(factor);
    }
    std::string name() const override { return "srvgg"; }
    int scale() const override { return scale_; }
    std::size_t tileHalo() const override { return convs_.size(); }
    std::map<std::string,std::string> details() const override {
        const auto name=options_.srvggActivation==SrvggActivation::PRelu ? "prelu" :
            options_.srvggActivation==SrvggActivation::Relu ? "relu" : "leakyrelu";
        return {{"activation",name}};
    }
    CpuImage run(const CpuImage &input,std::stop_token stop) const override {
        auto current=input;
        for(std::size_t layer=0;layer<convs_.size();++layer) {
            if(upscaleStopRequested(stop,activeUpscaleStopProbe())) throw UpscaleCancelled();
            const auto &prefix=convs_[layer];
            current=convolve(current,weight(weights_,prefix+".weight"),weight(weights_,prefix+".bias"),
                             *executor_,options_.maxWorkingBytes,stop);
            if(layer+1<convs_.size()) {
                const ModelTensor *slope=nullptr;
                if(options_.srvggActivation==SrvggActivation::PRelu)
                    slope=&weight(weights_,"body."+std::to_string(layer*2+1)+".weight");
                for(std::size_t c=0;c<current.channels;++c) for(std::size_t y=0;y<current.height;++y) {
                    if(upscaleStopRequested(stop,activeUpscaleStopProbe())) throw UpscaleCancelled();
                    for(std::size_t x=0;x<current.width;++x) {
                        auto &v=current.at(x,y,c);
                        if(v<0) v*=slope ? slope->values[c] : options_.srvggActivation==SrvggActivation::Relu ? 0.f : .1f;
                    }
                }
            }
        }
        auto result=pixelShuffle(current,std::size_t(scale_),options_.maxWorkingBytes,stop);
        if(result.channels!=3) throw std::runtime_error("SRVGG output is not RGB.");
        for(std::size_t c=0;c<3;++c) for(std::size_t y=0;y<result.height;++y) {
            if(upscaleStopRequested(stop,activeUpscaleStopProbe())) throw UpscaleCancelled();
            for(std::size_t x=0;x<result.width;++x) result.at(x,y,c)+=input.at(x/std::size_t(scale_),y/std::size_t(scale_),c);
        }
        return result;
    }
};
class ModelSession final : public UpscalerSession {
    NativeModelUpscalerOptions options_;
    std::shared_ptr<CpuExecutor> executor_;
    std::unique_ptr<CpuUpscaleModel> model_;
    CpuImage runTiled(const CpuImage &input, std::stop_token stop) const {
        const auto tile=std::size_t(options_.tileSize);
        if (!tile || (input.width<=tile && input.height<=tile)) return model_->run(input,stop);
        const auto factor=std::size_t(model_->scale()), halo=model_->tileHalo(), alignment=model_->tileAlignment();
        if (!alignment || input.width>std::numeric_limits<std::size_t>::max()/factor
            || input.height>std::numeric_limits<std::size_t>::max()/factor)
            throw std::overflow_error("Invalid tiled model dimensions.");
        auto result=image(input.width*factor,input.height*factor,3,options_.maxWorkingBytes);
        const auto begin=[&](std::size_t core) { return core>halo ? ((core-halo)/alignment)*alignment : 0; };
        const auto end=[&](std::size_t core,std::size_t limit) {
            if (halo>=limit-core) return limit;
            const auto wanted=core+halo;
            const auto remainder=wanted%alignment;
            if (!remainder) return wanted;
            return alignment-remainder<limit-wanted ? wanted+alignment-remainder : limit;
        };
        for(std::size_t y=0;y<input.height;) {
            const auto nextY=y+std::min(tile,input.height-y), top=begin(y), bottom=end(nextY,input.height);
            for(std::size_t x=0;x<input.width;) {
                if(upscaleStopRequested(stop,activeUpscaleStopProbe())) throw UpscaleCancelled();
                const auto nextX=x+std::min(tile,input.width-x), left=begin(x), right=end(nextX,input.width);
                auto region=image(right-left,bottom-top,3,options_.maxWorkingBytes);
                for(std::size_t c=0;c<3;++c) for(std::size_t row=top;row<bottom;++row) {
                    if(upscaleStopRequested(stop,activeUpscaleStopProbe())) throw UpscaleCancelled();
                    std::copy_n(input.values.data()+(c*input.height+row)*input.width+left,
                                right-left,&region.at(0,row-top,c));
                }
                auto expanded=model_->run(region,stop);
                if(expanded.width!=region.width*factor || expanded.height!=region.height*factor || expanded.channels!=3)
                    throw std::runtime_error("A tiled model returned an invalid tensor shape.");
                for(std::size_t c=0;c<3;++c) for(std::size_t row=y*factor;row<nextY*factor;++row) {
                    if(upscaleStopRequested(stop,activeUpscaleStopProbe())) throw UpscaleCancelled();
                    std::copy_n(&expanded.at((x-left)*factor,row-top*factor,c),
                                (nextX-x)*factor,&result.at(x*factor,row,c));
                }
                x=nextX;
            }
            y=nextY;
        }
        return result;
    }
public:
    ModelSession(NativeModelUpscalerOptions options,ModelWeights weights) : options_(std::move(options)) {
        auto architecture=options_.architecture;
        if(architecture==UpscaleModelArchitecture::Automatic) {
            if(weights.contains("body.0.weight")) architecture=UpscaleModelArchitecture::SRVGG;
            else if(weights.contains("layers.0.residual_group.blocks.0.attn.qkv.weight")) architecture=UpscaleModelArchitecture::SwinIR;
            else throw std::invalid_argument("Unrecognized upscaler model architecture.");
        }
        std::unique_ptr<ModelGpu> gpu;
        if(options_.device!=UpscaleDevice::Cpu && options_.device!=UpscaleDevice::Automatic) {
#if IILD_HAS_MODEL_GPU
            gpu=makeModelGpu(options_.device);
#else
            throw std::runtime_error("The selected GPU backend is not built into this configuration.");
#endif
        }
        executor_=std::make_shared<CpuExecutor>(options_.threads,std::move(gpu));
        if(architecture==UpscaleModelArchitecture::SRVGG) {
            if(options_.swinIRImageRange!=1.f || options_.swinIRQkScale!=0.f)
                throw std::invalid_argument("SwinIR settings cannot be applied to SRVGG.");
            model_=std::make_unique<SrvggModel>(std::move(weights),options_,executor_);
        }
        else if(architecture==UpscaleModelArchitecture::SwinIR) {
            if(options_.srvggActivation!=SrvggActivation::PRelu)
                throw std::invalid_argument("SRVGG activation cannot be applied to SwinIR.");
            model_=makeSwinIRModel(std::move(weights),options_,executor_);
        }
        else throw std::invalid_argument("Unsupported upscaler architecture.");
    }
    double defaultScale() const override { return model_->scale(); }
    std::map<std::string,std::string> details() const override {
        auto details=model_->details();
        details.insert({{"architecture",model_->name()},{"model",options_.model.string()},
                        {"model_scale",std::to_string(model_->scale())},
                        {"threads",std::to_string(options_.threads)}, {"tile_size",std::to_string(options_.tileSize)},
                        {"tile_halo",std::to_string(model_->tileHalo())}, {"tile_alignment",std::to_string(model_->tileAlignment())}});
        if(executor_->gpu()) {
            auto gpuDetails=executor_->gpu()->details();
            details.insert(gpuDetails.begin(),gpuDetails.end());
        } else details.emplace("actual_device","cpu");
        return details;
    }
    UpscaleImage process(const UpscaleImage &input,std::uint32_t w,std::uint32_t h,std::stop_token stop) override {
        const auto scale=std::size_t(model_->scale());
        if(input.channels!=3 || std::uint64_t(input.width)*scale!=w || std::uint64_t(input.height)*scale!=h)
            throw std::invalid_argument("Model requires RGB input and its native output scale.");
        auto data=image(input.width,input.height,3,options_.maxWorkingBytes);
        for(std::size_t y=0;y<input.height;++y) {
            if(upscaleStopRequested(stop,activeUpscaleStopProbe())) throw UpscaleCancelled();
            for(std::size_t x=0;x<input.width;++x) for(std::size_t c=0;c<3;++c)
                data.at(x,y,c)=float(input.pixels[(y*input.width+x)*3+c])/255.f;
        }
        auto output=runTiled(data,stop);
        if(output.width!=w || output.height!=h || output.channels!=3) throw std::runtime_error("Model returned an invalid tensor shape.");
        UpscaleImage result{w,h,3,std::vector<std::uint8_t>(std::size_t(w)*h*3)};
        for(std::size_t y=0;y<h;++y) {
            if(upscaleStopRequested(stop,activeUpscaleStopProbe())) throw UpscaleCancelled();
            for(std::size_t x=0;x<w;++x) for(std::size_t c=0;c<3;++c) {
                const auto value=output.at(x,y,c);
                if(!std::isfinite(value)) throw std::runtime_error("Model produced a non-finite pixel.");
                result.pixels[(y*w+x)*3+c]=std::uint8_t(std::lround(std::clamp(value,0.f,1.f)*255.f));
            }
        }
        return result;
    }
};
class ModelUpscaler final : public Upscaler {
    NativeModelUpscalerOptions options_;
public:
    explicit ModelUpscaler(NativeModelUpscalerOptions options) : options_(std::move(options)) {
        auto ext=options_.model.extension().string();
        std::transform(ext.begin(),ext.end(),ext.begin(),[](unsigned char c){ return char(std::tolower(c)); });
        if(ext!=".safetensors") throw std::invalid_argument("This model provider requires .safetensors weights.");
        if(options_.device<UpscaleDevice::Automatic || options_.device>UpscaleDevice::Vulkan)
            throw std::invalid_argument("Invalid model device.");
        if(options_.tileSize<0 || options_.tileSize>8192 || options_.threads<1 || options_.threads>256 || !options_.maxWeightBytes || !options_.maxWorkingBytes)
            throw std::invalid_argument("Invalid model resource options.");
        if(!std::isfinite(options_.swinIRImageRange) || options_.swinIRImageRange<=0 ||
           !std::isfinite(options_.swinIRQkScale) || options_.swinIRQkScale<0)
            throw std::invalid_argument("Invalid SwinIR image range or QK scale.");
        if(options_.srvggActivation!=SrvggActivation::PRelu && options_.srvggActivation!=SrvggActivation::Relu
           && options_.srvggActivation!=SrvggActivation::LeakyRelu) throw std::invalid_argument("Invalid SRVGG activation.");
    }
    std::string name() const override {
        switch(options_.architecture) {
            case UpscaleModelArchitecture::Automatic: return "native-model";
            case UpscaleModelArchitecture::SRVGG: return "srvgg";
            case UpscaleModelArchitecture::SwinIR: return "swinir";
        }
        throw std::invalid_argument("Invalid model architecture.");
    }
    void validateInput(const UpscaleImage &image) const override { if(image.channels!=3) throw std::invalid_argument("The model requires RGB input."); }
    std::unique_ptr<UpscalerSession> open(const UpscaleOptions &options) override {
        if(!options.settings.empty()) throw std::invalid_argument("Unknown native model settings.");
        return std::make_unique<ModelSession>(options_,readWeights(options_));
    }
};
}
}
namespace iiLocalDiffusion {
std::unique_ptr<Upscaler> makeNativeModelUpscaler(NativeModelUpscalerOptions options) {
    return std::make_unique<native_detail::ModelUpscaler>(std::move(options));
}
}
