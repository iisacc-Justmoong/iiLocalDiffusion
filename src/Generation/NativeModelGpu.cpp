#include "NativeModelGpu.hpp"
#include "NativeModelUpscalerInternal.hpp"
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#if IILD_MODEL_METAL_ENABLED
#include <ggml-metal.h>
#endif
#if IILD_MODEL_VULKAN_ENABLED
#include <ggml-vulkan.h>
#endif
#if defined(_WIN32) && IILD_MODEL_DIRECTX_ENABLED
#include <ggml-webgpu.h>
#endif
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace iiLocalDiffusion::native_detail {
namespace {
using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
using Buffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>;
using Backend=std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)>;

Context context() {
    Context ctx(ggml_init({64*ggml_tensor_overhead()+ggml_graph_overhead_custom(64,false),nullptr,true}),ggml_free);
    if(!ctx) throw std::runtime_error("Cannot allocate GPU model graph metadata.");
    return ctx;
}
void checkStop(std::stop_token token) {
    if(upscaleStopRequested(token,activeUpscaleStopProbe())) throw UpscaleCancelled();
}
std::int64_t dim(std::size_t value) {
    if(!value || value>std::size_t(std::numeric_limits<std::int64_t>::max()))
        throw std::invalid_argument("GPU tensor dimension is out of range.");
    return std::int64_t(value);
}
struct ResidentWeight {
    Context ctx{nullptr,ggml_free};
    Buffer buffer{nullptr,ggml_backend_buffer_free};
    ggml_tensor *tensor=nullptr;
};
class GgmlModelGpu final : public ModelGpu {
    std::string name_,description_;
    ggml_backend_dev_t device_=nullptr;
    Backend backend_{nullptr,ggml_backend_free};
    std::unordered_map<const ModelTensor *,ResidentWeight> weights_;
    std::size_t convolutions_=0, projections_=0;
    ggml_tensor *resident(const ModelTensor &weight,bool convolution) {
        auto found=weights_.find(&weight);
        if(found!=weights_.end()) return found->second.tensor;
        ResidentWeight entry;
        entry.ctx=context();
        if(convolution) {
            const auto &s=weight.shape;
            entry.tensor=ggml_new_tensor_4d(entry.ctx.get(),GGML_TYPE_F32,dim(s[3]),dim(s[2]),dim(s[1]),dim(s[0]));
        } else {
            entry.tensor=ggml_new_tensor_2d(entry.ctx.get(),GGML_TYPE_F32,dim(weight.shape[1]),dim(weight.shape[0]));
        }
        entry.buffer.reset(ggml_backend_alloc_ctx_tensors(entry.ctx.get(),backend_.get()));
        if(!entry.buffer) throw std::runtime_error("Cannot allocate GPU model weights.");
        ggml_backend_buffer_set_usage(entry.buffer.get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_tensor_set(entry.tensor,weight.values.data(),0,weight.values.size()*sizeof(float));
        return weights_.emplace(&weight,std::move(entry)).first->second.tensor;
    }
    void execute(ggml_context *ctx,ggml_tensor *input,ggml_tensor *output,
                 const float *inputValues,std::size_t inputCount,float *outputValues,std::size_t outputCount,
                 std::stop_token stop) {
        checkStop(stop);
        if(!ggml_backend_dev_supports_op(device_,output))
            throw std::runtime_error("The selected GPU does not support the requested model operation.");
        auto *graph=ggml_new_graph_custom(ctx,64,false);
        ggml_build_forward_expand(graph,output);
        Buffer buffer(ggml_backend_alloc_ctx_tensors(ctx,backend_.get()),ggml_backend_buffer_free);
        if(!buffer) throw std::runtime_error("Cannot allocate GPU model working tensors.");
        ggml_backend_tensor_set(input,inputValues,0,inputCount*sizeof(float));
        if(ggml_backend_graph_compute(backend_.get(),graph)!=GGML_STATUS_SUCCESS)
            throw std::runtime_error("GPU model operation failed.");
        ggml_backend_tensor_get(output,outputValues,0,outputCount*sizeof(float));
        checkStop(stop);
    }
public:
    explicit GgmlModelGpu(UpscaleDevice requested) {
        ggml_backend_reg_t reg=nullptr;
        switch(requested) {
        case UpscaleDevice::Metal:
#if IILD_MODEL_METAL_ENABLED
            name_="metal"; reg=ggml_backend_metal_reg(); break;
#else
            throw std::runtime_error("The Metal model backend is not enabled for this build.");
#endif
        case UpscaleDevice::Vulkan:
#if IILD_MODEL_VULKAN_ENABLED
            name_="vulkan"; reg=ggml_backend_vk_reg(); break;
#else
            throw std::runtime_error("The Vulkan model backend is not enabled for this build.");
#endif
        case UpscaleDevice::DirectX:
#if defined(_WIN32) && IILD_MODEL_DIRECTX_ENABLED
            name_="directx"; reg=ggml_backend_webgpu_reg(); break;
#else
            throw std::runtime_error("The DirectX model backend is not enabled for this build.");
#endif
        default: throw std::invalid_argument("Explicit GPU device required.");
        }
        if(!reg || !ggml_backend_reg_dev_count(reg))
            throw std::runtime_error("Requested "+name_+" GPU backend is unavailable in this build or on this device.");
        device_=ggml_backend_reg_dev_get(reg,0);
        if(!device_ || (ggml_backend_dev_type(device_)!=GGML_BACKEND_DEVICE_TYPE_GPU
                        && ggml_backend_dev_type(device_)!=GGML_BACKEND_DEVICE_TYPE_IGPU))
            throw std::runtime_error("The selected backend is not a GPU device.");
        backend_.reset(ggml_backend_dev_init(device_,nullptr));
        if(!backend_) throw std::runtime_error("Cannot initialize the requested "+name_+" GPU.");
        const auto *description=ggml_backend_dev_description(device_);
        description_=description ? description : "";
    }
    CpuImage convolve(const CpuImage &input,const ModelTensor &kernel,const ModelTensor &bias,
                      std::uint64_t limit,std::stop_token stop) override {
        checkStop(stop);
        auto output=image(input.width,input.height,kernel.shape[0],limit);
        auto ctx=context();
        auto *x=ggml_new_tensor_4d(ctx.get(),GGML_TYPE_F32,dim(input.width),dim(input.height),dim(input.channels),1);
        auto *w=resident(kernel,true);
        const int padding=int(kernel.shape[2]/2);
        auto *result=ggml_conv_2d_direct(ctx.get(),w,x,1,1,padding,padding,1,1);
        execute(ctx.get(),x,result,input.values.data(),input.values.size(),output.values.data(),output.values.size(),stop);
        for(std::size_t c=0;c<output.channels;++c) for(std::size_t y=0;y<output.height;++y) {
            checkStop(stop);
            for(std::size_t xPos=0;xPos<output.width;++xPos) output.at(xPos,y,c)+=bias.values[c];
        }
        ++convolutions_;
        return output;
    }
    CpuImage linear(const CpuImage &input,const ModelTensor &matrix,const ModelTensor *bias,
                    std::uint64_t limit,std::stop_token stop) override {
        checkStop(stop);
        auto output=image(input.width,input.height,matrix.shape[0],limit);
        const auto pixels=input.width*input.height;
        std::vector<float> packed(input.values.size()),result(output.values.size());
        for(std::size_t y=0;y<input.height;++y) {
            checkStop(stop);
            for(std::size_t x=0;x<input.width;++x) for(std::size_t c=0;c<input.channels;++c) {
                const auto p=y*input.width+x;
                packed[p*input.channels+c]=input.values[c*pixels+p];
            }
        }
        auto ctx=context();
        auto *inputTensor=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,dim(input.channels),dim(pixels));
        auto *w=resident(matrix,false);
        auto *product=ggml_mul_mat(ctx.get(),w,inputTensor);
        execute(ctx.get(),inputTensor,product,packed.data(),packed.size(),result.data(),result.size(),stop);
        for(std::size_t y=0;y<output.height;++y) {
            checkStop(stop);
            for(std::size_t x=0;x<output.width;++x) for(std::size_t c=0;c<output.channels;++c) {
                const auto p=y*output.width+x;
                output.values[c*pixels+p]=result[p*output.channels+c]+(bias ? bias->values[c] : 0.f);
            }
        }
        ++projections_;
        return output;
    }
    std::map<std::string,std::string> details() const override {
        return {{"actual_device",name_},{"gpu_adapter",description_},{"execution","hybrid-cpu-gpu"},
                {"gpu_convolutions",std::to_string(convolutions_)},{"gpu_projections",std::to_string(projections_)}};
    }
};
}
std::unique_ptr<ModelGpu> makeModelGpu(UpscaleDevice device) { return std::make_unique<GgmlModelGpu>(device); }
}
