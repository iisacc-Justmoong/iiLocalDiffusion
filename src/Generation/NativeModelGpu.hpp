#pragma once
#include "NativeUpscaler.hpp"
#include <cstdint>
#include <map>
#include <memory>
#include <stop_token>
#include <string>

namespace iiLocalDiffusion::native_detail {
struct CpuImage;
struct ModelTensor;
struct ModelGpu {
    virtual ~ModelGpu() = default;
    virtual CpuImage convolve(const CpuImage &,const ModelTensor &,const ModelTensor &,
                              std::uint64_t,std::stop_token) = 0;
    virtual CpuImage linear(const CpuImage &,const ModelTensor &,const ModelTensor *,
                            std::uint64_t,std::stop_token) = 0;
    virtual std::map<std::string,std::string> details() const = 0;
};
#if IILD_HAS_MODEL_GPU
std::unique_ptr<ModelGpu> makeModelGpu(UpscaleDevice);
#endif
}
