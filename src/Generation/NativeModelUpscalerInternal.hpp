#pragma once
#include "NativeUpscaler.hpp"
#include "NativeUpscalerInternal.hpp"
#include "NativeModelGpu.hpp"
#include <map>
#include <span>
#include <string>
#include <vector>

namespace iiLocalDiffusion::native_detail {
struct ModelTensor {
    std::vector<std::size_t> shape;
    std::vector<float> values;
};
using ModelWeights = std::map<std::string,ModelTensor>;
struct CpuImage {
    std::size_t width=0,height=0,channels=0;
    std::vector<float> values; // CHW
    float &at(std::size_t x,std::size_t y,std::size_t c) { return values[(c*height+y)*width+x]; }
    float at(std::size_t x,std::size_t y,std::size_t c) const { return values[(c*height+y)*width+x]; }
};
std::size_t safeElements(std::span<const std::size_t> dims);
const ModelTensor &weight(const ModelWeights &,const std::string &,std::span<const std::size_t> expected = {});
CpuImage image(std::size_t w,std::size_t h,std::size_t c,std::uint64_t limit);
class CpuExecutor {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    explicit CpuExecutor(int threads, std::unique_ptr<ModelGpu> gpu = {});
    ~CpuExecutor();
    CpuExecutor(const CpuExecutor &) = delete;
    CpuExecutor &operator=(const CpuExecutor &) = delete;
    void run(std::size_t count, const std::function<void(std::size_t,std::size_t)> &work);
    ModelGpu *gpu() const;
};
CpuImage convolve(const CpuImage &, const ModelTensor &, const ModelTensor &, CpuExecutor &,
                  std::uint64_t limit,std::stop_token);
CpuImage pixelShuffle(const CpuImage &, std::size_t scale,std::uint64_t limit,std::stop_token = {});
struct CpuUpscaleModel {
    virtual ~CpuUpscaleModel() = default;
    virtual std::string name() const = 0;
    virtual int scale() const = 0;
    // Input-space context required to reproduce a whole-image evaluation.
    virtual std::size_t tileHalo() const = 0;
    virtual std::size_t tileAlignment() const { return 1; }
    virtual std::map<std::string,std::string> details() const { return {}; }
    virtual CpuImage run(const CpuImage &, std::stop_token) const = 0;
};
std::unique_ptr<CpuUpscaleModel> makeSwinIRModel(ModelWeights, const NativeModelUpscalerOptions &, std::shared_ptr<CpuExecutor>);
}
