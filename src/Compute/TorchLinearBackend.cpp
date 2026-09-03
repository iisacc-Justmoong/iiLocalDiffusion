#include "LinearBackend.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <future>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>

#if IILD_HAS_LIBTORCH
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/InferenceMode.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <torch/cuda.h>
#include <torch/version.h>
#include <safetensors.hh>
#endif

namespace iild
{

RocmCapabilities rocmCapabilities()
{
    RocmCapabilities result;
#if IILD_HAS_LIBTORCH
    result.libtorch = true;
    result.cpu = true;
    result.runtimeVersion = TORCH_VERSION;
    // hasROCM(), not hasCUDA(): HIP deliberately uses the CUDA tensor namespace.
    result.hip = at::Context::hasROCM();
    if (result.hip)
    {
        result.deviceCount = static_cast<std::uint32_t>(torch::cuda::device_count());
        result.available = result.deviceCount > 0;
    }
#endif
    return result;
}

namespace detail
{
#if IILD_HAS_LIBTORCH
namespace
{

at::ScalarType scalarType(const LinearPrecision precision)
{
    switch (precision)
    {
    case LinearPrecision::float32: return at::kFloat;
    case LinearPrecision::float16: return at::kHalf;
    case LinearPrecision::bfloat16: return at::kBFloat16;
    }
    throw std::invalid_argument("Unknown linear precision");
}

std::int64_t dimension(const std::size_t value)
{
    if (value == 0 || value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("Linear dimensions must be positive and fit the runtime");
    return static_cast<std::int64_t>(value);
}

at::Device selectTorchDevice(const ComputeOptions options)
{
    if (options.device == ComputeDevice::cpu)
    {
        if (options.deviceIndex != 0)
            throw std::runtime_error("Requested LibTorch CPU device index is unavailable");
        return at::Device{at::kCPU};
    }
    const auto capabilities = rocmCapabilities();
    if (options.device != ComputeDevice::rocm || !capabilities.available ||
        options.deviceIndex >= capabilities.deviceCount ||
        options.deviceIndex > static_cast<std::uint32_t>(std::numeric_limits<c10::DeviceIndex>::max()))
        throw std::runtime_error("Requested ROCm device index is unavailable; no CPU fallback is allowed");
    return at::Device{at::kCUDA, static_cast<c10::DeviceIndex>(options.deviceIndex)};
}

void finite(const at::Tensor &values)
{
    if (!at::isfinite(values).all().item<bool>())
        throw std::invalid_argument("Linear values must be finite in the selected precision");
}

at::Tensor copyRows(const at::Tensor &values, const std::size_t first, const std::size_t count,
                    const at::Device device, const at::ScalarType dtype)
{
    auto result = values.narrow(0, static_cast<std::int64_t>(first), dimension(count)).to(
        at::TensorOptions{}.device(device).dtype(dtype), false, true).contiguous();
    if (result.device() != device) throw std::runtime_error("LibTorch placed weights on the wrong device");
    finite(result);
    return result;
}

struct TorchLinearState final : LinearBackend
{
    TorchLinearState(const ComputeOptions options, const LinearResourceOptions selectedResources,
                     const LinearMathOptions selectedMath)
        : LinearBackend{{"libtorch", TORCH_VERSION, options.device, options.deviceIndex,
                         options.device == ComputeDevice::rocm
                             ? "AMD ROCm device " + std::to_string(options.deviceIndex) : "CPU"}, selectedMath},
          device{selectTorchDevice(options)}, dtype{scalarType(selectedMath.precision)},
          resources{selectedResources}
    {
        if (device.is_cpu() && dtype != at::kFloat)
            throw std::invalid_argument("Reduced matrix precision requires a GPU; CPU shards stay float32");
        if (!std::isfinite(resources.cpuShare) || resources.cpuShare < 0 || resources.cpuShare >= 1)
            throw std::invalid_argument("CPU share must be finite and in [0, 1)");
        if (resources.weightStorage != WeightStorage::device && resources.weightStorage != WeightStorage::ram)
            throw std::invalid_argument("Unknown linear weight storage");
        if (resources.cpuShare > 0 && device.is_cpu())
            throw std::invalid_argument("CPU sharing requires a selected GPU; use CPU directly for CPU-only work");
        // Actually execute the requested dtype before retaining any model weights.
        // Unsupported GPU/driver/dtype combinations fail here, never on CPU retry.
        const c10::InferenceMode inference;
        const c10::DeviceGuard guard{device};
        const c10::impl::ExcludeDispatchKeyGuard noAutocast{c10::autocast_dispatch_keyset};
        const auto probe = at::ones({16, 16}, at::TensorOptions{}.device(device).dtype(dtype));
        const auto result = at::matmul(probe, probe);
        if (result.device() != device || result.scalar_type() != dtype || !result.eq(16).all().item<bool>())
            throw std::runtime_error("LibTorch arithmetic preflight returned an incorrect device/result");
    }

    void setWeights(const at::Tensor &sourceWeights, const std::optional<at::Tensor> &sourceBias)
    {
        const c10::InferenceMode inference;
        const c10::DeviceGuard guard{device};
        if (sourceWeights.dim() != 2)
            throw std::invalid_argument("Linear weight must have shape [out, in]");
        inputs = static_cast<std::size_t>(sourceWeights.size(1));
        outputs = static_cast<std::size_t>(sourceWeights.size(0));
        (void)dimension(inputs);
        (void)dimension(outputs);
        if (outputs > std::numeric_limits<std::size_t>::max() / inputs / sizeof(float))
            throw std::invalid_argument("Linear weight shape overflows host storage");
        if (sourceBias && (sourceBias->dim() != 1 || sourceBias->size(0) != sourceWeights.size(0)))
            throw std::invalid_argument("Linear bias must have shape [out]");
        finite(sourceWeights);
        if (sourceBias) finite(*sourceBias);
        if (resources.cpuShare > 0)
        {
            if (outputs < 2)
                throw std::invalid_argument("CPU/GPU sharing requires at least two output features");
            cpuOutputs = std::clamp(static_cast<std::size_t>(
                std::floor(static_cast<double>(outputs) * resources.cpuShare)), std::size_t{1}, outputs - 1);
            cpuWeights = copyRows(sourceWeights, 0, cpuOutputs, at::Device{at::kCPU}, at::kFloat);
            if (sourceBias) cpuBias = copyRows(*sourceBias, 0, cpuOutputs, at::Device{at::kCPU}, at::kFloat);
        }
        mainOutputs = outputs - cpuOutputs;
        const std::size_t valuesPerOutput = inputs + (sourceBias ? 1U : 0U);
        const std::size_t bytesPerOutput = valuesPerOutput * c10::elementSize(dtype);
        const bool onGpu = !device.is_cpu();
        const bool staged = onGpu && resources.weightStorage == WeightStorage::ram;
        if (staged && resources.gpuWeightBudgetBytes < bytesPerOutput)
            throw std::invalid_argument("GPU weight budget cannot hold one output feature and its bias");
        chunkOutputs = staged ? std::min(mainOutputs, resources.gpuWeightBudgetBytes / bytesPerOutput)
                              : mainOutputs;
        const at::Device storage = staged ? at::Device{at::kCPU} : device;
        weights = copyRows(sourceWeights, cpuOutputs, mainOutputs, storage, dtype);
        if (sourceBias) bias = copyRows(*sourceBias, cpuOutputs, mainOutputs, storage, dtype);
        allocation.cpuOutputFeatures = onGpu ? cpuOutputs : outputs;
        allocation.acceleratorOutputFeatures = onGpu ? mainOutputs : 0;
        allocation.hostWeightBytes = cpuOutputs * valuesPerOutput * sizeof(float) +
            ((!onGpu || staged) ? mainOutputs * bytesPerOutput : 0);
        allocation.residentAcceleratorWeightBytes = onGpu && !staged ? mainOutputs * bytesPerOutput : 0;
        allocation.stagedAcceleratorWeightBytes = staged ? chunkOutputs * bytesPerOutput : 0;
        allocation.gpuWeightChunkOutputs = onGpu ? chunkOutputs : 0;
    }

    static std::vector<float> evaluate(const at::Tensor &input, const at::Tensor &weight,
                                       const std::optional<at::Tensor> &offset)
    {
        // Respect this component's explicit dtype without mutating the host
        // application's autocast settings. The guard restores its TLS state.
        const c10::impl::ExcludeDispatchKeyGuard noAutocast{c10::autocast_dispatch_keyset};
        auto result = at::matmul(input, weight.transpose(0, 1));
        if (offset) result = at::add(result, *offset);
        if (result.device() != input.device() || result.scalar_type() != input.scalar_type())
            throw std::runtime_error("LibTorch computed on the wrong device or dtype");
        // Blocking readback is also the completion boundary for ROCm operations.
        auto host = result.to(at::TensorOptions{}.device(at::kCPU).dtype(at::kFloat), false, true).contiguous();
        const auto *data = host.const_data_ptr<float>();
        const auto count = static_cast<std::size_t>(host.numel());
        if (!std::all_of(data, data + count, [](const float value) { return std::isfinite(value); }))
            throw std::runtime_error("Linear computation produced non-finite values; choose a wider precision");
        return {data, data + count};
    }

    std::vector<float> forward(const std::span<const float> input, const std::size_t batchSize) override
    {
        const std::lock_guard lock{mutex};
        const c10::InferenceMode inference;
        const c10::DeviceGuard guard{device};
        auto hostInput = at::from_blob(const_cast<float *>(input.data()),
            {dimension(batchSize), dimension(inputs)}, at::TensorOptions{}.dtype(at::kFloat));
        std::future<std::vector<float>> cpuWork;
        if (cpuOutputs > 0)
        {
            cpuWork = std::async(std::launch::async, [this, hostInput] {
                const c10::InferenceMode cpuInference;
                return evaluate(hostInput, cpuWeights, cpuBias);
            });
        }
        auto acceleratorInput = hostInput.to(at::TensorOptions{}.device(device).dtype(dtype), false, true);
        finite(acceleratorInput);
        std::vector<float> result(batchSize * outputs);
        const auto merge = [&](const std::vector<float> &part, const std::size_t offset,
                               const std::size_t columns) {
            for (std::size_t row = 0; row < batchSize; ++row)
                std::copy_n(part.data() + row * columns, columns, result.data() + row * outputs + offset);
        };
        for (std::size_t first = 0; first < mainOutputs; first += chunkOutputs)
        {
            const auto columns = std::min(chunkOutputs, mainOutputs - first);
            if (allocation.stagedAcceleratorWeightBytes > 0)
            {
                const auto block = copyRows(weights, first, columns, device, dtype);
                std::optional<at::Tensor> blockBias;
                if (bias) blockBias = copyRows(*bias, first, columns, device, dtype);
                merge(evaluate(acceleratorInput, block, blockBias), cpuOutputs + first, columns);
            }
            else
                merge(evaluate(acceleratorInput, weights, bias), cpuOutputs, mainOutputs);
        }
        if (cpuWork.valid()) merge(cpuWork.get(), 0, cpuOutputs);
        return result;
    }

    at::Device device;
    at::ScalarType dtype;
    LinearResourceOptions resources;
    at::Tensor weights;
    std::optional<at::Tensor> bias;
    at::Tensor cpuWeights;
    std::optional<at::Tensor> cpuBias;
    std::size_t cpuOutputs = 0;
    std::size_t mainOutputs = 0;
    std::size_t chunkOutputs = 0;
    std::mutex mutex;
};

void validateStorage(const safetensors::safetensors_t &file)
{
    if constexpr (std::endian::native != std::endian::little)
        throw std::runtime_error("The safetensors reader requires a little-endian host");
    // The external reader documents incomplete shape checks. Bound every shape
    // before asking it to validate offsets, including tensors not being selected.
    std::vector<std::pair<std::size_t, std::size_t>> intervals;
    for (const auto &key : file.tensors.keys())
    {
        safetensors::tensor_t tensor;
        if (!file.tensors.at(key, &tensor)) throw std::invalid_argument("Missing tensor metadata");
        std::size_t count = 1;
        for (const auto size : tensor.shape)
        {
            if (size > 0 && count > std::numeric_limits<std::size_t>::max() / size)
                throw std::invalid_argument("Safetensors shape overflows storage");
            count *= size;
        }
        const auto bytes = safetensors::get_dtype_bytes(tensor.dtype);
        const auto first = tensor.data_offsets[0];
        const auto last = tensor.data_offsets[1];
        if (bytes == 0 || count > std::numeric_limits<std::size_t>::max() / bytes ||
            last < first || last > file.databuffer_size || last - first != count * bytes)
            throw std::invalid_argument("Safetensors shape/offset byte counts do not match");
        intervals.emplace_back(first, last);
    }
    // Upstream checks each interval's size, not full coverage or aliasing.
    // The safetensors contract permits neither holes nor overlapping ranges.
    std::sort(intervals.begin(), intervals.end());
    std::size_t end = 0;
    for (const auto &[first, last] : intervals)
    {
        if (first != end)
            throw std::invalid_argument("Safetensors data ranges overlap or contain a gap");
        end = last;
    }
    if (end != file.databuffer_size)
        throw std::invalid_argument("Safetensors contains unreferenced trailing data");
    std::string error;
    if (!safetensors::validate_data_offsets(file, error))
        throw std::invalid_argument("Invalid safetensors offsets: " + error);
}

at::Tensor readTensor(const safetensors::safetensors_t &file, const std::string &key,
                      const std::size_t expectedRank)
{
    safetensors::tensor_t tensor;
    if (!file.tensors.at(key, &tensor)) throw std::invalid_argument("Linear tensor key is missing: " + key);
    if (tensor.shape.size() != expectedRank)
        throw std::invalid_argument("Linear tensor has an incorrect rank: " + key);
    at::ScalarType dtype;
    switch (tensor.dtype)
    {
    case safetensors::kFLOAT32: dtype = at::kFloat; break;
    case safetensors::kFLOAT16: dtype = at::kHalf; break;
    case safetensors::kBFLOAT16: dtype = at::kBFloat16; break;
    default: throw std::invalid_argument("Linear weights require float16, bfloat16, or float32");
    }
    std::vector<std::int64_t> shape;
    for (const auto size : tensor.shape) shape.push_back(dimension(size));
    auto value = at::empty(shape, at::TensorOptions{}.device(at::kCPU).dtype(dtype));
    const auto bytes = tensor.data_offsets[1] - tensor.data_offsets[0];
    // Avoid alignment/lifetime assumptions about the mapped file. LibTorch owns
    // the destination storage and all FP16/BF16 conversion, not our file reader.
    std::memcpy(value.mutable_data_ptr(), file.databuffer_addr + tensor.data_offsets[0], bytes);
    return value.to(at::kFloat);
}

} // namespace

std::unique_ptr<LinearBackend> makeTorchLinear(
    const std::span<const float> weights, const std::size_t inputs, const std::size_t outputs,
    const std::span<const float> bias, const ComputeOptions options,
    const LinearResourceOptions resources, const LinearMathOptions math)
{
    auto result = std::make_unique<TorchLinearState>(options, resources, math);
    const c10::InferenceMode inference;
    const auto weight = at::from_blob(const_cast<float *>(weights.data()),
        {dimension(outputs), dimension(inputs)}, at::TensorOptions{}.dtype(at::kFloat));
    std::optional<at::Tensor> offset;
    if (!bias.empty()) offset = at::from_blob(const_cast<float *>(bias.data()),
        {dimension(outputs)}, at::TensorOptions{}.dtype(at::kFloat));
    result->setWeights(weight, offset);
    return result;
}

std::unique_ptr<LinearBackend> makeTorchLinear(
    const std::filesystem::path &path, const std::string &weightKey, const std::string &biasKey,
    const ComputeOptions options, const LinearResourceOptions resources, const LinearMathOptions math)
{
    auto result = std::make_unique<TorchLinearState>(options, resources, math);
    safetensors::safetensors_t file;
    std::string warning, error;
    const auto utf8 = path.u8string();
    const std::string filename{reinterpret_cast<const char *>(utf8.data()), utf8.size()};
    if (!safetensors::mmap_from_file(filename, &file, &warning, &error) || !warning.empty())
        throw std::invalid_argument("Cannot read safetensors: " + error + warning);
    validateStorage(file);
    const c10::InferenceMode inference;
    const auto weights = readTensor(file, weightKey, 2);
    std::optional<at::Tensor> bias;
    if (!biasKey.empty()) bias = readTensor(file, biasKey, 1);
    result->setWeights(weights, bias);
    return result;
}
#else
std::unique_ptr<LinearBackend> makeTorchLinear(
    std::span<const float>, std::size_t, std::size_t, std::span<const float>,
    ComputeOptions, LinearResourceOptions, LinearMathOptions)
{
    throw std::runtime_error("LibTorch computation is disabled; configure IILD_ENABLE_LIBTORCH=ON");
}

std::unique_ptr<LinearBackend> makeTorchLinear(
    const std::filesystem::path &, const std::string &, const std::string &,
    ComputeOptions, LinearResourceOptions, LinearMathOptions)
{
    throw std::runtime_error("LibTorch computation is disabled; configure IILD_ENABLE_LIBTORCH=ON");
}
#endif
} // namespace detail
} // namespace iild
