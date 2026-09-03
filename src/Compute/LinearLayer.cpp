#include "LinearLayer.hpp"
#include "LinearBackend.hpp"

#include <algorithm>
#include <cmath>
#include <future>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

#if IILD_HAS_MLX
#include <mlx/mlx.h>
#include <mlx/version.h>
#endif

namespace iild
{

namespace
{

std::size_t checkedElements(const std::size_t rows, const std::size_t columns)
{
    const auto maximumDimension = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (rows == 0 || columns == 0 || rows > maximumDimension || columns > maximumDimension ||
        rows > std::numeric_limits<std::size_t>::max() / columns / sizeof(float))
    {
        throw std::invalid_argument("Linear dimensions must be positive and fit the runtime");
    }
    return rows * columns;
}

void requireFinite(const std::span<const float> values)
{
    if (!std::all_of(values.begin(), values.end(), [](const float value) {
            return std::isfinite(value);
        }))
    {
        throw std::invalid_argument("Linear inputs and weights must be finite");
    }
}

} // namespace

std::string_view linearPrecisionName(const LinearPrecision precision)
{
    switch (precision)
    {
    case LinearPrecision::float32: return "fp32";
    case LinearPrecision::float16: return "fp16";
    case LinearPrecision::bfloat16: return "bf16";
    }
    throw std::invalid_argument("Unknown linear precision");
}

#if IILD_HAS_MLX
namespace mx = mlx::core;

struct MlxLinearState final : detail::LinearBackend
{
    explicit MlxLinearState(const ComputeOptions options, const LinearResourceOptions resourceOptions,
                            const LinearMathOptions mathOptions)
        : LinearBackend{{"mlx", mx::version(), selectComputeDevice(options.device, computeCapabilities()),
                         options.deviceIndex, {}}, mathOptions},
          device{info.device == ComputeDevice::cpu ? mx::Device::cpu : mx::Device::gpu,
                 checkedIndex(options.deviceIndex)},
          stream{makeStream(device)},
          hostStream{makeStream(mx::Device::cpu)},
          resources{resourceOptions}
    {
        (void)linearPrecisionName(math.precision);
        if (info.device == ComputeDevice::cpu && math.precision != LinearPrecision::float32)
            throw std::invalid_argument("Reduced matrix precision requires a GPU; CPU shards stay float32");
        if (math.precision == LinearPrecision::float16) dtype = mx::float16;
        if (math.precision == LinearPrecision::bfloat16) dtype = mx::bfloat16;
        if (info.device == ComputeDevice::cuda && math.precision == LinearPrecision::bfloat16 &&
            !tensorCoreCapabilities(options.deviceIndex).bf16Eligible)
            throw std::invalid_argument("CUDA bfloat16 matrix execution requires Ampere-or-newer hardware");
        if (!std::isfinite(resources.cpuShare) || resources.cpuShare < 0 || resources.cpuShare >= 1)
        {
            throw std::invalid_argument("CPU share must be finite and in [0, 1)");
        }
        if (resources.weightStorage != WeightStorage::device && resources.weightStorage != WeightStorage::ram)
        {
            throw std::invalid_argument("Unknown linear weight storage");
        }
        if (resources.cpuShare > 0 && info.device == ComputeDevice::cpu)
        {
            throw std::invalid_argument("CPU sharing requires a selected GPU; use CPU directly for CPU-only work");
        }
        const auto &properties = mx::device_info(device);
        const auto found = properties.find("device_name");
        if (found != properties.end())
        {
            if (const auto *name = std::get_if<std::string>(&found->second))
            {
                info.deviceName = *name;
            }
        }
    }

    std::vector<float> forward(std::span<const float> input, std::size_t batchSize) override;

    static int checkedIndex(const std::uint32_t value)
    {
        if (value > static_cast<std::uint32_t>(std::numeric_limits<int>::max()))
        {
            throw std::invalid_argument("Device index is too large");
        }
        return static_cast<int>(value);
    }

    static mx::Stream makeStream(const mx::Device device)
    {
        if (!mx::is_available(device))
        {
            throw std::runtime_error("Requested MLX device index is unavailable");
        }
        return mx::new_thread_unsafe_stream(device);
    }

    void setWeights(mx::array selectedWeights, std::optional<mx::array> selectedBias)
    {
        if (selectedWeights.ndim() != 2)
        {
            throw std::invalid_argument("Linear weight must have shape [out, in]");
        }
        inputs = static_cast<std::size_t>(selectedWeights.shape(1));
        outputs = static_cast<std::size_t>(selectedWeights.shape(0));
        (void)checkedElements(outputs, inputs);
        requireDtype(selectedWeights);
        if (selectedBias)
        {
            if (selectedBias->ndim() != 1 || selectedBias->shape(0) != selectedWeights.shape(0))
            {
                throw std::invalid_argument("Linear bias must have shape [out]");
            }
            requireDtype(*selectedBias);
        }
        if (resources.cpuShare > 0)
        {
            if (outputs < 2)
            {
                throw std::invalid_argument("CPU/GPU sharing requires at least two output features");
            }
            cpuOutputs = std::clamp(static_cast<std::size_t>(
                std::floor(static_cast<double>(outputs) * resources.cpuShare)), std::size_t{1}, outputs - 1);
            cpuStream = makeStream(mx::Device::cpu);
            cpuWeights = materialize(weightRows(selectedWeights, 0, cpuOutputs), *cpuStream);
            if (selectedBias)
            {
                cpuBias = materialize(biasRows(*selectedBias, 0, cpuOutputs), *cpuStream);
            }
        }
        mainOutputs = outputs - cpuOutputs;
        const std::size_t valuesPerOutput = inputs + (selectedBias ? 1 : 0);
        const std::size_t bytesPerOutput = valuesPerOutput * dtype.size();
        const std::size_t cpuBytes = cpuOutputs * valuesPerOutput * sizeof(float);
        const bool onGpu = info.device != ComputeDevice::cpu;
        const bool staged = onGpu && resources.weightStorage == WeightStorage::ram;
        if (staged && resources.gpuWeightBudgetBytes < bytesPerOutput)
        {
            throw std::invalid_argument("GPU weight budget cannot hold one output feature and its bias");
        }
        chunkOutputs = staged ? std::min(mainOutputs, resources.gpuWeightBudgetBytes / bytesPerOutput)
                              : mainOutputs;
        const auto storageStream = staged ? hostStream : stream;
        weights = materialize(weightRows(selectedWeights, cpuOutputs, outputs), storageStream, dtype);
        if (selectedBias)
        {
            bias = materialize(biasRows(*selectedBias, cpuOutputs, outputs), storageStream, dtype);
        }
        allocation.cpuOutputFeatures = onGpu ? cpuOutputs : outputs;
        allocation.acceleratorOutputFeatures = onGpu ? mainOutputs : 0;
        allocation.hostWeightBytes = cpuBytes + ((!onGpu || staged) ? mainOutputs * bytesPerOutput : 0);
        allocation.residentAcceleratorWeightBytes = onGpu && !staged ? mainOutputs * bytesPerOutput : 0;
        allocation.stagedAcceleratorWeightBytes = staged ? chunkOutputs * bytesPerOutput : 0;
        allocation.gpuWeightChunkOutputs = onGpu ? chunkOutputs : 0;
    }

    mx::array weightRows(const mx::array &array, const std::size_t begin, const std::size_t end) const
    {
        return mx::slice(array, {static_cast<int>(begin), 0},
                         {static_cast<int>(end), static_cast<int>(inputs)}, hostStream);
    }

    mx::array biasRows(const mx::array &array, const std::size_t begin, const std::size_t end) const
    {
        return mx::slice(array, {static_cast<int>(begin)}, {static_cast<int>(end)}, hostStream);
    }

    static mx::array materialize(const mx::array &array, const mx::Stream target,
                                 const mx::Dtype type = mx::float32)
    {
        auto result = mx::copy(mx::astype(array, type, target), target);
        mx::eval(result);
        checkFinite(result, target);
        mx::synchronize(target);
        return result;
    }

    static void requireDtype(const mx::array &array)
    {
        if (array.dtype() != mx::float16 && array.dtype() != mx::bfloat16 &&
            array.dtype() != mx::float32)
        {
            throw std::invalid_argument("Linear weights require float16, bfloat16, or float32");
        }
    }

    static void checkFinite(const mx::array &array, const mx::Stream target)
    {
        if (!mx::all(mx::isfinite(array, target), target).item<bool>())
        {
            throw std::invalid_argument("Linear values must be finite in the selected precision");
        }
    }

    static std::vector<float> evaluate(const mx::array &input, const mx::array &weight,
                                       const std::optional<mx::array> &offset, const mx::Stream target)
    {
        auto result = mx::matmul(input, mx::transpose(weight, target), target);
        if (offset)
        {
            result = mx::add(result, *offset, target);
        }
        result = mx::contiguous(result, false, target);
        mx::eval(result);
        mx::synchronize(target);
        auto host = mx::copy(mx::astype(result, mx::float32, mx::Device::cpu), mx::Device::cpu);
        mx::eval(host);
        if (!std::all_of(host.data<float>(), host.data<float>() + host.size(),
                         [](const float value) { return std::isfinite(value); }))
            throw std::runtime_error("Linear computation produced non-finite values; choose a wider precision");
        return {host.data<float>(), host.data<float>() + host.size()};
    }

    mx::Device device;
    mx::Stream stream;
    mx::Stream hostStream;
    LinearResourceOptions resources;
    mx::Dtype dtype = mx::float32;
    std::optional<mx::array> weights;
    std::optional<mx::array> bias;
    std::optional<mx::Stream> cpuStream;
    std::optional<mx::array> cpuWeights;
    std::optional<mx::array> cpuBias;
    std::size_t cpuOutputs = 0;
    std::size_t mainOutputs = 0;
    std::size_t chunkOutputs = 0;
    std::mutex mutex;
};
#endif

struct LinearLayer::Impl
{
    std::unique_ptr<detail::LinearBackend> backend;
};

LinearLayer::LinearLayer(std::unique_ptr<Impl> implementation)
    : implementation_{std::move(implementation)}
{
}

LinearLayer::~LinearLayer() = default;
LinearLayer::LinearLayer(LinearLayer &&) noexcept = default;
LinearLayer &LinearLayer::operator=(LinearLayer &&) noexcept = default;

LinearLayer::Impl &LinearLayer::implementation() const
{
    if (!implementation_)
    {
        throw std::logic_error("Cannot use a moved-from linear layer");
    }
    return *implementation_;
}

LinearLayer LinearLayer::fromWeights(const std::span<const float> weights,
                                    const std::size_t inputFeatures,
                                    const std::size_t outputFeatures,
                                    const std::span<const float> bias,
                                    const ComputeOptions options)
{
    return fromWeights(weights, inputFeatures, outputFeatures, bias, options, {});
}

LinearLayer LinearLayer::fromWeights(const std::span<const float> weights,
                                    const std::size_t inputFeatures,
                                    const std::size_t outputFeatures,
                                    const std::span<const float> bias,
                                    const ComputeOptions options,
                                    const LinearResourceOptions resources)
{
    return fromWeights(weights, inputFeatures, outputFeatures, bias, options, resources, {});
}

LinearLayer LinearLayer::fromWeights(const std::span<const float> weights,
                                    const std::size_t inputFeatures,
                                    const std::size_t outputFeatures,
                                    const std::span<const float> bias,
                                    const ComputeOptions options,
                                    const LinearResourceOptions resources,
                                    const LinearMathOptions math)
{
    if (weights.size() != checkedElements(outputFeatures, inputFeatures) ||
        (!bias.empty() && bias.size() != outputFeatures))
    {
        throw std::invalid_argument("Linear weight or bias element count is incorrect");
    }
    requireFinite(weights);
    requireFinite(bias);
    const auto capabilities = computeCapabilities();
    const ComputeOptions selected{selectComputeDevice(options.device, capabilities, rocmCapabilities()),
                                  options.deviceIndex};
    auto implementation = std::make_unique<Impl>();
    if (selected.device == ComputeDevice::rocm ||
        (selected.device == ComputeDevice::cpu && !capabilities.cpu))
    {
        implementation->backend = detail::makeTorchLinear(weights, inputFeatures, outputFeatures,
                                                          bias, selected, resources, math);
        return LinearLayer{std::move(implementation)};
    }
#if IILD_HAS_MLX
    auto backend = std::make_unique<MlxLinearState>(selected, resources, math);
    std::optional<mx::array> biasArray;
    if (!bias.empty())
    {
        biasArray.emplace(bias.begin(), mx::Shape{static_cast<int>(outputFeatures)});
    }
    backend->setWeights(
        mx::array{weights.begin(), {static_cast<int>(outputFeatures), static_cast<int>(inputFeatures)}},
        std::move(biasArray));
    implementation->backend = std::move(backend);
    return LinearLayer{std::move(implementation)};
#else
    throw std::runtime_error("MLX compute is disabled in this build");
#endif
}

LinearLayer LinearLayer::fromSafetensors(const std::filesystem::path &file,
                                        const std::string &weightKey,
                                        const std::string &biasKey,
                                        const ComputeOptions options)
{
    return fromSafetensors(file, weightKey, biasKey, options, {});
}

LinearLayer LinearLayer::fromSafetensors(const std::filesystem::path &file,
                                        const std::string &weightKey,
                                        const std::string &biasKey,
                                        const ComputeOptions options,
                                        const LinearResourceOptions resources)
{
    return fromSafetensors(file, weightKey, biasKey, options, resources, {});
}

LinearLayer LinearLayer::fromSafetensors(const std::filesystem::path &file,
                                        const std::string &weightKey,
                                        const std::string &biasKey,
                                        const ComputeOptions options,
                                        const LinearResourceOptions resources,
                                        const LinearMathOptions math)
{
    if ((file.extension() != ".safetensors" && file.extension() != ".safetensor") ||
        !std::filesystem::is_regular_file(file) || std::filesystem::file_size(file) == 0 ||
        weightKey.empty())
    {
        throw std::invalid_argument("Select an existing safetensors file and an exact weight key");
    }
    const auto capabilities = computeCapabilities();
    const ComputeOptions selected{selectComputeDevice(options.device, capabilities, rocmCapabilities()),
                                  options.deviceIndex};
    auto implementation = std::make_unique<Impl>();
    if (selected.device == ComputeDevice::rocm ||
        (selected.device == ComputeDevice::cpu && !capabilities.cpu))
    {
        implementation->backend = detail::makeTorchLinear(file, weightKey, biasKey, selected, resources, math);
        return LinearLayer{std::move(implementation)};
    }
#if IILD_HAS_MLX
    auto backend = std::make_unique<MlxLinearState>(selected, resources, math);
    // MLX's file reader is a host operation; arithmetic and conversion below
    // still use the explicitly selected compute stream.
    auto [arrays, metadata] = mx::load_safetensors(file.string(), mx::Device::cpu);
    const auto weight = arrays.find(weightKey);
    if (weight == arrays.end())
    {
        throw std::invalid_argument("Linear weight key is missing: " + weightKey);
    }
    std::optional<mx::array> bias;
    if (!biasKey.empty())
    {
        const auto found = arrays.find(biasKey);
        if (found == arrays.end())
        {
            throw std::invalid_argument("Linear bias key is missing: " + biasKey);
        }
        bias = found->second;
    }
    backend->setWeights(weight->second, std::move(bias));
    implementation->backend = std::move(backend);
    return LinearLayer{std::move(implementation)};
#else
    throw std::runtime_error("MLX compute is disabled in this build");
#endif
}

std::vector<float> LinearLayer::forward(const std::span<const float> input,
                                      const std::size_t batchSize) const
{
    auto &state = *implementation().backend;
    if (input.size() != checkedElements(batchSize, state.inputs))
    {
        throw std::invalid_argument("Linear input must have shape [batch, inputFeatures]");
    }
    (void)checkedElements(batchSize, state.outputs);
    requireFinite(input);
    return state.forward(input, batchSize);
}

#if IILD_HAS_MLX
std::vector<float> MlxLinearState::forward(const std::span<const float> input,
                                         const std::size_t batchSize)
{
    auto &state = *this;
    const std::lock_guard lock{state.mutex};
    std::future<std::vector<float>> cpuWork;
    if (state.cpuOutputs > 0)
    {
        // Independent output columns share only immutable weights/input.
        // A failed GPU call still joins this worker before releasing the layer.
        cpuWork = std::async(std::launch::async, [&state, input, batchSize] {
            auto cpuInput = mx::copy(
                mx::array{input.begin(), {static_cast<int>(batchSize), static_cast<int>(state.inputs)}},
                *state.cpuStream);
            return MlxLinearState::evaluate(cpuInput, *state.cpuWeights, state.cpuBias, *state.cpuStream);
        });
    }
    auto inputArray = MlxLinearState::materialize(
        mx::array{input.begin(), {static_cast<int>(batchSize), static_cast<int>(state.inputs)}},
        state.stream, state.dtype);
    std::vector<float> result(batchSize * state.outputs);
    const auto merge = [&](const std::vector<float> &part, const std::size_t offset,
                           const std::size_t columns) {
        for (std::size_t row = 0; row < batchSize; ++row)
        {
            std::copy_n(part.data() + row * columns, columns,
                        result.data() + row * state.outputs + offset);
        }
    };
    for (std::size_t begin = 0; begin < state.mainOutputs; begin += state.chunkOutputs)
    {
        const std::size_t end = std::min(state.mainOutputs, begin + state.chunkOutputs);
        if (state.allocation.stagedAcceleratorWeightBytes > 0)
        {
            auto block = mx::copy(state.weightRows(*state.weights, begin, end), state.stream);
            std::optional<mx::array> blockBias;
            if (state.bias)
            {
                blockBias = mx::copy(state.biasRows(*state.bias, begin, end), state.stream);
            }
            merge(MlxLinearState::evaluate(inputArray, block, blockBias, state.stream),
                  state.cpuOutputs + begin, end - begin);
        }
        else
        {
            merge(MlxLinearState::evaluate(inputArray, *state.weights, state.bias, state.stream),
                  state.cpuOutputs, state.mainOutputs);
        }
    }
    if (cpuWork.valid())
    {
        merge(cpuWork.get(), 0, state.cpuOutputs);
    }
    return result;
}
#endif

std::size_t LinearLayer::inputFeatures() const { return implementation().backend->inputs; }
std::size_t LinearLayer::outputFeatures() const { return implementation().backend->outputs; }
const ComputeInfo &LinearLayer::computeInfo() const { return implementation().backend->info; }
const LinearResourceInfo &LinearLayer::resourceInfo() const { return implementation().backend->allocation; }
LinearPrecision LinearLayer::precision() const { return implementation().backend->math.precision; }

} // namespace iild
