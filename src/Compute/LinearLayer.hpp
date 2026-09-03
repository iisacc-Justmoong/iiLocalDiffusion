#pragma once

#include "ComputeRuntime.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace iild
{

enum class WeightStorage
{
    device,
    ram
};

struct LinearResourceOptions
{
    // Fraction of output features computed on CPU alongside the selected GPU.
    double cpuShare = 0.0;
    WeightStorage weightStorage = WeightStorage::device;
    // Bound for one staged weight/bias block, not total process/GPU memory.
    std::size_t gpuWeightBudgetBytes = 64 * 1024 * 1024;
};

struct LinearResourceInfo
{
    std::size_t cpuOutputFeatures = 0;
    std::size_t acceleratorOutputFeatures = 0;
    std::size_t hostWeightBytes = 0;
    std::size_t residentAcceleratorWeightBytes = 0;
    std::size_t stagedAcceleratorWeightBytes = 0;
    std::size_t gpuWeightChunkOutputs = 0;
};

enum class LinearPrecision
{
    float32,
    float16,
    bfloat16
};

struct LinearMathOptions
{
    LinearPrecision precision = LinearPrecision::float32;
};

[[nodiscard]] IILD_EXPORT std::string_view linearPrecisionName(LinearPrecision precision);

// A component-level neural operation, not a public tensor or graph abstraction.
// Weight layout is [outputFeatures, inputFeatures]; batches are row-major.
class IILD_EXPORT LinearLayer final
{
public:
    [[nodiscard]] static LinearLayer fromWeights(
        std::span<const float> weights,
        std::size_t inputFeatures,
        std::size_t outputFeatures,
        std::span<const float> bias = {},
        ComputeOptions options = {});

    [[nodiscard]] static LinearLayer fromWeights(
        std::span<const float> weights,
        std::size_t inputFeatures,
        std::size_t outputFeatures,
        std::span<const float> bias,
        ComputeOptions options,
        LinearResourceOptions resources);

    [[nodiscard]] static LinearLayer fromSafetensors(
        const std::filesystem::path &file,
        const std::string &weightKey,
        const std::string &biasKey = {},
        ComputeOptions options = {});

    [[nodiscard]] static LinearLayer fromSafetensors(
        const std::filesystem::path &file,
        const std::string &weightKey,
        const std::string &biasKey,
        ComputeOptions options,
        LinearResourceOptions resources);

    [[nodiscard]] static LinearLayer fromWeights(
        std::span<const float> weights, std::size_t inputFeatures, std::size_t outputFeatures,
        std::span<const float> bias, ComputeOptions options, LinearResourceOptions resources,
        LinearMathOptions math);

    [[nodiscard]] static LinearLayer fromSafetensors(
        const std::filesystem::path &file, const std::string &weightKey, const std::string &biasKey,
        ComputeOptions options, LinearResourceOptions resources, LinearMathOptions math);

    ~LinearLayer();
    LinearLayer(LinearLayer &&) noexcept;
    LinearLayer &operator=(LinearLayer &&) noexcept;
    LinearLayer(const LinearLayer &) = delete;
    LinearLayer &operator=(const LinearLayer &) = delete;

    [[nodiscard]] std::vector<float> forward(
        std::span<const float> input, std::size_t batchSize) const;
    [[nodiscard]] std::size_t inputFeatures() const;
    [[nodiscard]] std::size_t outputFeatures() const;
    [[nodiscard]] const ComputeInfo &computeInfo() const;
    [[nodiscard]] const LinearResourceInfo &resourceInfo() const;
    [[nodiscard]] LinearPrecision precision() const;

private:
    struct Impl;
    explicit LinearLayer(std::unique_ptr<Impl> implementation);
    [[nodiscard]] Impl &implementation() const;
    std::unique_ptr<Impl> implementation_;
};

} // namespace iild
