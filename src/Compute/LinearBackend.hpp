#pragma once

// Private component boundary shared by the two implemented matrix runtimes.
// Not installed; no general tensor/graph API or backend registry is introduced.
#include "LinearLayer.hpp"

#include <utility>

namespace iild::detail
{

struct LinearBackend
{
    LinearBackend(ComputeInfo selected, LinearMathOptions selectedMath)
        : info{std::move(selected)}, math{selectedMath} {}
    virtual ~LinearBackend() = default;
    virtual std::vector<float> forward(std::span<const float> input, std::size_t batchSize) = 0;

    ComputeInfo info;
    LinearMathOptions math;
    LinearResourceInfo allocation;
    std::size_t inputs = 0;
    std::size_t outputs = 0;
};

[[nodiscard]] std::unique_ptr<LinearBackend> makeTorchLinear(
    std::span<const float> weights, std::size_t inputFeatures, std::size_t outputFeatures,
    std::span<const float> bias, ComputeOptions options, LinearResourceOptions resources,
    LinearMathOptions math);
[[nodiscard]] std::unique_ptr<LinearBackend> makeTorchLinear(
    const std::filesystem::path &file, const std::string &weightKey, const std::string &biasKey,
    ComputeOptions options, LinearResourceOptions resources, LinearMathOptions math);

} // namespace iild::detail
