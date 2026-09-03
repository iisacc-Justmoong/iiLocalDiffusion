#pragma once

#include "Export.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace iild
{

enum class ComputeDevice
{
    automatic,
    metal,
    cuda,
    cpu,
    rocm
};

struct ComputeOptions
{
    ComputeDevice device = ComputeDevice::automatic;
    std::uint32_t deviceIndex = 0;
};

struct ComputeCapabilities
{
    bool mlx = false;
    bool metal = false;
    bool cuda = false;
    bool cpu = false;
};

struct ComputeInfo
{
    std::string runtime;
    std::string runtimeVersion;
    ComputeDevice device = ComputeDevice::automatic;
    std::uint32_t deviceIndex = 0;
    std::string deviceName;
};

// Additive discovery API: the existing ComputeCapabilities layout stays intact.
// HIP builds use LibTorch's CUDA namespace internally, never NVIDIA Tensor Cores.
struct RocmCapabilities
{
    bool libtorch = false;
    bool cpu = false;
    bool hip = false;
    bool available = false;
    std::uint32_t deviceCount = 0;
    std::string runtimeVersion;
};

enum class AcceleratorSupport
{
    unavailable,
    unsupported,
    supported,
    unknown
};

struct TensorCoreCapabilities
{
    AcceleratorSupport support = AcceleratorSupport::unavailable;
    std::string deviceName;
    std::uint32_t deviceIndex = 0;
    std::size_t capabilityMajor = 0;
    std::size_t capabilityMinor = 0;
    bool fp16Eligible = false;
    bool bf16Eligible = false;
    bool tf32Eligible = false;
    // Architectural inference, never an assertion about executed instructions.
    bool usageVerified = false;
};

[[nodiscard]] IILD_EXPORT ComputeCapabilities computeCapabilities();
[[nodiscard]] IILD_EXPORT std::string_view computeDeviceName(ComputeDevice device);
[[nodiscard]] IILD_EXPORT ComputeDevice selectComputeDevice(
    ComputeDevice requested, const ComputeCapabilities &capabilities);
[[nodiscard]] IILD_EXPORT ComputeDevice selectComputeDevice(
    ComputeDevice requested, const ComputeCapabilities &capabilities,
    const RocmCapabilities &rocm);
[[nodiscard]] IILD_EXPORT RocmCapabilities rocmCapabilities();
[[nodiscard]] IILD_EXPORT AcceleratorSupport classifyTensorCoreSupport(
    std::string_view deviceName, std::size_t capabilityMajor, std::size_t capabilityMinor);
[[nodiscard]] IILD_EXPORT std::string_view acceleratorSupportName(AcceleratorSupport support);
[[nodiscard]] IILD_EXPORT TensorCoreCapabilities tensorCoreCapabilities(std::uint32_t deviceIndex = 0);

} // namespace iild
