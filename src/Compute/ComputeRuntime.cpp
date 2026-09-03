#include "ComputeRuntime.hpp"

#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <limits>
#include <variant>

#if IILD_HAS_MLX
#include <mlx/backend/cuda/cuda.h>
#include <mlx/backend/metal/metal.h>
#include <mlx/device.h>
#endif

namespace iild
{

AcceleratorSupport classifyTensorCoreSupport(const std::string_view deviceName,
                                             const std::size_t major, const std::size_t minor)
{
    std::string name{deviceName};
    std::transform(name.begin(), name.end(), name.begin(), [](const unsigned char value) {
        return static_cast<char>(std::toupper(value));
    });
    if (major < 7 || (major == 7 && minor == 5 && name.find("GTX 16") != std::string::npos))
    {
        return AcceleratorSupport::unsupported;
    }
    if (major >= 8 || (major == 7 && (minor == 0 || minor == 2)) ||
        (major == 7 && minor == 5 && (name.find("RTX") != std::string::npos ||
            name == "T4" || name == "TESLA T4" || name == "NVIDIA T4" || name == "NVIDIA TESLA T4")))
    {
        return AcceleratorSupport::supported;
    }
    return AcceleratorSupport::unknown;
}

std::string_view acceleratorSupportName(const AcceleratorSupport support)
{
    switch (support)
    {
    case AcceleratorSupport::unavailable: return "unavailable";
    case AcceleratorSupport::unsupported: return "unsupported";
    case AcceleratorSupport::supported: return "supported";
    case AcceleratorSupport::unknown: return "unknown";
    }
    throw std::invalid_argument("Unknown accelerator support status");
}

TensorCoreCapabilities tensorCoreCapabilities(const std::uint32_t deviceIndex)
{
    TensorCoreCapabilities result;
    result.deviceIndex = deviceIndex;
#if IILD_HAS_MLX
    if (mlx::core::cu::is_available())
    {
        if (deviceIndex > static_cast<std::uint32_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("CUDA device index is too large");
        const auto &properties = mlx::core::device_info(
            mlx::core::Device{mlx::core::Device::gpu, static_cast<int>(deviceIndex)});
        result.deviceName = std::get<std::string>(properties.at("device_name"));
        result.capabilityMajor = std::get<std::size_t>(properties.at("compute_capability_major"));
        result.capabilityMinor = std::get<std::size_t>(properties.at("compute_capability_minor"));
        result.support = classifyTensorCoreSupport(result.deviceName, result.capabilityMajor,
                                                   result.capabilityMinor);
        result.fp16Eligible = result.support == AcceleratorSupport::supported;
        result.bf16Eligible = result.fp16Eligible && result.capabilityMajor >= 8;
        result.tf32Eligible = result.bf16Eligible;
    }
#endif
    return result;
}

ComputeCapabilities computeCapabilities()
{
#if IILD_HAS_MLX
    return {true, mlx::core::metal::is_available(), mlx::core::cu::is_available(),
            mlx::core::is_available(mlx::core::Device::cpu)};
#else
    return {};
#endif
}

std::string_view computeDeviceName(const ComputeDevice device)
{
    switch (device)
    {
    case ComputeDevice::automatic: return "auto";
    case ComputeDevice::metal: return "metal";
    case ComputeDevice::cuda: return "cuda";
    case ComputeDevice::cpu: return "cpu";
    case ComputeDevice::rocm: return "rocm";
    }
    throw std::invalid_argument("Unknown compute device");
}

ComputeDevice selectComputeDevice(const ComputeDevice requested,
                                 const ComputeCapabilities &capabilities)
{
    return selectComputeDevice(requested, capabilities, {});
}

ComputeDevice selectComputeDevice(const ComputeDevice requested,
                                 const ComputeCapabilities &capabilities,
                                 const RocmCapabilities &rocm)
{
    const bool amd = rocm.libtorch && rocm.hip && rocm.available && rocm.deviceCount > 0;
    if (requested == ComputeDevice::automatic)
    {
        if (capabilities.mlx && capabilities.cuda)
        {
            return ComputeDevice::cuda;
        }
        if (amd)
        {
            return ComputeDevice::rocm;
        }
        if (capabilities.mlx && capabilities.metal)
        {
            return ComputeDevice::metal;
        }
        throw std::runtime_error(
            "No GPU is available; automatic CPU fallback is disabled. Select CPU explicitly.");
    }
    if ((requested == ComputeDevice::metal && capabilities.mlx && capabilities.metal) ||
        (requested == ComputeDevice::cuda && capabilities.mlx && capabilities.cuda) ||
        (requested == ComputeDevice::rocm && amd) ||
        (requested == ComputeDevice::cpu && ((capabilities.mlx && capabilities.cpu) ||
                                             (rocm.libtorch && rocm.cpu))))
    {
        return requested;
    }
    if (requested == ComputeDevice::rocm)
        throw std::runtime_error("ROCm requires IILD_ENABLE_LIBTORCH=ON, an AMD HIP-enabled "
                                 "LibTorch build, and a supported GPU/driver");
    throw std::runtime_error("Requested compute device is unavailable: " +
                             std::string{computeDeviceName(requested)});
}

} // namespace iild
