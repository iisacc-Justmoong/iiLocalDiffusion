#include "ComputeCommands.hpp"

#include "Compute/ComputeRuntime.hpp"
#include "Compute/CoreMLModel.hpp"
#include "Compute/LinearLayer.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{

iild::ComputeDevice parseDevice(const std::string_view value)
{
    if (value == "auto") return iild::ComputeDevice::automatic;
    if (value == "metal") return iild::ComputeDevice::metal;
    if (value == "cuda") return iild::ComputeDevice::cuda;
    if (value == "rocm") return iild::ComputeDevice::rocm;
    if (value == "cpu") return iild::ComputeDevice::cpu;
    throw std::invalid_argument("--device must be auto, metal, cuda, rocm, or cpu");
}

iild::LinearPrecision parsePrecision(const std::string_view value)
{
    if (value == "fp32") return iild::LinearPrecision::float32;
    if (value == "fp16") return iild::LinearPrecision::float16;
    if (value == "bf16") return iild::LinearPrecision::bfloat16;
    throw std::invalid_argument("--precision must be fp32, fp16, or bf16");
}

void printCapabilities()
{
    const auto capabilities = iild::computeCapabilities();
    const auto coreML = iild::coreMLCapabilities();
    const auto tensorCores = iild::tensorCoreCapabilities();
    const auto amd = iild::rocmCapabilities();
    const auto status = [](const bool available) { return available ? "available" : "unavailable"; };
    std::cout << "MLX: " << status(capabilities.mlx) << '\n'
              << "Metal: " << status(capabilities.metal) << '\n'
              << "CUDA: " << status(capabilities.cuda) << '\n'
              << "CPU (MLX): " << status(capabilities.cpu) << '\n'
              << "LibTorch: " << status(amd.libtorch) << '\n'
              << "CPU (LibTorch): " << status(amd.cpu) << '\n'
              << "ROCm (AMD): " << status(amd.available) << '\n'
              << "LibTorch HIP build: " << (amd.hip ? "yes" : "no") << '\n'
              << "ROCm device count: " << amd.deviceCount << '\n'
              << "Core ML: " << status(coreML.runtime) << '\n'
              << "Apple Neural Engine: " << (coreML.deviceDiscovery ? status(coreML.neuralEngine)
                    : (coreML.runtime ? "unknown" : "unavailable"));
    if (coreML.deviceDiscovery) std::cout << " (" << coreML.neuralEngineCores << " cores)";
    std::cout << "\nNeural Engine plan inspection: " << status(coreML.computePlan) << '\n'
              << "Tensor Cores (CUDA): " << iild::acceleratorSupportName(tensorCores.support) << '\n'
              << "FP16 Tensor eligibility: " << (tensorCores.fp16Eligible ? "yes" : "no") << '\n'
              << "BF16 Tensor eligibility: " << (tensorCores.bf16Eligible ? "yes" : "no") << '\n'
              << "TF32 Tensor eligibility: " << (tensorCores.tf32Eligible ? "yes" : "no") << '\n'
              << "Tensor Core detection: architecture/device family; kernel usage not profiled\n"
              << "Default policy: GPU required; CPU is explicit\n";
}

void checkCompute(const iild::ComputeOptions options, const iild::LinearResourceOptions resources,
                  const iild::LinearMathOptions math)
{
    constexpr std::size_t inputs = 256;
    constexpr std::size_t outputs = 128;
    constexpr std::size_t batch = 64;
    const std::vector<float> weights(inputs * outputs, 1.0F);
    const std::vector<float> bias(outputs, 0.5F);
    const std::vector<float> data(batch * inputs, 1.0F);
    auto layer = iild::LinearLayer::fromWeights(weights, inputs, outputs, bias, options, resources, math);
    const auto result = layer.forward(data, batch);
    const auto &allocation = layer.resourceInfo();
    if (result.size() != batch * outputs)
        throw std::runtime_error("Hardware linear check returned an incorrect result");
    for (std::size_t index = 0; index < result.size(); ++index)
    {
        const bool cpuShard = index % outputs < allocation.cpuOutputFeatures;
        const float expected = !cpuShard && math.precision == iild::LinearPrecision::bfloat16
            ? 256.0F : 256.5F;
        if (result[index] != expected)
            throw std::runtime_error("Hardware linear check returned an incorrect result for its precision");
    }
    const auto &info = layer.computeInfo();
    std::cout << "Runtime: " << info.runtime << ' ' << info.runtimeVersion << '\n'
              << "Device: " << iild::computeDeviceName(info.device) << ':' << info.deviceIndex << '\n'
              << "Hardware: " << info.deviceName << '\n'
              << "GPU precision: " << (allocation.acceleratorOutputFeatures
                    ? iild::linearPrecisionName(layer.precision()) : "not used") << '\n'
              << "CPU output features: " << allocation.cpuOutputFeatures << '\n'
              << "GPU output features: " << allocation.acceleratorOutputFeatures << '\n'
              << "RAM weight bytes: " << allocation.hostWeightBytes << '\n'
              << "Resident GPU weight bytes: " << allocation.residentAcceleratorWeightBytes << '\n'
              << "Staged GPU weight bytes: " << allocation.stagedAcceleratorWeightBytes << '\n'
              << "Operation: linear [64,256] x [128,256]^T + bias -> [64,128]\n"
              << "Result: verified\n";
    if (info.device == iild::ComputeDevice::cuda)
    {
        const auto tensor = iild::tensorCoreCapabilities(info.deviceIndex);
        std::cout << "Tensor Core support: " << iild::acceleratorSupportName(tensor.support)
                  << "; CUDA compute capability " << tensor.capabilityMajor << '.' << tensor.capabilityMinor
                  << "\nTensor Core dispatch: runtime-auto; instruction usage not profiled\n";
    }
    if (info.device == iild::ComputeDevice::rocm)
        std::cout << "ROCm dispatch: LibTorch/HIP; no NVIDIA TF32 policy or CPU fallback\n";
}

} // namespace

int runComputeCommand(const int argc, const char *const argv[])
{
    try
    {
        if (std::string_view{argv[1]} == "devices")
        {
            if (argc != 2)
            {
                throw std::invalid_argument("devices does not accept arguments");
            }
            printCapabilities();
            return 0;
        }
        iild::ComputeOptions options;
        iild::LinearResourceOptions resources;
        iild::LinearMathOptions math;
        bool deviceSeen = false;
        bool indexSeen = false;
        bool shareSeen = false;
        bool storageSeen = false;
        bool budgetSeen = false;
        bool precisionSeen = false;
        for (int index = 2; index < argc; index += 2)
        {
            const std::string_view argument{argv[index]};
            if (index + 1 >= argc)
            {
                throw std::invalid_argument("Compute option is missing a value");
            }
            const std::string_view value{argv[index + 1]};
            if (argument == "--device" && !deviceSeen)
            {
                options.device = parseDevice(value);
                deviceSeen = true;
            }
            else if (argument == "--device-index" && !indexSeen)
            {
                const auto parsed = std::from_chars(value.data(), value.data() + value.size(),
                                                    options.deviceIndex);
                if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
                {
                    throw std::invalid_argument("--device-index must be a non-negative integer");
                }
                indexSeen = true;
            }
            else if (argument == "--cpu-share" && !shareSeen)
            {
                const auto parsed = std::from_chars(value.data(), value.data() + value.size(),
                                                    resources.cpuShare);
                if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
                    !std::isfinite(resources.cpuShare) || resources.cpuShare < 0 || resources.cpuShare >= 1)
                {
                    throw std::invalid_argument("--cpu-share must be a finite fraction in [0, 1)");
                }
                shareSeen = true;
            }
            else if (argument == "--precision" && !precisionSeen)
            {
                math.precision = parsePrecision(value);
                precisionSeen = true;
            }
            else if (argument == "--weight-storage" && !storageSeen)
            {
                if (value == "device") resources.weightStorage = iild::WeightStorage::device;
                else if (value == "ram") resources.weightStorage = iild::WeightStorage::ram;
                else throw std::invalid_argument("--weight-storage must be device or ram");
                storageSeen = true;
            }
            else if (argument == "--gpu-weight-mib" && !budgetSeen)
            {
                std::size_t mib = 0;
                constexpr std::size_t bytesPerMib = 1024 * 1024;
                const auto parsed = std::from_chars(value.data(), value.data() + value.size(), mib);
                if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || mib == 0 ||
                    mib > std::numeric_limits<std::size_t>::max() / bytesPerMib)
                {
                    throw std::invalid_argument("--gpu-weight-mib must be a positive, representable MiB count");
                }
                resources.gpuWeightBudgetBytes = mib * bytesPerMib;
                budgetSeen = true;
            }
            else
            {
                throw std::invalid_argument("Unknown or repeated compute option: " + std::string{argument});
            }
        }
        if (budgetSeen && (resources.weightStorage != iild::WeightStorage::ram ||
                          options.device == iild::ComputeDevice::cpu))
        {
            throw std::invalid_argument("--gpu-weight-mib requires RAM weight storage and a GPU");
        }
        checkCompute(options, resources, math);
        return 0;
    }
    catch (const std::invalid_argument &error)
    {
        std::cerr << "error[compute-argument]: " << error.what() << '\n';
        return 2;
    }
    catch (const std::exception &error)
    {
        std::cerr << "error[compute-unavailable]: " << error.what() << '\n';
        return 6;
    }
}
