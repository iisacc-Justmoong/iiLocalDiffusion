#include "Compute/ComputeRuntime.hpp"
#include "Compute/LinearLayer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if IILD_TEST_MLX_ENABLED
#include <mlx/mlx.h>
#endif

namespace
{

void check(const bool value, const std::string &message)
{
    if (!value)
    {
        throw std::runtime_error(message);
    }
}

template<typename Function>
void expectFailure(Function &&function)
{
    bool failed = false;
    try
    {
        function();
    }
    catch (const std::exception &)
    {
        failed = true;
    }
    check(failed, "Expected an unavailable device or invalid input to fail");
}

float compare(const std::vector<float> &actual, const std::vector<float> &expected)
{
    check(actual.size() == expected.size(), "Output shape mismatch");
    float maximumError = 0.0F;
    for (std::size_t index = 0; index < actual.size(); ++index)
    {
        check(std::isfinite(actual[index]), "Non-finite output");
        const float difference = std::abs(actual[index] - expected[index]);
        maximumError = std::max(maximumError, difference);
        check(difference <= 0.0004F + 0.0001F * std::abs(expected[index]),
              "Neural output differs from the CPU/oracle result");
    }
    return maximumError;
}

void testSelection()
{
    using Device = iild::ComputeDevice;
    iild::RocmCapabilities amd;
    amd.libtorch = true;
    amd.cpu = true;
    amd.hip = true;
    amd.available = true;
    amd.deviceCount = 1;
    check(iild::selectComputeDevice(Device::automatic, {}, amd) == Device::rocm,
          "Automatic selection did not use AMD ROCm without MLX");
    check(iild::selectComputeDevice(Device::rocm, {}, amd) == Device::rocm,
          "Explicit ROCm was rejected");
    check(iild::selectComputeDevice(Device::cpu, {}, amd) == Device::cpu,
          "Explicit LibTorch CPU was rejected");
    check(iild::computeDeviceName(Device::rocm) == "rocm", "Incorrect AMD device name");
    check(iild::selectComputeDevice(Device::automatic, {true, false, true, true}, amd) ==
          Device::cuda, "Existing NVIDIA selection priority changed");
    for (const int missing : {0, 1, 2})
    {
        auto invalid = amd;
        if (missing == 0) invalid.hip = false;
        if (missing == 1) invalid.libtorch = false;
        if (missing == 2) invalid.deviceCount = 0;
        expectFailure([&] { (void)iild::selectComputeDevice(Device::rocm, {}, invalid); });
    }
    expectFailure([] { (void)iild::selectComputeDevice(Device::rocm,
                                                      {true, true, true, true}, {}); });
    amd.available = false;
    amd.deviceCount = 0;
    expectFailure([&] { (void)iild::selectComputeDevice(Device::automatic, {}, amd); });
    check(iild::selectComputeDevice(Device::automatic, {true, true, false, true}, amd) ==
          Device::metal, "Missing AMD hardware interfered with Metal selection");
    check(iild::selectComputeDevice(Device::automatic, {true, true, false, true}) ==
              Device::metal, "Auto did not select Metal");
    check(iild::selectComputeDevice(Device::automatic, {true, false, true, true}) ==
              Device::cuda, "Auto did not select CUDA");
    check(iild::selectComputeDevice(Device::cpu, {true, false, false, true}) ==
              Device::cpu, "Explicit CPU was rejected");
    expectFailure([] { (void)iild::selectComputeDevice(Device::automatic,
                                                      {true, false, false, true}); });
    expectFailure([] { (void)iild::selectComputeDevice(Device::metal,
                                                      {true, false, true, true}); });
    expectFailure([] { (void)iild::selectComputeDevice(Device::cuda,
                                                      {true, true, false, true}); });
    expectFailure([] { (void)iild::selectComputeDevice(Device::cpu, {}); });
    expectFailure([] { (void)iild::computeDeviceName(static_cast<Device>(99)); });
    using Support = iild::AcceleratorSupport;
    check(iild::classifyTensorCoreSupport("NVIDIA RTX 4090", 8, 9) == Support::supported,
          "Ampere-or-newer Tensor capability was missed");
    check(iild::classifyTensorCoreSupport("GeForce GTX 1650", 7, 5) == Support::unsupported,
          "CUDA support was incorrectly equated with Tensor Cores");
    check(iild::classifyTensorCoreSupport("Unidentified Turing GPU", 7, 5) == Support::unknown,
          "Unidentified hardware must not be declared supported");
    check(iild::classifyTensorCoreSupport("Tesla T4", 7, 5) == Support::supported,
          "T4 Tensor capability was missed");
}

void testMatrixPrecision()
{
    const auto capabilities = iild::computeCapabilities();
    if (!capabilities.mlx) return;
    expectFailure([] { (void)iild::LinearLayer::fromWeights(std::vector<float>{1}, 1, 1, {},
        {iild::ComputeDevice::cpu}, {}, {iild::LinearPrecision::float16}); });
    if (!capabilities.metal && !capabilities.cuda) return;
    for (const auto precision : {iild::LinearPrecision::float16, iild::LinearPrecision::bfloat16})
    {
        if (precision == iild::LinearPrecision::bfloat16 && capabilities.cuda &&
            !iild::tensorCoreCapabilities().bf16Eligible)
        {
            expectFailure([] { (void)iild::LinearLayer::fromWeights(std::vector<float>{1}, 1, 1,
                {}, {iild::ComputeDevice::cuda}, {}, {iild::LinearPrecision::bfloat16}); });
            continue;
        }
        const float rounded = precision == iild::LinearPrecision::float16
            ? 0.0999755859375F : 0.10009765625F;
        for (const auto storage : {iild::WeightStorage::device, iild::WeightStorage::ram})
        {
            auto layer = iild::LinearLayer::fromWeights(std::vector<float>{0.1000001F, 0.1000001F},
                1, 2, {}, {}, {0.5, storage, 2}, {precision});
            const auto output = layer.forward(std::vector<float>{1}, 1);
            check(output == std::vector<float>({0.1000001F, rounded}),
                  "Requested GPU precision or float32 CPU shard was not used");
            check(layer.precision() == precision, "Wrong reported matrix precision");
            const auto &allocation = layer.resourceInfo();
            check(allocation.hostWeightBytes == (storage == iild::WeightStorage::ram ? 6U : 4U),
                  "Mixed-precision RAM accounting is incorrect");
            check(allocation.stagedAcceleratorWeightBytes == (storage == iild::WeightStorage::ram ? 2U : 0U),
                  "Mixed-precision staging budget is incorrect");
        }
    }
    expectFailure([] { (void)iild::LinearLayer::fromWeights(std::vector<float>{1}, 1, 1, {},
        {}, {}, {static_cast<iild::LinearPrecision>(99)}); });
    auto half = iild::LinearLayer::fromWeights(std::vector<float>{300}, 1, 1, {}, {}, {},
        {iild::LinearPrecision::float16});
    expectFailure([&] { (void)half.forward(std::vector<float>{300}, 1); });
    expectFailure([&] { (void)half.forward(std::vector<float>{70000}, 1); });
    expectFailure([] { (void)iild::LinearLayer::fromWeights(std::vector<float>{70000}, 1, 1,
        {}, {}, {}, {iild::LinearPrecision::float16}); });
}

void testLinear()
{
    const std::vector<float> weights{1, 2, 3, -1, 0, 1};
    const std::vector<float> bias{0.5F, -0.5F};
    const std::vector<float> input{1, 2, 3, 4, 5, 6};
    const std::vector<float> expected{14.5F, 1.5F, 32.5F, 1.5F};
    const auto capabilities = iild::computeCapabilities();
    if (!capabilities.mlx)
    {
        if (!iild::rocmCapabilities().available)
            expectFailure([&] { (void)iild::LinearLayer::fromWeights(weights, 3, 2); });
        return;
    }
    auto cpu = iild::LinearLayer::fromWeights(weights, 3, 2, bias, {iild::ComputeDevice::cpu});
    check(cpu.inputFeatures() == 3 && cpu.outputFeatures() == 2, "Wrong layer dimensions");
    check(cpu.computeInfo().runtime == "mlx", "The native runtime is not MLX");
    check(cpu.computeInfo().runtimeVersion == "0.32.2", "Unpinned MLX runtime");
    check(cpu.resourceInfo().cpuOutputFeatures == 2 &&
          cpu.resourceInfo().acceleratorOutputFeatures == 0 &&
          cpu.resourceInfo().hostWeightBytes == 32 &&
          cpu.resourceInfo().stagedAcceleratorWeightBytes == 0,
          "CPU-only resource accounting is incorrect");
    (void)compare(cpu.forward(input, 2), expected);
    expectFailure([&] { (void)cpu.forward(input, 0); });
    expectFailure([&] { (void)cpu.forward(input, 3); });
    expectFailure([&] { (void)cpu.forward(input, std::numeric_limits<std::size_t>::max()); });
    expectFailure([&] { (void)iild::LinearLayer::fromWeights(weights, 2, 2); });
    expectFailure([&] { (void)iild::LinearLayer::fromWeights(weights, 0, 2); });
    expectFailure([&] { (void)iild::LinearLayer::fromWeights(weights, 3, 2, input); });
    const float invalid = std::numeric_limits<float>::quiet_NaN();
    expectFailure([&] { (void)cpu.forward(std::vector<float>{1, 2, invalid}, 1); });
    expectFailure([&] { (void)iild::LinearLayer::fromWeights(
        std::vector<float>{invalid}, 1, 1, {}, {iild::ComputeDevice::cpu}); });
    expectFailure([&] { (void)iild::LinearLayer::fromWeights(
        std::vector<float>{1}, 1, 1, std::vector<float>{invalid}, {iild::ComputeDevice::cpu}); });
    auto noBias = iild::LinearLayer::fromWeights(weights, 3, 2, {}, {iild::ComputeDevice::cpu});
    (void)compare(noBias.forward(input, 2), {14, 2, 32, 2});
    expectFailure([&] {
        (void)iild::LinearLayer::fromWeights(weights, 3, 2, {},
                                           {iild::ComputeDevice::cpu, 99});
    });
    auto moved = std::move(cpu);
    (void)compare(moved.forward(input, 2), expected);
    expectFailure([&] { (void)cpu.forward(input, 2); });
    if (capabilities.metal || capabilities.cuda || iild::rocmCapabilities().available)
    {
        auto gpu = iild::LinearLayer::fromWeights(weights, 3, 2, bias);
        check(gpu.computeInfo().device != iild::ComputeDevice::cpu,
              "Default neural operations fell back to CPU");
        (void)compare(gpu.forward(input, 2), expected);
        std::cout << "Native GPU linear verified: "
                  << iild::computeDeviceName(gpu.computeInfo().device) << '\n';
    }
    else
    {
        expectFailure([&] { (void)iild::LinearLayer::fromWeights(weights, 3, 2); });
    }
}

void testBatchedAndConcurrentLinear()
{
    const auto capabilities = iild::computeCapabilities();
    if (!capabilities.mlx)
    {
        return;
    }
    constexpr std::size_t inputFeatures = 67;
    constexpr std::size_t outputFeatures = 41;
    constexpr std::size_t batch = 32;
    std::vector<float> weights(inputFeatures * outputFeatures);
    std::vector<float> input(batch * inputFeatures);
    std::vector<float> bias(outputFeatures, 0.125F);
    for (std::size_t index = 0; index < weights.size(); ++index)
    {
        weights[index] = static_cast<float>(static_cast<int>(index % 11) - 5) / 8.0F;
    }
    for (std::size_t index = 0; index < input.size(); ++index)
    {
        input[index] = static_cast<float>(static_cast<int>(index % 7) - 3) / 4.0F;
    }
    auto cpu = iild::LinearLayer::fromWeights(weights, inputFeatures, outputFeatures, bias,
                                             {iild::ComputeDevice::cpu});
    const auto reference = cpu.forward(input, batch);
    iild::ComputeOptions options;
    if (!capabilities.metal && !capabilities.cuda)
    {
        options.device = iild::ComputeDevice::cpu;
    }
    auto layer = iild::LinearLayer::fromWeights(weights, inputFeatures, outputFeatures, bias,
                                               options);
    std::vector<std::future<std::vector<float>>> calls;
    for (int index = 0; index < 4; ++index)
    {
        calls.push_back(std::async(std::launch::async, [&layer, &input] {
            return layer.forward(input, batch);
        }));
    }
    for (auto &call : calls)
    {
        (void)compare(call.get(), reference);
    }
    if (capabilities.metal || capabilities.cuda)
    {
        // Thirteen-output blocks leave a final partial block (41 - 10 = 31).
        auto hybrid = iild::LinearLayer::fromWeights(weights, inputFeatures, outputFeatures,
            bias, options, {0.25, iild::WeightStorage::ram, 13 * (inputFeatures + 1) * sizeof(float)});
        check(hybrid.resourceInfo().cpuOutputFeatures == 10 &&
              hybrid.resourceInfo().gpuWeightChunkOutputs == 13, "Wrong hybrid partition");
        calls.clear();
        for (int index = 0; index < 4; ++index)
        {
            calls.push_back(std::async(std::launch::async, [&hybrid, &input] {
                return hybrid.forward(input, batch);
            }));
        }
        for (auto &call : calls)
        {
            (void)compare(call.get(), reference);
        }
        auto noBias = iild::LinearLayer::fromWeights(weights, inputFeatures, outputFeatures,
            {}, options, {0.25, iild::WeightStorage::ram, 13 * inputFeatures * sizeof(float)});
        auto noBiasExpected = reference;
        for (auto &value : noBiasExpected) value -= 0.125F;
        (void)compare(noBias.forward(input, batch), noBiasExpected);
    }
}

void testCpuAndRamParticipation()
{
    const auto capabilities = iild::computeCapabilities();
    if (!capabilities.mlx)
    {
        if (!iild::rocmCapabilities().available)
            expectFailure([] { (void)iild::LinearLayer::fromWeights(
                std::vector<float>{1, 2}, 1, 2, {}, {}, {0.5, iild::WeightStorage::ram, 4}); });
        return;
    }
    const std::vector<float> weights{1, 0, 0, 0, 2, 0, 0, 0, 3, 1, 1, 1};
    const std::vector<float> bias{0.5F, 1, 1.5F, 2};
    const std::vector<float> input{1, 2, 3, 4, 5, 6};
    const std::vector<float> expected{1.5F, 5, 10.5F, 8, 4.5F, 11, 19.5F, 17};
    iild::LinearResourceOptions resources;
    resources.cpuShare = 0.5;
    resources.weightStorage = iild::WeightStorage::ram;
    resources.gpuWeightBudgetBytes = 16;
    expectFailure([&] { (void)iild::LinearLayer::fromWeights(
        weights, 3, 4, bias, {iild::ComputeDevice::cpu}, resources); });
    if (!capabilities.metal && !capabilities.cuda && !iild::rocmCapabilities().available)
    {
        expectFailure([&] { (void)iild::LinearLayer::fromWeights(
            weights, 3, 4, bias, {}, resources); });
        return;
    }
    auto hybrid = iild::LinearLayer::fromWeights(weights, 3, 4, bias, {}, resources);
    const auto &allocation = hybrid.resourceInfo();
    check(allocation.cpuOutputFeatures == 2 && allocation.acceleratorOutputFeatures == 2,
          "CPU/GPU work was not partitioned");
    check(allocation.hostWeightBytes == 64 && allocation.residentAcceleratorWeightBytes == 0,
          "RAM weight ownership is incorrect");
    check(allocation.gpuWeightChunkOutputs == 1 && allocation.stagedAcceleratorWeightBytes == 16,
          "GPU staging exceeded its weight budget");
    (void)compare(hybrid.forward(input, 2), expected);
    (void)compare(hybrid.forward(input, 2), expected);

    resources.weightStorage = iild::WeightStorage::device;
    auto resident = iild::LinearLayer::fromWeights(weights, 3, 4, bias, {}, resources);
    check(resident.resourceInfo().hostWeightBytes == 32 &&
          resident.resourceInfo().residentAcceleratorWeightBytes == 32,
          "Resident CPU/GPU weights were not partitioned");
    (void)compare(resident.forward(input, 2), expected);
    resources.cpuShare = 0;
    resources.weightStorage = iild::WeightStorage::ram;
    auto ram = iild::LinearLayer::fromWeights(weights, 3, 4, bias, {}, resources);
    (void)compare(ram.forward(input, 2), expected);
    check(ram.resourceInfo().cpuOutputFeatures == 0 && ram.resourceInfo().hostWeightBytes == 64,
          "RAM-only staging incorrectly claimed CPU arithmetic");
    resources.gpuWeightBudgetBytes = 15;
    expectFailure([&] { (void)iild::LinearLayer::fromWeights(weights, 3, 4, bias, {}, resources); });
    resources.gpuWeightBudgetBytes = 16;
    for (const double invalid : {-0.1, 1.0, std::numeric_limits<double>::quiet_NaN()})
    {
        resources.cpuShare = invalid;
        expectFailure([&] { (void)iild::LinearLayer::fromWeights(weights, 3, 4, bias, {}, resources); });
    }
    resources.cpuShare = 0.5;
    expectFailure([&] { (void)iild::LinearLayer::fromWeights(
        std::vector<float>{1}, 1, 1, {}, {}, resources); });
    resources.weightStorage = static_cast<iild::WeightStorage>(99);
    expectFailure([&] { (void)iild::LinearLayer::fromWeights(weights, 3, 4, bias, {}, resources); });
    std::cout << "CPU/GPU partition and RAM staging verified\n";
}

#if IILD_TEST_MLX_ENABLED
namespace mx = mlx::core;

std::vector<float> values(mx::array array)
{
    array = mx::contiguous(mx::astype(array, mx::float32, mx::Device::cpu), false,
                          mx::Device::cpu);
    mx::eval(array);
    return {array.data<float>(), array.data<float>() + array.size()};
}

void testSafetensors(const std::filesystem::path &directory)
{
    std::filesystem::create_directories(directory);
    const auto stem = "linear-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const auto file = directory / (stem + ".safetensor");
    const auto canonical = directory / (stem + ".safetensors");
    struct FixtureCleanup
    {
        std::filesystem::path alias;
        std::filesystem::path source;
        ~FixtureCleanup()
        {
            std::error_code ignored;
            std::filesystem::remove(alias, ignored);
            std::filesystem::remove(source, ignored);
        }
    } cleanup{file, canonical};
    const auto weights = mx::array({1.0F, 2.0F, 3.0F, -1.0F, 0.0F, 1.0F}, {2, 3});
    mx::save_safetensors(canonical.string(), {
        {"weight", weights},
        {"weight_fp16", mx::astype(weights, mx::float16, mx::Device::cpu)},
        {"weight_bf16", mx::astype(weights, mx::bfloat16, mx::Device::cpu)},
        {"weight_int", mx::astype(weights, mx::int32, mx::Device::cpu)},
        {"weight_nan", mx::array({std::numeric_limits<float>::quiet_NaN()}, {1, 1})},
        {"bias", mx::array({0.5F, -0.5F})},
    });
    std::filesystem::create_symlink(canonical, file);
    auto layer = iild::LinearLayer::fromSafetensors(file, "weight", "bias",
                                                  {iild::ComputeDevice::cpu});
    (void)compare(layer.forward(std::vector<float>{1, 2, 3}, 1), {14.5F, 1.5F});
    // File I/O must be on the host, but the selected layer must execute on the GPU.
    const auto capabilities = iild::computeCapabilities();
    std::optional<iild::LinearLayer> loadedHybrid;
    if (capabilities.metal || capabilities.cuda)
    {
        loadedHybrid.emplace(iild::LinearLayer::fromSafetensors(
            file, "weight_fp16", "bias", {}, {0.5, iild::WeightStorage::ram, 16}));
        (void)compare(loadedHybrid->forward(std::vector<float>{1, 2, 3}, 1), {14.5F, 1.5F});
        for (const std::string key : {"weight", "weight_fp16", "weight_bf16"})
        {
            auto gpu = iild::LinearLayer::fromSafetensors(file, key, "bias");
            check(gpu.computeInfo().device != iild::ComputeDevice::cpu,
                  "Safetensors layer silently used CPU");
            (void)compare(gpu.forward(std::vector<float>{1, 2, 3}, 1), {14.5F, 1.5F});
        }
    }
    expectFailure([&] { (void)iild::LinearLayer::fromSafetensors(
        file, "missing", {}, {iild::ComputeDevice::cpu}); });
    expectFailure([&] { (void)iild::LinearLayer::fromSafetensors(
        file, "bias", {}, {iild::ComputeDevice::cpu}); });
    expectFailure([&] { (void)iild::LinearLayer::fromSafetensors(
        file, "weight", "weight", {iild::ComputeDevice::cpu}); });
    expectFailure([&] { (void)iild::LinearLayer::fromSafetensors(
        file, "weight", "missing", {iild::ComputeDevice::cpu}); });
    expectFailure([&] { (void)iild::LinearLayer::fromSafetensors(
        file, "weight_int", {}, {iild::ComputeDevice::cpu}); });
    expectFailure([&] { (void)iild::LinearLayer::fromSafetensors(
        file, "weight_nan", {}, {iild::ComputeDevice::cpu}); });
    expectFailure([&] { (void)iild::LinearLayer::fromSafetensors(directory / "absent.pt",
                                                              "weight"); });
    std::filesystem::remove(file);
    std::filesystem::remove(canonical);
    // Construction must materialize weights, not retain a lazy dependency on the file.
    (void)compare(layer.forward(std::vector<float>{1, 2, 3}, 1), {14.5F, 1.5F});
    if (loadedHybrid)
    {
        (void)compare(loadedHybrid->forward(std::vector<float>{1, 2, 3}, 1), {14.5F, 1.5F});
    }
}

void oracleParity(const std::filesystem::path &directory)
{
    // This existing validator is specifically an MLX oracle. A new ROCm
    // auto-selection must never cause it to label LibTorch output as MLX.
    const auto capabilities = iild::computeCapabilities();
    const iild::ComputeOptions options{capabilities.cuda ? iild::ComputeDevice::cuda
                                                        : iild::ComputeDevice::metal};
    auto [inputs, inputMetadata] = mx::load_safetensors((directory / "input.safetensors").string(),
                                                       mx::Device::cpu);
    auto [expected, expectedMetadata] = mx::load_safetensors(
        (directory / "expected.safetensors").string(), mx::Device::cpu);
    const auto batch = static_cast<std::size_t>(inputs.at("input").shape(0));
    const auto input = values(inputs.at("input"));
    const auto reference = values(expected.at("output"));
    auto gpu = iild::LinearLayer::fromSafetensors(directory / "weights.safetensors",
                                                "weight", "bias", options);
    const float error = compare(gpu.forward(input, batch), reference);
    std::ofstream report{directory / "mlx-parity.json"};
    report << "{\"runtime\":\"mlx\",\"version\":\"" << gpu.computeInfo().runtimeVersion
           << "\",\"device\":\"" << iild::computeDeviceName(gpu.computeInfo().device)
           << "\",\"batch\":" << batch << ",\"input_features\":" << gpu.inputFeatures()
           << ",\"output_features\":" << gpu.outputFeatures()
           << ",\"maximum_absolute_error\":" << error << ",\"hybrid\":";
    if (gpu.outputFeatures() >= 2)
    {
        const std::size_t budget = std::max(std::size_t{1024 * 1024},
                                           (gpu.inputFeatures() + 1) * sizeof(float));
        auto hybrid = iild::LinearLayer::fromSafetensors(directory / "weights.safetensors",
            "weight", "bias", options, {0.25, iild::WeightStorage::ram, budget});
        const float hybridError = compare(hybrid.forward(input, batch), reference);
        auto resident = iild::LinearLayer::fromSafetensors(directory / "weights.safetensors",
            "weight", "bias", options, {0.25, iild::WeightStorage::device, budget});
        const float residentError = compare(resident.forward(input, batch), reference);
        const auto &allocation = hybrid.resourceInfo();
        report << "{\"cpu_output_features\":" << allocation.cpuOutputFeatures
               << ",\"gpu_output_features\":" << allocation.acceleratorOutputFeatures
               << ",\"ram_weight_bytes\":" << allocation.hostWeightBytes
               << ",\"gpu_weight_budget_bytes\":" << budget
               << ",\"staged_gpu_weight_bytes\":" << allocation.stagedAcceleratorWeightBytes
               << ",\"maximum_absolute_error\":" << hybridError
               << ",\"resident_maximum_absolute_error\":" << residentError << '}';
        std::cout << "PyTorch / MLX CPU+GPU+RAM parity passed, max absolute error: " << hybridError << '\n';
    }
    else
    {
        report << "null"; // A single output feature cannot be partitioned.
    }
    report << ",\"passed\":true}\n";
    check(static_cast<bool>(report), "Could not write parity report");
    std::cout << "PyTorch / MLX GPU parity passed, max absolute error: " << error << '\n';
}
#endif

} // namespace

int main(const int argc, const char *const argv[])
{
    try
    {
        testSelection();
        testMatrixPrecision();
        testLinear();
        testBatchedAndConcurrentLinear();
        testCpuAndRamParticipation();
#if IILD_TEST_MLX_ENABLED
        testSafetensors(IILD_COMPUTE_TEST_DIRECTORY);
        if (argc == 2)
        {
            oracleParity(argv[1]);
        }
#else
        (void)argc;
        (void)argv;
#endif
        std::cout << "Compute contracts passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
