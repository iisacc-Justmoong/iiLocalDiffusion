#include "Compute/LinearLayer.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if IILD_TEST_LIBTORCH_ENABLED
#include <ATen/autocast_mode.h>
#endif

namespace
{
void check(const bool value, const std::string &message)
{
    if (!value) throw std::runtime_error(message);
}

template<typename Function> void rejects(Function &&function,
                                        const std::string &context = "native operation")
{
    bool failed = false;
    try { function(); }
    catch (const std::exception &) { failed = true; }
    check(failed, "An invalid/unsupported ROCm operation did not fail: " + context);
}

void compare(const std::vector<float> &actual, const std::vector<float> &expected)
{
    check(actual.size() == expected.size(), "Wrong output element count");
    for (std::size_t index = 0; index < actual.size(); ++index)
        check(std::isfinite(actual[index]) && std::abs(actual[index] - expected[index]) < 0.0004F,
              "Native output differs from the independent fixed oracle");
}

const std::vector<float> weights{1, 2, 3, -1, 0, 1};
const std::vector<float> bias{0.5F, -0.5F};
const std::vector<float> input{1, 2, 3, 4, 5, 6};
const std::vector<float> expected{14.5F, 1.5F, 32.5F, 1.5F};

void appendBytes(std::vector<std::uint8_t> &bytes, const std::uint64_t value, const std::size_t count)
{
    for (std::size_t index = 0; index < count; ++index)
        bytes.push_back(static_cast<std::uint8_t>((value >> (8 * index)) & 255U));
}

std::filesystem::path writeFixture(const std::string &name, std::string header,
                                  const std::vector<std::uint8_t> &payload)
{
    const std::filesystem::path directory{IILD_ROCM_TEST_DIRECTORY};
    std::filesystem::create_directories(directory);
    while (header.size() % 8 != 0) header += ' ';
    std::vector<std::uint8_t> bytes;
    appendBytes(bytes, header.size(), 8);
    bytes.insert(bytes.end(), header.begin(), header.end());
    bytes.insert(bytes.end(), payload.begin(), payload.end());
    const auto path = directory / name;
    std::ofstream stream{path, std::ios::binary | std::ios::trunc};
    stream.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    stream.close();
    check(static_cast<bool>(stream), "Could not write test fixture");
    return path;
}

std::filesystem::path weightFixture(const std::string &type, const bool singular)
{
    std::vector<std::uint8_t> payload;
    const std::vector<float> values{1, 2, 3, -1, 0, 1, 0.5F, -0.5F};
    if (type == "F32")
        for (const auto value : values) appendBytes(payload, std::bit_cast<std::uint32_t>(value), 4);
    else if (type == "F16")
        for (const auto value : {0x3c00U, 0x4000U, 0x4200U, 0xbc00U, 0U, 0x3c00U, 0x3800U, 0xb800U})
            appendBytes(payload, value, 2);
    else
        for (const auto value : values) appendBytes(payload, std::bit_cast<std::uint32_t>(value) >> 16, 2);
    const std::size_t width = type == "F32" ? 4U : 2U;
    return writeFixture(type + (singular ? ".safetensor" : ".safetensors"),
        "{\"weight\":{\"dtype\":\"" + type + "\",\"shape\":[2,3],\"data_offsets\":[0," +
        std::to_string(6 * width) + "]},\"bias\":{\"dtype\":\"" + type +
        "\",\"shape\":[2],\"data_offsets\":[" + std::to_string(6 * width) + "," +
        std::to_string(8 * width) + "]}}", payload);
}

void testExecution(const iild::ComputeDevice device)
{
    const iild::ComputeOptions options{device};
#if IILD_TEST_LIBTORCH_ENABLED
    // A host application's autocast scope must not override our explicit math dtype.
    {
        const auto kind = device == iild::ComputeDevice::cpu ? at::kCPU : at::kCUDA;
        const auto previous = at::autocast::is_autocast_enabled(kind);
        const auto previousDtype = at::autocast::get_autocast_dtype(kind);
        struct RestoreAutocast
        {
            at::DeviceType kind;
            bool enabled;
            at::ScalarType dtype;
            ~RestoreAutocast()
            {
                at::autocast::set_autocast_enabled(kind, enabled);
                at::autocast::set_autocast_dtype(kind, dtype);
            }
        } restore{kind, previous, previousDtype};
        at::autocast::set_autocast_dtype(kind, device == iild::ComputeDevice::cpu ? at::kBFloat16 : at::kHalf);
        at::autocast::set_autocast_enabled(kind, true);
        auto precise = iild::LinearLayer::fromWeights(std::vector<float>{0.1000001F}, 1, 1, {}, options);
        check(precise.forward(std::vector<float>{1}, 1) == std::vector<float>{0.1000001F},
              "Ambient LibTorch autocast overrode the requested FP32 precision");
        check(at::autocast::is_autocast_enabled(kind), "The caller's autocast state was changed");
    }
#endif
    auto layer = iild::LinearLayer::fromWeights(weights, 3, 2, bias, options);
    check(layer.computeInfo().runtime == "libtorch", "The intended native backend was not selected");
    check(layer.computeInfo().device == device, "Hardware selection changed silently");
    compare(layer.forward(input, 2), expected);
    std::vector<std::future<std::vector<float>>> calls;
    for (int index = 0; index < 4; ++index)
        calls.push_back(std::async(std::launch::async, [&] { return layer.forward(input, 2); }));
    for (auto &call : calls) compare(call.get(), expected);
    auto moved = std::move(layer);
    compare(moved.forward(input, 2), expected);
    rejects([&] { (void)layer.forward(input, 2); });
    rejects([&] { (void)moved.forward(input, 3); });
    rejects([&] { (void)moved.forward(std::vector<float>{NAN, 2, 3}, 1); });
    rejects([&] { (void)iild::LinearLayer::fromWeights(weights, 3, 2, bias,
        {device, std::numeric_limits<std::uint32_t>::max()}); });
    rejects([&] { (void)iild::LinearLayer::fromWeights(weights, 3, 2, bias, options,
        {0, static_cast<iild::WeightStorage>(99)}); });
    rejects([&] { (void)iild::LinearLayer::fromWeights(weights, 3, 2, bias, options,
        {std::numeric_limits<double>::quiet_NaN()}); });

    for (const auto &type : {std::string{"F32"}, std::string{"F16"}, std::string{"BF16"}})
    {
        const auto path = weightFixture(type, type != "F32");
        auto fileLayer = iild::LinearLayer::fromSafetensors(path, "weight", "bias", options);
        compare(fileLayer.forward(input, 2), expected);
        auto noBias = iild::LinearLayer::fromSafetensors(path, "weight", "", options);
        compare(noBias.forward(input, 2), {14, 2, 32, 2});
        rejects([&] { (void)iild::LinearLayer::fromSafetensors(path, "absent", "bias", options); });
        rejects([&] { (void)iild::LinearLayer::fromSafetensors(path, "weight", "absent", options); });
        rejects([&] { (void)iild::LinearLayer::fromSafetensors(path, "bias", "", options); });
        rejects([&] { (void)iild::LinearLayer::fromSafetensors(path, "weight", "weight", options); });
        const auto unicodePath = path.parent_path() /
            std::filesystem::path{u8"\ub77c\ub370\uc628 \uac00\uc911\uce58.safetensors"};
        std::filesystem::copy_file(path, unicodePath, std::filesystem::copy_options::overwrite_existing);
        auto unicodeLayer = iild::LinearLayer::fromSafetensors(unicodePath, "weight", "bias", options);
        std::filesystem::remove(unicodePath);
        compare(unicodeLayer.forward(input, 2), expected);
        std::filesystem::remove(path);
        compare(fileLayer.forward(input, 2), expected);
    }
    if (device == iild::ComputeDevice::cpu)
    {
        check(moved.resourceInfo().hostWeightBytes == 32, "Incorrect CPU storage accounting");
        rejects([&] { (void)iild::LinearLayer::fromWeights(weights, 3, 2, bias, options, {0.5}); });
        rejects([&] { (void)iild::LinearLayer::fromWeights(weights, 3, 2, bias, options, {},
            {iild::LinearPrecision::float16}); });
        return;
    }
    for (const auto precision : {iild::LinearPrecision::float32, iild::LinearPrecision::float16})
    {
        for (const auto storage : {iild::WeightStorage::device, iild::WeightStorage::ram})
        {
            const auto elementBytes = precision == iild::LinearPrecision::float32 ? 4U : 2U;
            auto hybrid = iild::LinearLayer::fromWeights(weights, 3, 2, bias, options,
                {0.5, storage, 4 * elementBytes}, {precision});
            compare(hybrid.forward(input, 2), expected);
            check(hybrid.resourceInfo().cpuOutputFeatures == 1 &&
                  hybrid.resourceInfo().acceleratorOutputFeatures == 1, "Wrong cooperative partition");
            check(hybrid.resourceInfo().hostWeightBytes ==
                  16 + (storage == iild::WeightStorage::ram ? 4 * elementBytes : 0),
                  "Wrong mixed-dtype RAM storage accounting");
            check(hybrid.resourceInfo().stagedAcceleratorWeightBytes ==
                  (storage == iild::WeightStorage::ram ? 4 * elementBytes : 0),
                  "GPU weight staging violated its budget");
        }
    }
    // Uneven chunks also exercise independent CPU columns and final short tiles.
    const std::vector<float> wideWeights{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto tiled = iild::LinearLayer::fromWeights(wideWeights, 1, 10, {}, options,
        {0.2, iild::WeightStorage::ram, 12});
    compare(tiled.forward(std::vector<float>{2}, 1), {2, 4, 6, 8, 10, 12, 14, 16, 18, 20});
    rejects([&] { (void)iild::LinearLayer::fromWeights(weights, 3, 2, bias, options,
        {0.5, iild::WeightStorage::ram, 1}); });
    rejects([&] { (void)iild::LinearLayer::fromWeights(std::vector<float>{70000}, 1, 1, {}, options,
        {}, {iild::LinearPrecision::float16}); });
    auto half = iild::LinearLayer::fromWeights(std::vector<float>{300}, 1, 1, {}, options, {},
        {iild::LinearPrecision::float16});
    rejects([&] { (void)half.forward(std::vector<float>{300}, 1); });
}

void testBadFiles(const iild::ComputeDevice device)
{
    std::size_t counter = 0;
    for (const auto *const header : {
        R"({"weight":{"dtype":"F32","shape":[1,2],"data_offsets":[0,4]}})",
        R"({"weight":{"dtype":"F32","shape":[1,1],"data_offsets":[0,8]}})",
        R"({"weight":{"dtype":"F32","shape":[1,1],"data_offsets":[8,4]}})",
        R"({"weight":{"dtype":"F32","shape":[18446744073709551615,2],"data_offsets":[0,4]}})",
        R"({"weight":{"dtype":"I32","shape":[1,1],"data_offsets":[0,4]}})",
        R"({"weight":{"dtype":"F32","shape":[0,1],"data_offsets":[0,0]}})",
        R"({"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})",
        R"({"weight":{"dtype":"F32","shape":[1,1],"data_offsets":[0,4]},"alias":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})",
        "not a JSON header"})
    {
        const auto path = writeFixture("invalid-" + std::to_string(counter++) + ".safetensors",
                                       header, {0, 0, 128, 63});
        rejects([&] { (void)iild::LinearLayer::fromSafetensors(path, "weight", "", {device}); }, path.string());
    }
    const auto nonfinite = writeFixture("nonfinite.safetensors",
        R"({"weight":{"dtype":"F32","shape":[1,1],"data_offsets":[0,4]}})", {0, 0, 128, 127});
    rejects([&] { (void)iild::LinearLayer::fromSafetensors(nonfinite, "weight", "", {device}); });
    for (const auto *const header : {
        R"({"weight":{"dtype":"F32","shape":[1,1],"data_offsets":[0,4]}})",
        R"({"weight":{"dtype":"F32","shape":[1,1],"data_offsets":[4,8]}})"})
    {
        const auto path = writeFixture("unused-" + std::to_string(counter++) + ".safetensors",
                                       header, {0, 0, 128, 63, 0, 0, 128, 63});
        rejects([&] { (void)iild::LinearLayer::fromSafetensors(path, "weight", "", {device}); }, path.string());
    }
}
} // namespace

int main(const int argc, const char *const argv[])
{
    try
    {
        const bool required = argc == 2 && std::string_view{argv[1]} == "--require-rocm";
        check(argc == 1 || required, "Only --require-rocm is accepted");
        const auto amd = iild::rocmCapabilities();
        check(amd.libtorch == static_cast<bool>(IILD_TEST_LIBTORCH_ENABLED), "Wrong LibTorch build capability");
        check(amd.available == (amd.hip && amd.deviceCount > 0), "Invalid ROCm availability report");
        if (required) check(amd.available, "Physical ROCm hardware is required but unavailable");
        if (!amd.available)
            rejects([] { (void)iild::LinearLayer::fromWeights(weights, 3, 2, bias, {iild::ComputeDevice::rocm}); });
        if (amd.libtorch && !iild::computeCapabilities().cpu)
        {
            testExecution(iild::ComputeDevice::cpu);
            testBadFiles(iild::ComputeDevice::cpu);
            std::cout << "LibTorch CPU and F32/F16/BF16 safetensors bridge verified\n";
        }
        if (amd.available)
        {
            testExecution(iild::ComputeDevice::rocm);
            testBadFiles(iild::ComputeDevice::rocm);
            std::cout << "ROCm GPU, CPU cooperation, and RAM staging verified\n";
        }
        else std::cout << "ROCm unavailable: absence policy verified; AMD physical execution NOT verified\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
