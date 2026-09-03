#include "Compute/CoreMLModel.hpp"

#include <json-c/json.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

void check(const bool condition, const std::string &message)
{
    if (!condition) throw std::runtime_error(message);
}

template<typename Function>
void expectFailure(Function &&function)
{
    bool failed = false;
    try { function(); }
    catch (const std::exception &) { failed = true; }
    check(failed, "Expected an invalid or unavailable Core ML request to fail");
}

json_object *member(json_object *object, const char *name)
{
    json_object *value = nullptr;
    check(object && json_object_object_get_ex(object, name, &value),
          "Missing oracle field: " + std::string{name});
    return value;
}

iild::CoreMLModel::Features features(json_object *object)
{
    check(json_object_is_type(object, json_type_object), "Expected feature dictionary");
    iild::CoreMLModel::Features result;
    json_object_object_foreach(object, key, value)
    {
        check(json_object_is_type(value, json_type_array), "Expected flat feature array");
        auto &data = result[key];
        for (std::size_t index = 0; index < json_object_array_length(value); ++index)
        {
            auto *number = json_object_array_get_idx(value, index);
            check(json_object_is_type(number, json_type_double) ||
                  json_object_is_type(number, json_type_int), "Non-numeric oracle value");
            const auto scalar = static_cast<float>(json_object_get_double(number));
            check(std::isfinite(scalar), "Non-finite oracle value");
            data.push_back(scalar);
        }
    }
    return result;
}

float compare(const iild::CoreMLModel::Features &actual,
              const iild::CoreMLModel::Features &expected, const float atol, const float rtol,
              const std::string &context = "cpu+neural-engine")
{
    check(actual.size() == expected.size(), "Output feature count differs from oracle");
    float error = 0;
    for (const auto &[name, data] : expected)
    {
        const auto &output = actual.at(name);
        check(output.size() == data.size(), "Output shape differs from oracle");
        for (std::size_t index = 0; index < output.size(); ++index)
        {
            const float difference = std::abs(output[index] - data[index]);
            check(std::isfinite(output[index]) && difference <= atol + rtol * std::abs(data[index]),
                  "Core ML output differs from the independent float32 oracle: " + context +
                  " " + name + "[" + std::to_string(index) + "] actual=" +
                  std::to_string(output[index]) + " expected=" + std::to_string(data[index]));
            error = std::max(error, difference);
        }
    }
    return error;
}

void validateFixture(const std::filesystem::path &directory)
{
    using Json = std::unique_ptr<json_object, decltype(&json_object_put)>;
    Json oracle{json_object_from_file((directory / "oracle.json").string().c_str()), json_object_put};
    check(oracle != nullptr, "Cannot read Core ML oracle.json");
    const auto inputs = features(member(oracle.get(), "inputs"));
    const auto expected = features(member(oracle.get(), "outputs"));
    const auto atol = static_cast<float>(json_object_get_double(member(oracle.get(), "atol")));
    const auto rtol = static_cast<float>(json_object_get_double(member(oracle.get(), "rtol")));
    check(std::isfinite(atol) && atol >= 0 && std::isfinite(rtol) && rtol >= 0,
          "Invalid oracle tolerance");
    check(!inputs.empty(), "Oracle must have at least one input");
    const auto compiled = directory / "model.mlmodelc";
    const bool requireNeural = json_object_get_boolean(
        member(oracle.get(), "require_neural_engine_plan"));
    auto model = iild::CoreMLModel::load(compiled,
        {iild::CoreMLComputeUnits::cpuAndNeuralEngine, requireNeural});
    check(!model.info().hardwareUsageVerified, "A plan was confused with measured usage");
    check(model.info().computeUnits == iild::CoreMLComputeUnits::cpuAndNeuralEngine,
          "Core ML unexpectedly permits the GPU");
    if (requireNeural)
        check(model.info().planAvailable && model.info().neuralEnginePreferredOperations > 0,
              "No operation was planned for Neural Engine");
    if (model.info().planAvailable && model.info().neuralEnginePreferredOperations == 0)
        expectFailure([&] { (void)iild::CoreMLModel::load(compiled); });
    const float maximumError = compare(model.predict(inputs), expected, atol, rtol);
    expectFailure([&] { (void)model.predict({}); });
    auto incorrect = inputs;
    auto first = incorrect.extract(incorrect.begin());
    first.key() = "incorrect-feature-name";
    incorrect.insert(std::move(first));
    expectFailure([&] { (void)model.predict(incorrect); });
    incorrect = inputs;
    incorrect["unexpected-feature"] = {1};
    expectFailure([&] { (void)model.predict(incorrect); });
    incorrect = inputs;
    incorrect.begin()->second.push_back(1);
    expectFailure([&] { (void)model.predict(incorrect); });
    incorrect = inputs;
    incorrect.begin()->second.front() = std::numeric_limits<float>::infinity();
    expectFailure([&] { (void)model.predict(incorrect); });
    incorrect.begin()->second.front() = std::numeric_limits<float>::quiet_NaN();
    expectFailure([&] { (void)model.predict(incorrect); });
    for (const auto &feature : model.info().inputs)
        if (feature.scalarType == iild::CoreMLScalarType::float16)
        {
            incorrect = inputs;
            incorrect.at(feature.name).front() = 70000.0F;
            expectFailure([&] { (void)model.predict(incorrect); });
        }
    std::vector<std::future<iild::CoreMLModel::Features>> calls;
    for (int index = 0; index < 3; ++index)
        calls.push_back(std::async(std::launch::async, [&] { return model.predict(inputs); }));
    for (auto &call : calls) (void)compare(call.get(), expected, atol, rtol);
    auto cpu = iild::CoreMLModel::load(compiled, {iild::CoreMLComputeUnits::cpuOnly, false});
    check(cpu.info().neuralEnginePreferredOperations == 0, "CPU-only plan includes Neural Engine");
    // This converted graph computes in float16. CPU accumulation can be less
    // accurate than ANE's; use a separate exact zero-input/bias oracle rather
    // than weakening the independent float32 tolerance for Neural Engine.
    const auto cpuInputs = features(member(oracle.get(), "cpu_inputs"));
    const auto cpuExpected = features(member(oracle.get(), "cpu_outputs"));
    (void)compare(cpu.predict(cpuInputs), cpuExpected, 0, 0, "cpu-only exact fixture");
    auto moved = std::move(model);
    (void)compare(moved.predict(inputs), expected, atol, rtol);
    expectFailure([&] { (void)model.predict(inputs); });
    expectFailure([&] { (void)model.info(); });
    std::cout << "Core ML prediction verified; planned Neural Engine operations: "
              << moved.info().neuralEnginePreferredOperations << "; max abs error: "
              << maximumError << "; hardware counters verified: false\n";
    Json report{json_object_new_object(), json_object_put};
    json_object_object_add(report.get(), "passed", json_object_new_boolean(true));
    json_object_object_add(report.get(), "compute_units", json_object_new_string("cpu+neural-engine"));
    json_object_object_add(report.get(), "max_abs_error", json_object_new_double(maximumError));
    json_object_object_add(report.get(), "neural_engine_preferred_operations",
        json_object_new_uint64(moved.info().neuralEnginePreferredOperations));
    json_object_object_add(report.get(), "hardware_usage_verified", json_object_new_boolean(false));
    json_object_object_add(report.get(), "cpu_zero_prediction_verified", json_object_new_boolean(true));
    std::cout << "COREML_RESULT " << json_object_to_json_string_ext(report.get(), JSON_C_TO_STRING_PLAIN) << '\n';
}

} // namespace

int main(const int argc, const char *const argv[])
{
    try
    {
        const auto capabilities = iild::coreMLCapabilities();
        if (!IILD_TEST_COREML_ENABLED)
            check(!capabilities.runtime && !capabilities.neuralEngine,
                  "Disabled Core ML claims runtime hardware support");
        check(capabilities.neuralEngine == (capabilities.neuralEngineCores > 0),
              "Neural Engine discovery is inconsistent");
        check(iild::coreMLComputeUnitsName(iild::CoreMLComputeUnits::cpuAndNeuralEngine) == "cpu+neural-engine",
              "Wrong Core ML units label");
        expectFailure([] { (void)iild::coreMLComputeUnitsName(static_cast<iild::CoreMLComputeUnits>(99)); });
        expectFailure([] { (void)iild::CoreMLModel::load("not-a-model.safetensors"); });
        expectFailure([] { (void)iild::CoreMLModel::load("missing.mlmodelc"); });
        expectFailure([] { (void)iild::CoreMLModel::load("missing.mlmodelc",
            {iild::CoreMLComputeUnits::cpuOnly, true}); });
        expectFailure([] { (void)iild::CoreMLModel::load("missing.mlmodelc",
            {static_cast<iild::CoreMLComputeUnits>(99), false}); });
        if (argc == 3 && std::string{argv[1]} == "--fixture") validateFixture(argv[2]);
        else check(argc == 1, "Usage: CoreMLTests [--fixture DIRECTORY]");
        std::cout << "Core ML contracts passed; Neural Engine cores: "
                  << capabilities.neuralEngineCores << '\n';
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
