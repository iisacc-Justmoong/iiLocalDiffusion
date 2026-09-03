#include "NeuralCommands.hpp"

#include "Compute/CoreMLModel.hpp"

#include <charconv>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

int runNeuralComputeCommand(const int argc, const char *const argv[])
{
    try
    {
        std::filesystem::path modelPath;
        iild::CoreMLOptions options;
        std::size_t iterations = 1;
        bool modelSeen = false;
        bool unitsSeen = false;
        bool iterationsSeen = false;
        bool allowCpuSeen = false;
        for (int index = 2; index < argc; ++index)
        {
            const std::string_view argument{argv[index]};
            if (argument == "--allow-cpu-plan" && !allowCpuSeen)
            {
                options.requireNeuralEnginePlan = false;
                allowCpuSeen = true;
                continue;
            }
            if (index + 1 >= argc) throw std::invalid_argument("Core ML option is missing a value");
            const std::string_view value{argv[++index]};
            if (argument == "--model" && !modelSeen)
            {
                modelPath = std::string{value};
                modelSeen = true;
            }
            else if (argument == "--compute-units" && !unitsSeen)
            {
                if (value == "cpu-ne") options.computeUnits = iild::CoreMLComputeUnits::cpuAndNeuralEngine;
                else if (value == "all") options.computeUnits = iild::CoreMLComputeUnits::all;
                else if (value == "cpu") options.computeUnits = iild::CoreMLComputeUnits::cpuOnly;
                else throw std::invalid_argument("--compute-units must be cpu-ne, all, or cpu");
                unitsSeen = true;
            }
            else if (argument == "--iterations" && !iterationsSeen)
            {
                const auto parsed = std::from_chars(value.data(), value.data() + value.size(), iterations);
                if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
                    iterations == 0 || iterations > 10000)
                    throw std::invalid_argument("--iterations must be an integer in [1, 10000]");
                iterationsSeen = true;
            }
            else throw std::invalid_argument("Unknown or repeated Core ML option: " + std::string{argument});
        }
        if (!modelSeen) throw std::invalid_argument("neural-compute requires --model PATH.mlmodelc");
        auto model = iild::CoreMLModel::load(modelPath, options);
        const auto &info = model.info();
        iild::CoreMLModel::Features inputs;
        for (const auto &feature : info.inputs)
            inputs.emplace(feature.name, std::vector<float>(feature.elementCount, 1.0F));
        const auto started = std::chrono::steady_clock::now();
        std::size_t outputScalars = 0;
        for (std::size_t index = 0; index < iterations; ++index)
        {
            const auto output = model.predict(inputs);
            outputScalars = 0;
            for (const auto &[name, values] : output)
            {
                (void)name;
                outputScalars += values.size();
            }
        }
        const auto elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        std::cout << "Runtime: Core ML\n"
                  << "Permitted compute units: " << iild::coreMLComputeUnitsName(info.computeUnits) << '\n'
                  << "Neural Engine cores: " << iild::coreMLCapabilities().neuralEngineCores << '\n'
                  << "Compute plan: " << (info.planAvailable ? "available" : "unavailable") << '\n'
                  << "Neural Engine preferred operations: " << info.neuralEnginePreferredOperations << '\n'
                  << "Hardware counters verified: false (plan is anticipated dispatch)\n"
                  << "Input: synthetic all-ones diagnostic data; caller-provided model\n"
                  << "Prediction: completed; finite output scalars: " << outputScalars << '\n'
                  << "Iterations: " << iterations << "; prediction/readback milliseconds: " << elapsed << '\n';
        return 0;
    }
    catch (const std::invalid_argument &error)
    {
        std::cerr << "error[coreml-argument]: " << error.what() << '\n';
        return 2;
    }
    catch (const std::exception &error)
    {
        std::cerr << "error[coreml-unavailable]: " << error.what() << '\n';
        return 6;
    }
}
