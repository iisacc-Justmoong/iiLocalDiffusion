#include "ModelManifest/DiffusionModelManifest.hpp"
#include "ComputeCommands.hpp"
#include "NeuralCommands.hpp"

#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string_view>
#include <type_traits>
#include <variant>

namespace
{

constexpr int usageError = 2;
constexpr int inputUnavailableError = 3;
constexpr int invalidPackageError = 4;
constexpr int unsupportedModelError = 5;
constexpr int internalError = 70;

std::string displayPath(const std::filesystem::path &path);

void printUsage(std::ostream &output)
{
    output << "Usage: iild-run inspect <model-root>\n"
           << "       iild-run devices\n"
           << "       iild-run compute [--device auto|metal|cuda|rocm|cpu] [--device-index N]\n"
           << "                        [--cpu-share FRACTION] [--weight-storage device|ram]\n"
           << "                        [--gpu-weight-mib N] [--precision fp32|fp16|bf16]\n"
           << "       iild-run neural-compute --model PATH.mlmodelc [--compute-units cpu-ne|all|cpu]\n"
           << "                               [--allow-cpu-plan] [--iterations N]\n";
}

void printManifest(const iild::StableDiffusionModelManifest &manifest)
{
    std::cout << "iiLocalDiffusion Model Inspector\n\n"
              << "Model root: " << displayPath(manifest.root()) << '\n'
              << "Format: Diffusers\n"
              << "Pipeline: " << manifest.pipelineClass() << '\n'
              << "Compatibility: " << manifest.compatibility() << "\n\n"
              << "Tokenizer metadata: valid\n"
              << "Text encoder metadata: valid\n";
    if (manifest.compatibilityKind() == iild::StableDiffusionCompatibility::xlBase)
    {
        std::cout << "Tokenizer 2 metadata: valid\n"
                  << "Text encoder 2 metadata: valid\n";
    }
    std::cout << "UNet metadata: valid\n"
              << "VAE metadata: valid\n"
              << "Scheduler metadata: valid\n\n"
              << "Weight files: present; contents not inspected\n"
              << "Result: valid-metadata\n";
}

void printManifest(const iild::FluxModelManifest &manifest)
{
    std::cout << "iiLocalDiffusion Model Inspector\n\n"
              << "Model root: " << displayPath(manifest.root()) << '\n'
              << "Format: Diffusers\n"
              << "Pipeline: " << manifest.pipelineClass() << '\n'
              << "Compatibility: " << manifest.compatibility() << "\n\n"
              << "Tokenizer metadata: valid\n"
              << "Text encoder metadata: valid\n"
              << "Tokenizer 2 metadata: valid\n"
              << "Text encoder 2 metadata: valid\n"
              << "Transformer metadata: valid\n"
              << "VAE metadata: valid\n"
              << "Scheduler metadata: valid\n\n"
              << "Weight files: present; contents not inspected\n"
              << "Result: valid-metadata\n";
}

std::string displayPath(const std::filesystem::path &path)
{
    std::ostringstream output;
    for (const char rawCharacter : path.generic_string())
    {
        const auto character = static_cast<unsigned char>(rawCharacter);
        if (character >= 0x20 && character <= 0x7e && character != '\\')
        {
            output << rawCharacter;
        }
        else if (character == '\\')
        {
            output << "\\\\";
        }
        else
        {
            output << "\\x" << std::hex << std::setw(2) << std::setfill('0')
                   << static_cast<unsigned int>(character) << std::dec;
        }
    }
    return output.str();
}

} // namespace

int main(const int argc, const char *const argv[])
{
    if (argc >= 2 && std::string_view{argv[1]} == "neural-compute")
        return runNeuralComputeCommand(argc, argv);
    if (argc == 2 && (std::string_view{argv[1]} == "--help" || std::string_view{argv[1]} == "-h"))
    {
        printUsage(std::cout);
        return 0;
    }

    if (argc >= 2 && (std::string_view{argv[1]} == "devices" ||
                      std::string_view{argv[1]} == "compute"))
    {
        return runComputeCommand(argc, argv);
    }

    if (argc != 3 || std::string_view{argv[1]} != "inspect")
    {
        printUsage(std::cerr);
        return usageError;
    }

    try
    {
        const auto manifest = iild::loadModelManifest(argv[2]);
        std::visit([](const auto &value) { printManifest(value); }, manifest);
        return 0;
    }
    catch (const iild::ModelManifestError &error)
    {
        switch (error.code())
        {
        case iild::ModelManifestErrorCode::inputUnavailable:
            std::cerr << "error[input-unavailable]: " << error.what() << '\n';
            return inputUnavailableError;
        case iild::ModelManifestErrorCode::invalidPackage:
            std::cerr << "error[invalid-package]: " << error.what() << '\n';
            return invalidPackageError;
        case iild::ModelManifestErrorCode::unsupportedModel:
            std::cerr << "error[unsupported-model]: " << error.what() << '\n';
            return unsupportedModelError;
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << "Inspection failed: " << error.what() << '\n';
        return internalError;
    }

    return internalError;
}
