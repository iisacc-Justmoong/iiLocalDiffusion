#include "CoreMLModel.hpp"

#include <stdexcept>
#include <utility>

namespace iild
{

std::string_view coreMLComputeUnitsName(const CoreMLComputeUnits units)
{
    switch (units)
    {
    case CoreMLComputeUnits::cpuAndNeuralEngine: return "cpu+neural-engine";
    case CoreMLComputeUnits::all: return "cpu+gpu+neural-engine";
    case CoreMLComputeUnits::cpuOnly: return "cpu";
    }
    throw std::invalid_argument("Unknown Core ML compute units");
}

#if !IILD_HAS_COREML

struct CoreMLModel::Impl {};

CoreMLCapabilities coreMLCapabilities() { return {}; }

CoreMLModel CoreMLModel::load(const std::filesystem::path &, const CoreMLOptions options)
{
    (void)coreMLComputeUnitsName(options.computeUnits);
    throw std::runtime_error("Core ML is not built; it requires macOS and IILD_ENABLE_COREML=ON");
}

CoreMLModel::CoreMLModel(std::unique_ptr<Impl> implementation)
    : implementation_{std::move(implementation)} {}
CoreMLModel::~CoreMLModel() = default;
CoreMLModel::CoreMLModel(CoreMLModel &&) noexcept = default;
CoreMLModel &CoreMLModel::operator=(CoreMLModel &&) noexcept = default;

CoreMLModel::Impl &CoreMLModel::implementation() const
{
    throw std::runtime_error("Core ML is not built");
}

CoreMLModel::Features CoreMLModel::predict(const Features &) const
{
    throw std::runtime_error("Core ML is not built");
}

const CoreMLModelInfo &CoreMLModel::info() const
{
    throw std::runtime_error("Core ML is not built");
}

#endif

} // namespace iild
