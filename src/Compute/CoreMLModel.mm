#include "CoreMLModel.hpp"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <span>
#include <stdexcept>
#include <utility>

namespace iild
{
namespace
{

std::string stringValue(NSString *value)
{
    const char *bytes = value.UTF8String;
    return bytes ? std::string{bytes} : std::string{};
}

NSString *nativeString(const std::string &value)
{
    if (value.find('\0') != std::string::npos)
        throw std::invalid_argument("Core ML paths and feature names cannot contain NUL bytes");
    NSString *result = [[NSString alloc] initWithBytes:value.data() length:value.size()
                                             encoding:NSUTF8StringEncoding];
    if (!result) throw std::invalid_argument("Core ML requires UTF-8 paths and feature names");
    return result;
}

std::runtime_error runtimeError(const std::string &operation, NSError *error)
{
    return std::runtime_error(operation + ": " + stringValue(error.localizedDescription));
}

MLComputeUnits nativeUnits(const CoreMLComputeUnits units) API_AVAILABLE(macos(13.0))
{
    switch (units)
    {
    case CoreMLComputeUnits::cpuAndNeuralEngine: return MLComputeUnitsCPUAndNeuralEngine;
    case CoreMLComputeUnits::all: return MLComputeUnitsAll;
    case CoreMLComputeUnits::cpuOnly: return MLComputeUnitsCPUOnly;
    }
    throw std::invalid_argument("Unknown Core ML compute units");
}

std::vector<std::size_t> readShape(NSArray<NSNumber *> *dimensions)
{
    if (dimensions.count == 0) throw std::invalid_argument("Core ML requires a declared non-empty shape");
    std::vector<std::size_t> shape;
    for (NSNumber *dimension in dimensions)
    {
        if (dimension.longLongValue <= 0)
            throw std::invalid_argument("Core ML dimensions must be positive");
        shape.push_back(static_cast<std::size_t>(dimension.unsignedLongLongValue));
    }
    return shape;
}

std::size_t elementCount(const std::vector<std::size_t> &shape)
{
    std::size_t count = 1;
    for (const auto dimension : shape)
    {
        if (dimension > static_cast<std::size_t>(std::numeric_limits<NSInteger>::max()) / count /
                            sizeof(float))
            throw std::invalid_argument("Core ML feature is too large");
        count *= dimension;
    }
    return count;
}

void validateShapeConstraint(MLMultiArrayConstraint *constraint)
{
    auto *flexibility = constraint.shapeConstraint;
    if (flexibility.type == MLMultiArrayShapeConstraintTypeEnumerated &&
        (flexibility.enumeratedShapes.count != 1 ||
         ![flexibility.enumeratedShapes.firstObject isEqualToArray:constraint.shape]))
        throw std::invalid_argument("Core ML component interfaces must bind a single shape");
    if (flexibility.type == MLMultiArrayShapeConstraintTypeRange)
    {
        if (flexibility.sizeRangeForDimension.count != constraint.shape.count)
            throw std::invalid_argument("Invalid Core ML shape range");
        for (NSUInteger axis = 0; axis < constraint.shape.count; ++axis)
        {
            const NSRange range = flexibility.sizeRangeForDimension[axis].rangeValue;
            if (range.length != 1 || range.location != constraint.shape[axis].unsignedIntegerValue)
                throw std::invalid_argument("Flexible Core ML shapes are not supported; bind them at conversion");
        }
    }
}

std::vector<CoreMLFeature> readFeatures(NSDictionary<NSString *, MLFeatureDescription *> *descriptions)
{
    std::vector<CoreMLFeature> result;
    for (NSString *name in [descriptions.allKeys sortedArrayUsingSelector:@selector(compare:)])
    {
        MLFeatureDescription *description = descriptions[name];
        if (description.type != MLFeatureTypeMultiArray || description.optional)
            throw std::invalid_argument("Core ML component interfaces require non-optional multi-array features");
        MLMultiArrayConstraint *constraint = description.multiArrayConstraint;
        CoreMLScalarType scalarType;
        if (constraint.dataType == MLMultiArrayDataTypeFloat32) scalarType = CoreMLScalarType::float32;
        else if (constraint.dataType == MLMultiArrayDataTypeFloat16) scalarType = CoreMLScalarType::float16;
        else throw std::invalid_argument("Core ML component interfaces support only float32 and float16");
        validateShapeConstraint(constraint);
        auto shape = readShape(constraint.shape);
        const auto count = elementCount(shape);
        result.push_back({stringValue(name), std::move(shape), count, scalarType});
    }
    if (result.empty()) throw std::invalid_argument("Core ML components need declared inputs and outputs");
    return result;
}

std::string deviceLabel(id<MLComputeDeviceProtocol> device) API_AVAILABLE(macos(14.0))
{
    if ([device isKindOfClass:MLNeuralEngineComputeDevice.class]) return "neural-engine";
    if ([device isKindOfClass:MLGPUComputeDevice.class]) return "gpu";
    if ([device isKindOfClass:MLCPUComputeDevice.class]) return "cpu";
    return "unknown";
}

void appendOperation(CoreMLModelInfo &info, NSString *name, NSString *operation,
                     MLComputePlanDeviceUsage *usage) API_AVAILABLE(macos(14.4))
{
    CoreMLPlanOperation entry;
    entry.name = stringValue(name);
    entry.operation = stringValue(operation);
    entry.preferredDevice = usage ? deviceLabel(usage.preferredComputeDevice) : "unknown";
    if (usage)
        for (id<MLComputeDeviceProtocol> device in usage.supportedComputeDevices)
            entry.supportedDevices.push_back(deviceLabel(device));
    if (entry.preferredDevice == "neural-engine") ++info.neuralEnginePreferredOperations;
    info.plan.push_back(std::move(entry));
}

void readBlock(CoreMLModelInfo &info, MLComputePlan *plan,
               MLModelStructureProgramBlock *block) API_AVAILABLE(macos(14.4))
{
    for (MLModelStructureProgramOperation *operation in block.operations)
    {
        appendOperation(info, operation.outputs.firstObject.name, operation.operatorName,
                        [plan computeDeviceUsageForMLProgramOperation:operation]);
        for (MLModelStructureProgramBlock *child in operation.blocks) readBlock(info, plan, child);
    }
}

void readPlan(CoreMLModelInfo &info, NSURL *url,
              MLModelConfiguration *configuration) API_AVAILABLE(macos(14.4))
{
    dispatch_semaphore_t complete = dispatch_semaphore_create(0);
    __block MLComputePlan *plan = nil;
    __block NSError *failure = nil;
    [MLComputePlan loadContentsOfURL:url configuration:configuration
                  completionHandler:^(MLComputePlan *loaded, NSError *error) {
        plan = loaded;
        failure = error;
        dispatch_semaphore_signal(complete);
    }];
    if (dispatch_semaphore_wait(complete, dispatch_time(DISPATCH_TIME_NOW, 60 * NSEC_PER_SEC)) != 0)
        throw std::runtime_error("Core ML compute plan timed out after 60 seconds");
    if (!plan) throw runtimeError("Cannot load Core ML compute plan", failure);
    if (plan.modelStructure.program)
    {
        auto *functions = plan.modelStructure.program.functions;
        if (functions.count != 1 || !functions[@"main"])
            throw std::invalid_argument("Core ML components require a single main function");
        readBlock(info, plan, functions[@"main"].block);
    }
    else if (plan.modelStructure.neuralNetwork)
    {
        for (MLModelStructureNeuralNetworkLayer *layer in plan.modelStructure.neuralNetwork.layers)
            appendOperation(info, layer.name, layer.type,
                            [plan computeDeviceUsageForNeuralNetworkLayer:layer]);
    }
    else throw std::invalid_argument("Core ML component plans support ML Program and NeuralNetwork models only");
    info.planAvailable = true;
}

// These helpers only marshal external runtime buffers. They implement no model math.
std::vector<std::size_t> checkedStrides(const std::vector<std::size_t> &shape,
                                      NSArray<NSNumber *> *strides, const NSInteger bytes)
{
    if (strides.count != shape.size() || bytes < static_cast<NSInteger>(sizeof(float)))
        throw std::runtime_error("Invalid Core ML buffer layout");
    const auto capacity = static_cast<std::size_t>(bytes) / sizeof(float);
    std::size_t maximumOffset = 0;
    std::vector<std::size_t> result;
    for (std::size_t axis = 0; axis < shape.size(); ++axis)
    {
        if (strides[axis].longLongValue < 0)
            throw std::runtime_error("Negative Core ML buffer stride");
        const auto stride = static_cast<std::size_t>(strides[axis].unsignedLongLongValue);
        const auto steps = shape[axis] - 1;
        if (steps > 0 && (stride == 0 || stride > (capacity - 1 - maximumOffset) / steps))
            throw std::runtime_error("Core ML buffer stride exceeds its storage");
        maximumOffset += steps * stride;
        result.push_back(stride);
    }
    return result;
}

std::size_t bufferOffset(std::size_t index, const std::vector<std::size_t> &shape,
                         const std::vector<std::size_t> &strides)
{
    std::size_t offset = 0;
    for (std::size_t axis = shape.size(); axis-- > 0;)
    {
        offset += (index % shape[axis]) * strides[axis];
        index /= shape[axis];
    }
    return offset;
}

bool contiguous(const std::vector<std::size_t> &shape, const std::vector<std::size_t> &strides)
{
    std::size_t stride = 1;
    for (std::size_t axis = shape.size(); axis-- > 0;)
    {
        if (shape[axis] > 1 && strides[axis] != stride) return false;
        stride *= shape[axis];
    }
    return true;
}

MLMultiArray *makeInput(const CoreMLFeature &feature, const std::vector<float> &data)
{
    if (data.size() != feature.elementCount)
        throw std::invalid_argument("Wrong Core ML input size: " + feature.name);
    for (const float value : data)
        if (!std::isfinite(value) ||
            (feature.scalarType == CoreMLScalarType::float16 && std::abs(value) > 65504.0F))
            throw std::invalid_argument("Non-finite or out-of-range Core ML input: " + feature.name);
    NSMutableArray<NSNumber *> *shape = [NSMutableArray arrayWithCapacity:feature.shape.size()];
    for (const auto dimension : feature.shape) [shape addObject:@(dimension)];
    NSError *error = nil;
    MLMultiArray *array = [[MLMultiArray alloc] initWithShape:shape dataType:MLMultiArrayDataTypeFloat32
                                                     error:&error];
    if (!array) throw runtimeError("Cannot allocate Core ML input", error);
    const std::span<const float> values{data};
    [array getMutableBytesWithHandler:^(void *bytes, NSInteger size, NSArray<NSNumber *> *rawStrides) {
        const auto strides = checkedStrides(feature.shape, rawStrides, size);
        auto *destination = static_cast<float *>(bytes);
        if (!destination) throw std::runtime_error("Missing Core ML input storage");
        if (contiguous(feature.shape, strides)) std::memcpy(destination, values.data(), values.size_bytes());
        else
            for (std::size_t index = 0; index < values.size(); ++index)
                destination[bufferOffset(index, feature.shape, strides)] = values[index];
    }];
    if (feature.scalarType == CoreMLScalarType::float16)
        array = [MLMultiArray multiArrayByConcatenatingMultiArrays:@[array] alongAxis:0
                                                         dataType:MLMultiArrayDataTypeFloat16];
    return array;
}

std::vector<float> readOutput(MLMultiArray *array, const CoreMLFeature &feature)
{
    const auto requiredType = feature.scalarType == CoreMLScalarType::float16
        ? MLMultiArrayDataTypeFloat16 : MLMultiArrayDataTypeFloat32;
    if (!array || array.dataType != requiredType || readShape(array.shape) != feature.shape ||
        array.count < 0 || static_cast<std::size_t>(array.count) != feature.elementCount)
        throw std::runtime_error("Core ML output violates its interface: " + feature.name);
    if (array.dataType != MLMultiArrayDataTypeFloat32)
        array = [MLMultiArray multiArrayByConcatenatingMultiArrays:@[array] alongAxis:0
                                                         dataType:MLMultiArrayDataTypeFloat32];
    std::vector<float> result(feature.elementCount);
    float *destination = result.data();
    [array getBytesWithHandler:^(const void *bytes, NSInteger size) {
        const auto strides = checkedStrides(feature.shape, array.strides, size);
        const auto *source = static_cast<const float *>(bytes);
        if (!source) throw std::runtime_error("Missing Core ML output storage");
        if (contiguous(feature.shape, strides))
            std::memcpy(destination, source, feature.elementCount * sizeof(float));
        else
            for (std::size_t index = 0; index < feature.elementCount; ++index)
                destination[index] = source[bufferOffset(index, feature.shape, strides)];
    }];
    if (!std::all_of(result.begin(), result.end(), [](const float value) { return std::isfinite(value); }))
        throw std::runtime_error("Core ML prediction produced non-finite values: " + feature.name);
    return result;
}

} // namespace

struct CoreMLModel::Impl
{
    MLModel *model = nil;
    CoreMLModelInfo info;
    std::mutex mutex;
};

CoreMLCapabilities coreMLCapabilities()
{
    CoreMLCapabilities result;
    @autoreleasepool
    {
        if (@available(macOS 13.0, *)) result.runtime = true;
        if (@available(macOS 14.0, *))
        {
            result.deviceDiscovery = true;
            for (id<MLComputeDeviceProtocol> device in MLModel.availableComputeDevices)
                if ([device isKindOfClass:MLNeuralEngineComputeDevice.class])
                {
                    const NSInteger count = ((MLNeuralEngineComputeDevice *)device).totalCoreCount;
                    if (count > 0) result.neuralEngineCores += static_cast<std::size_t>(count);
                }
            result.neuralEngine = result.neuralEngineCores > 0;
        }
        if (@available(macOS 14.4, *)) result.computePlan = true;
    }
    return result;
}

CoreMLModel CoreMLModel::load(const std::filesystem::path &compiledModel, const CoreMLOptions options)
{
    (void)coreMLComputeUnitsName(options.computeUnits);
    if (options.requireNeuralEnginePlan && options.computeUnits == CoreMLComputeUnits::cpuOnly)
        throw std::invalid_argument("CPU-only execution cannot require a Neural Engine plan");
    if (compiledModel.extension() != ".mlmodelc" || !std::filesystem::is_directory(compiledModel))
        throw std::invalid_argument("Core ML requires an existing compiled .mlmodelc directory; convert weights first");
    @autoreleasepool
    {
        @try
        {
            const auto capabilities = coreMLCapabilities();
            if (!capabilities.runtime) throw std::runtime_error("Core ML components require macOS 13 or newer");
            if (options.requireNeuralEnginePlan && (!capabilities.neuralEngine || !capabilities.computePlan))
                throw std::runtime_error("Neural Engine plans require a detected Neural Engine and macOS 14.4 or newer");
            if (@available(macOS 13.0, *))
            {
                NSURL *url = [NSURL fileURLWithPath:nativeString(std::filesystem::canonical(compiledModel).string())
                                       isDirectory:YES];
                MLModelConfiguration *configuration = [MLModelConfiguration new];
                configuration.computeUnits = nativeUnits(options.computeUnits);
                NSError *error = nil;
                MLModel *model = [MLModel modelWithContentsOfURL:url configuration:configuration error:&error];
                if (!model) throw runtimeError("Cannot load Core ML component", error);
                auto implementation = std::make_unique<Impl>();
                implementation->model = model;
                auto &info = implementation->info;
                info.computeUnits = options.computeUnits;
                info.inputs = readFeatures(model.modelDescription.inputDescriptionsByName);
                info.outputs = readFeatures(model.modelDescription.outputDescriptionsByName);
                if (@available(macOS 14.4, *)) readPlan(info, url, configuration);
                if (options.requireNeuralEnginePlan && info.neuralEnginePreferredOperations == 0)
                    throw std::runtime_error("Core ML does not plan any operation on Neural Engine; CPU-only plans are rejected");
                return CoreMLModel{std::move(implementation)};
            }
        }
        @catch (NSException *exception)
        {
            throw std::runtime_error("Core ML exception: " + stringValue(exception.reason));
        }
    }
    throw std::runtime_error("Core ML is unavailable");
}

CoreMLModel::CoreMLModel(std::unique_ptr<Impl> implementation)
    : implementation_{std::move(implementation)} {}
CoreMLModel::~CoreMLModel() = default;
CoreMLModel::CoreMLModel(CoreMLModel &&) noexcept = default;
CoreMLModel &CoreMLModel::operator=(CoreMLModel &&) noexcept = default;

CoreMLModel::Impl &CoreMLModel::implementation() const
{
    if (!implementation_) throw std::logic_error("Cannot use a moved-from Core ML model");
    return *implementation_;
}

const CoreMLModelInfo &CoreMLModel::info() const { return implementation().info; }

CoreMLModel::Features CoreMLModel::predict(const Features &inputs) const
{
    auto &state = implementation();
    const std::lock_guard lock{state.mutex};
    if (inputs.size() != state.info.inputs.size())
        throw std::invalid_argument("Core ML inputs must exactly match the declared feature names");
    @autoreleasepool
    {
        @try
        {
            NSMutableDictionary<NSString *, MLFeatureValue *> *dictionary = [NSMutableDictionary dictionary];
            for (const auto &feature : state.info.inputs)
            {
                const auto found = inputs.find(feature.name);
                if (found == inputs.end()) throw std::invalid_argument("Missing Core ML input: " + feature.name);
                dictionary[nativeString(feature.name)] = [MLFeatureValue featureValueWithMultiArray:makeInput(feature, found->second)];
            }
            NSError *error = nil;
            MLDictionaryFeatureProvider *provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:dictionary error:&error];
            if (!provider) throw runtimeError("Cannot construct Core ML input provider", error);
            id<MLFeatureProvider> prediction = [state.model predictionFromFeatures:provider error:&error];
            if (!prediction) throw runtimeError("Core ML prediction failed", error);
            Features result;
            for (const auto &feature : state.info.outputs)
            {
                MLFeatureValue *value = [prediction featureValueForName:nativeString(feature.name)];
                if (!value || value.type != MLFeatureTypeMultiArray)
                    throw std::runtime_error("Missing Core ML output: " + feature.name);
                result.emplace(feature.name, readOutput(value.multiArrayValue, feature));
            }
            return result;
        }
        @catch (NSException *exception)
        {
            throw std::runtime_error("Core ML prediction exception: " + stringValue(exception.reason));
        }
    }
}

} // namespace iild
