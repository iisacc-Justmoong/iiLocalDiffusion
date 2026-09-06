#include "Generation/GenerationIO.hpp"

#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using namespace iild;

void require(const bool condition, const std::string &message)
{
    if (!condition) throw std::runtime_error(message);
}

void rejects(const std::function<void()> &operation, const std::string &fragment)
{
    try { operation(); }
    catch (const GenerationIOError &error)
    {
        require(std::string(error.what()).find(fragment) != std::string::npos,
                "expected diagnostic containing " + fragment + ", got " + error.what());
        return;
    }
    throw std::runtime_error("expected GenerationIOError: " + fragment);
}

GenerationTensorSpec latent(const GenerationSemantic semantic = GenerationSemantic::sample)
{
    return {TensorDType::float32, {1, 2}, "BC", semantic, "vae:fixture-v1"};
}

GenerationTensorSpec tokens()
{
    return {TensorDType::int64, {1, -1}, "BS", GenerationSemantic::tokenIds, "tokenizer:fixture-v1"};
}

GenerationTensor value(GenerationTensorSpec spec, std::vector<float> values)
{
    return {std::move(spec), std::move(values)};
}

GenerationStage model(const std::string &id, const GenerationArchitecture architecture,
                      const GenerationTensorSpec &input, const GenerationTensorSpec &output,
                      const GenerationSource &source)
{
    return {id, GenerationStageRole::model, architecture,
            {{"input", input}}, {{"output", output}}, {{"input", source}}};
}

GenerationPlan single(const GenerationArchitecture architecture = GenerationArchitecture::diffusion)
{
    return {{{"seed", latent()}},
            {model("generate", architecture, latent(), latent(), {"", "seed"})},
            {{"result", {"generate", "output"}, latent()}}};
}

GenerationValues identity(const GenerationStage &, const GenerationValues &inputs)
{
    return {{"output", inputs.at("input")}};
}

void supportsEveryArchitecture()
{
    for (const auto architecture : {GenerationArchitecture::diffusion,
                                   GenerationArchitecture::rectifiedFlow,
                                   GenerationArchitecture::flowMatching,
                                   GenerationArchitecture::autoregressive})
    {
        const auto result = executeGenerationPlan(single(architecture),
            {{"seed", value(latent(), {1.0F, 2.0F})}}, identity);
        require(std::get<std::vector<float>>(result.at("result").data) ==
                    std::vector<float>({1.0F, 2.0F}), "payload did not reach plan output");
    }
}

void rejectsInvalidPayloads()
{
    rejects([] { validateGenerationTensor(latent(), value(latent(), {1.0F})); }, "element count");
    rejects([] { validateGenerationTensor(latent(), value(latent(), {1.0F, INFINITY})); }, "finite");
    rejects([] { validateGenerationTensor(latent(), value(latent(), {NAN, 1.0F})); }, "finite");
    rejects([] { validateGenerationTensor(latent(), {latent(), std::vector<double>{1, 2}}); }, "storage");
    auto wrongLayout = latent(); wrongLayout.layout = "CB";
    rejects([&] { validateGenerationTensor(latent(), value(wrongLayout, {1, 2})); }, "layout");
    auto wrongSpace = latent(); wrongSpace.representationSpace = "vae:other";
    rejects([&] { validateGenerationTensor(latent(), value(wrongSpace, {1, 2})); }, "representationSpace");
    auto wrongSemantic = latent(GenerationSemantic::epsilon);
    rejects([&] { validateGenerationTensor(latent(), value(wrongSemantic, {1, 2})); }, "semantic");
    auto wrongDType = latent(); wrongDType.dtype = TensorDType::float64;
    rejects([&] { validateGenerationTensor(latent(), {wrongDType, std::vector<double>{1, 2}}); }, "dtype");
    auto enormous = latent(); enormous.shape = {std::numeric_limits<std::int64_t>::max(), 4};
    rejects([&] { validateGenerationTensor(enormous, value(enormous, {})); }, "overflow");
    auto byteOverflow = latent(); byteOverflow.shape = {std::numeric_limits<std::int64_t>::max()};
    rejects([&] { validateGenerationTensor(byteOverflow, value(byteOverflow, {})); }, "overflow");
}

void supportsConcreteDynamicAndEmptyTokenInputs()
{
    auto concrete = tokens(); concrete.shape = {1, 3};
    validateGenerationTensor(tokens(), {concrete, std::vector<std::int64_t>{0, 1, 2}});
    auto empty = tokens(); empty.shape = {1, 0};
    validateGenerationTensor(tokens(), {empty, std::vector<std::int64_t>{}});
    rejects([&] { validateGenerationTensor(tokens(), {tokens(), std::vector<std::int64_t>{}}); }, "concrete");
    rejects([&] { validateGenerationTensor(tokens(), {concrete, std::vector<std::int64_t>{0, -1, 2}}); }, "nonnegative");
    auto floating = concrete; floating.dtype = TensorDType::float32;
    rejects([&] { validateGenerationTensor(floating, value(floating, {0, 1, 2})); }, "tokenIds");
    auto invalid = latent(); invalid.shape = {1, -2};
    rejects([&] { validateGenerationTensor(invalid, value(latent(), {1, 2})); }, "dimension");
}

void validatesHalfPrecisionBitPatterns()
{
    for (const auto dtype : {TensorDType::float16, TensorDType::bfloat16})
    {
        auto spec = latent(); spec.dtype = dtype;
        validateGenerationTensor(spec, {spec, std::vector<std::uint16_t>{0, 0x3c00}});
        const auto infinity = static_cast<std::uint16_t>(dtype == TensorDType::float16 ? 0x7c00 : 0x7f80);
        rejects([&] { validateGenerationTensor(spec, {spec, std::vector<std::uint16_t>{0, infinity}}); }, "finite");
    }
}

void supportsTypedScalarPixelsAndLogits()
{
    auto scalar = latent(); scalar.shape = {}; scalar.layout = "scalar";
    scalar.dtype = TensorDType::float64;
    validateGenerationTensor(scalar, {scalar, std::vector<double>{0.25}});
    auto pixels = latent(GenerationSemantic::pixels); pixels.dtype = TensorDType::uint8;
    pixels.representationSpace = "pixels:srgb-uint8";
    validateGenerationTensor(pixels, {pixels, std::vector<std::uint8_t>{0, 255}});
    auto ids = tokens(); ids.shape = {1, 2}; ids.dtype = TensorDType::int32;
    validateGenerationTensor(ids, {ids, std::vector<std::int32_t>{0, 2147483647}});
    auto logits = latent(GenerationSemantic::logits); logits.representationSpace = "tokenizer:fixture-v1";
    validateGenerationTensor(logits, value(logits, {-0.25F, 0.5F}));
    logits.dtype = TensorDType::int32;
    rejects([&] { validateGenerationTensor(logits, {logits, std::vector<std::int32_t>{0, 1}}); }, "floating");
    rejects([] { static_cast<void>(generationArchitectureName(static_cast<GenerationArchitecture>(100))); }, "architecture");
    rejects([] { static_cast<void>(generationSemanticName(static_cast<GenerationSemantic>(100))); }, "semantic");
    rejects([] { static_cast<void>(tensorDTypeName(static_cast<TensorDType>(100))); }, "dtype");
}

void executesLogitsToTokensThroughExplicitBridge()
{
    auto ids = tokens(); ids.shape = {1, 2};
    const GenerationTensorSpec logits{TensorDType::float32, {1, 3}, "BV",
        GenerationSemantic::logits, "tokenizer:fixture-v1"};
    auto generatedId = tokens(); generatedId.shape = {1, 1};
    GenerationPlan plan{{{"prompt", ids}},
        {model("ar", GenerationArchitecture::autoregressive, ids, logits, {"", "prompt"}),
         {"selection", GenerationStageRole::bridge, std::nullopt,
          {{"scores", logits}}, {{"selected", generatedId}}, {{"scores", {"ar", "output"}}}}},
        {{"tokens", {"selection", "selected"}, generatedId}, {"logits", {"ar", "output"}, logits}}};
    const auto result = executeGenerationPlan(plan,
        {{"prompt", {ids, std::vector<std::int64_t>{4, 9}}}},
        [&](const GenerationStage &stage, const GenerationValues &inputs) {
            if (stage.id == "ar")
            {
                require(std::get<std::vector<std::int64_t>>(inputs.at("input").data) ==
                            std::vector<std::int64_t>{4, 9}, "autoregressive prompt was changed");
                return GenerationValues{{"output", value(logits, {-1, 4, 2})}};
            }
            require(std::get<std::vector<float>>(inputs.at("scores").data) == std::vector<float>{-1, 4, 2},
                    "bridge received incorrect logits");
            // A fixture callback stands in for the backend's sampling policy.
            return GenerationValues{{"selected", {generatedId, std::vector<std::int64_t>{1}}}};
        });
    require(std::get<std::vector<std::int64_t>>(result.at("tokens").data) == std::vector<std::int64_t>{1},
            "integer token output was not preserved");
    plan.stages.back().inputs.front().tensor = generatedId;
    rejects([&] { validateGenerationPlan(plan); }, "dtype");
}

void requiresExplicitSemanticAndSpaceBridges()
{
    for (const std::string space : {"auto", "unspecified", "unknown"})
    {
        auto invalid = latent(); invalid.representationSpace = space;
        rejects([&] { validateGenerationTensor(invalid, value(invalid, {1, 2})); }, "explicit identity");
        auto invalidPlan = single();
        invalidPlan.inputs.front().tensor.representationSpace = space;
        invalidPlan.stages.front().inputs.front().tensor.representationSpace = space;
        rejects([&] { validateGenerationPlan(invalidPlan); }, "explicit identity");
    }
    // Only the exact reserved sentinel strings are special; IDs remain opaque.
    auto caseSensitive = latent(); caseSensitive.representationSpace = "Unknown";
    validateGenerationTensor(caseSensitive, value(caseSensitive, {1, 2}));
    auto plan = single();
    plan.stages.push_back(model("flow", GenerationArchitecture::rectifiedFlow,
                                latent(GenerationSemantic::velocity), latent(), {"generate", "output"}));
    rejects([&] { validateGenerationPlan(plan); }, "semantic");
    plan.stages.back().inputs.front().tensor = latent();
    plan.stages.back().inputs.front().tensor.representationSpace = "vae:other";
    rejects([&] { validateGenerationPlan(plan); }, "representationSpace");
    plan.stages.back().inputs.front().tensor = latent();
    plan.stages.back().inputs.front().tensor.shape = {1, 4};
    rejects([&] { validateGenerationPlan(plan); }, "shape");
}

void executesAutoregressiveBridgeFlowHybrid()
{
    auto concreteTokens = tokens(); concreteTokens.shape = {1, 2};
    auto target = latent(); target.representationSpace = "vae:decoder-v2";
    GenerationStage bridge{"decode", GenerationStageRole::bridge, std::nullopt,
        {{"input", tokens()}}, {{"output", target}}, {{"input", {"ar", "output"}}}};
    GenerationPlan plan{{{"prompt", tokens()}},
        {model("ar", GenerationArchitecture::autoregressive, tokens(), tokens(), {"", "prompt"}),
         bridge, model("flow", GenerationArchitecture::flowMatching, target, target, {"decode", "output"})},
        {{"result", {"flow", "output"}, target}, {"tokens", {"ar", "output"}, tokens()}}};
    std::vector<std::string> calls;
    const auto result = executeGenerationPlan(plan,
        {{"prompt", {concreteTokens, std::vector<std::int64_t>{4, 9}}}},
        [&](const GenerationStage &stage, const GenerationValues &inputs) {
            calls.push_back(stage.id);
            if (stage.role == GenerationStageRole::bridge)
            {
                const auto &ids = std::get<std::vector<std::int64_t>>(inputs.at("input").data);
                return GenerationValues{{"output", value(target,
                    {static_cast<float>(ids[0]), static_cast<float>(ids[1])})}};
            }
            return identity(stage, inputs);
        });
    require(calls == std::vector<std::string>({"ar", "decode", "flow"}), "wrong stage order");
    require(std::get<std::vector<float>>(result.at("result").data) == std::vector<float>({4, 9}),
            "bridge did not consume and transform actual tokens");
    require(std::get<std::vector<std::int64_t>>(result.at("tokens").data) ==
            std::vector<std::int64_t>({4, 9}), "named intermediate output missing");
}

void validatesCompletePlanBeforeExecution()
{
    auto plan = single();
    plan.stages.front().bindings.front().source = {"future", "output"};
    bool executed = false;
    rejects([&] {
        static_cast<void>(executeGenerationPlan(plan, {{"seed", value(latent(), {1, 2})}},
            [&](const GenerationStage &stage, const GenerationValues &inputs) {
                executed = true; return identity(stage, inputs);
            }));
    }, "earlier stage");
    require(!executed, "invalid plan reached executor");
    plan = single(); plan.stages.front().bindings.clear();
    rejects([&] { validateGenerationPlan(plan); }, "binding");
    plan = single(); plan.stages.front().bindings.push_back(plan.stages.front().bindings.front());
    rejects([&] { validateGenerationPlan(plan); }, "duplicate");
    plan = single(); plan.stages.push_back(plan.stages.front());
    rejects([&] { validateGenerationPlan(plan); }, "duplicate");
    plan = single(); plan.stages.front().architecture.reset();
    rejects([&] { validateGenerationPlan(plan); }, "architecture");
    plan = single(); plan.stages.front().role = GenerationStageRole::bridge;
    rejects([&] { validateGenerationPlan(plan); }, "architecture");
    plan = single(); plan.outputs.front().source.port = "missing";
    rejects([&] { validateGenerationPlan(plan); }, "port");
}

void validatesAllExecutorOutputsAndInputs()
{
    const auto plan = single();
    const GenerationValues valid{{"seed", value(latent(), {1, 2})}};
    rejects([&] { static_cast<void>(executeGenerationPlan(plan, {}, identity)); }, "missing");
    auto extra = valid; extra.emplace("undeclared", value(latent(), {1, 2}));
    rejects([&] { static_cast<void>(executeGenerationPlan(plan, extra, identity)); }, "unexpected");
    rejects([&] { static_cast<void>(executeGenerationPlan(plan, valid, {})); }, "executor");
    rejects([&] { static_cast<void>(executeGenerationPlan(plan, valid,
        [](const GenerationStage &, const GenerationValues &) { return GenerationValues{}; })); }, "missing");
    rejects([&] { static_cast<void>(executeGenerationPlan(plan, valid,
        [](const GenerationStage &, const GenerationValues &) {
            return GenerationValues{{"output", value(latent(), {1, 2})},
                                    {"undeclared", value(latent(), {3, 4})}};
        })); }, "unexpected");
    rejects([&] { static_cast<void>(executeGenerationPlan(plan, valid,
        [](const GenerationStage &, const GenerationValues &) {
            return GenerationValues{{"output", value(latent(), {1, INFINITY})}};
        })); }, "finite");
    rejects([&] { static_cast<void>(executeGenerationPlan(plan, valid,
        [](const GenerationStage &, const GenerationValues &) -> GenerationValues {
            throw std::runtime_error("fixture failure");
        })); }, "generate");
}

void checksDynamicConnectionsAtRuntime()
{
    auto plan = single();
    plan.stages.front().outputs.front().tensor.shape = {1, -1};
    plan.stages.push_back(model("next", GenerationArchitecture::diffusion,
                                latent(), latent(), {"generate", "output"}));
    validateGenerationPlan(plan);
    int calls = 0;
    rejects([&] { static_cast<void>(executeGenerationPlan(plan,
        {{"seed", value(latent(), {1, 2})}}, [&](const GenerationStage &, const GenerationValues &) {
            ++calls;
            auto spec = latent(); spec.shape = {1, 3};
            return GenerationValues{{"output", value(spec, {1, 2, 3})}};
        })); }, "shape");
    require(calls == 1, "incompatible concrete payload reached next executor");
}

void preservesPredictionAndVelocityDistinctions()
{
    for (const auto semantic : {GenerationSemantic::epsilon, GenerationSemantic::sample,
                               GenerationSemantic::vPrediction, GenerationSemantic::velocity})
    {
        auto plan = single();
        plan.stages.front().outputs.front().tensor = latent(semantic);
        plan.outputs.front().tensor = latent(semantic);
        const auto result = executeGenerationPlan(plan, {{"seed", value(latent(), {1, 2})}},
            [&](const GenerationStage &, const GenerationValues &) {
                return GenerationValues{{"output", value(latent(semantic), {3, 4})}};
            });
        require(result.at("result").spec.semantic == semantic, "prediction semantic was rewritten");
    }
}

} // namespace

int main()
{
    int failures = 0;
    const auto run = [&](const char *name, const std::function<void()> &test) {
        try { test(); std::cout << "PASS " << name << '\n'; }
        catch (const std::exception &error)
        { ++failures; std::cerr << "FAIL " << name << ": " << error.what() << '\n'; }
    };
    run("all architecture declarations", supportsEveryArchitecture);
    run("typed payload, finite values and overflow", rejectsInvalidPayloads);
    run("dynamic dimensions and empty token prefix", supportsConcreteDynamicAndEmptyTokenInputs);
    run("half precision finite values", validatesHalfPrecisionBitPatterns);
    run("typed scalar, pixels and logits", supportsTypedScalarPixelsAndLogits);
    run("autoregressive logits and explicit token selection", executesLogitsToTokensThroughExplicitBridge);
    run("explicit semantic and representation bridges", requiresExplicitSemanticAndSpaceBridges);
    run("autoregressive decode flow hybrid", executesAutoregressiveBridgeFlowHybrid);
    run("preflight plan validation", validatesCompletePlanBeforeExecution);
    run("executor input and output validation", validatesAllExecutorOutputsAndInputs);
    run("concrete dynamic connection validation", checksDynamicConnectionsAtRuntime);
    run("prediction semantic preservation", preservesPredictionAndVelocityDistinctions);
    return failures == 0 ? 0 : 1;
}
