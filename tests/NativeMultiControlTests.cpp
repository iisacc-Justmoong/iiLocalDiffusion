#include "runtime/control_stack.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
int main() {
    try {
        ControlNetStack stack;
        std::vector<uint8_t> imageA(12, 32), imageB(12, 224), mask{0, 255};
        sd_control_input_t inputs[]{
            {{2, 2, 3, imageA.data()}, {2, 1, 1, mask.data()}, .5f},
            {{2, 2, 3, imageB.data()}, {}, 1.5f}};
        require(stack.set(inputs, 2), "Valid stack rejected");
        imageA.assign(12, 255); mask.assign(2, 0);
        std::vector<size_t> order;
        auto compute = [&](size_t index, const sd::Tensor<float>& hint)
            -> std::optional<std::vector<sd::Tensor<float>>> {
            order.push_back(index);
            require(hint.shape()[0] == 32 && hint.shape()[1] == 24, "Hint not resized to current pass");
            require(std::abs(hint.values()[0] - (index ? 224.f : 32.f) / 255.f) < 1e-5f,
                "Input pixels were not independently owned");
            return std::vector<sd::Tensor<float>>{
                sd::Tensor<float>({2, 1, 3, 2}, std::vector<float>(12, index ? 4 : 2)),
                sd::Tensor<float>({1, 1, 1, 1}, {index ? 8.f : 4.f})};
        };
        auto result = stack.compute(32, 24, compute);
        require(result && order == std::vector<size_t>({0, 1}), "Stack did not execute every model in order");
        for (size_t i = 0; i < 12; ++i)
            require((*result)[0].values()[i] == (i % 2 ? 7.f : 6.f), "Masked weighted sum was normalized or lost a layer");
        require((*result)[1].values()[0] == 13, "Mask did not follow residual resolution");
        order.clear();
        require(stack.compute(32, 24, compute).has_value() && order.size() == 2, "Warm hints skipped model compute");
        require(!stack.compute(32, 24, [&](size_t index, const auto& hint) {
            return index ? std::optional<std::vector<sd::Tensor<float>>>{} : compute(index, hint);
        }), "Partial stack failure was silently ignored");
        require(!stack.compute(32, 24, [&](size_t index, const auto&) {
            return std::optional<std::vector<sd::Tensor<float>>>{{sd::Tensor<float>({index ? 3 : 2, 1},
                std::vector<float>(index ? 3 : 2, 1.f))}};
        }), "Residual mismatch accepted");
        require(!stack.compute(32, 24, [&](size_t index, const auto&) {
            return std::optional<std::vector<sd::Tensor<float>>>{{sd::Tensor<float>({1, 1},
                {index ? std::numeric_limits<float>::infinity() : 1.f})}};
        }), "Non-finite later residual was accepted");
        auto bad = inputs[1]; bad.strength = std::numeric_limits<float>::quiet_NaN();
        require(!stack.set(&bad, 1) && stack.size() == 2, "Invalid edit replaced the active stack");
        bool aborted = true;
        sd_set_abort_callback([](void* p) { return *static_cast<bool*>(p); }, &aborted);
        order.clear();
        require(!stack.compute(32, 24, compute) && order.empty(), "Abort still ran models");
        aborted = false;
        require(!stack.compute(32, 24, [&](size_t index, const auto& hint) {
            auto value = compute(index, hint); aborted = true; return value;
        }) && order == std::vector<size_t>{0}, "Cancellation between models published a partial stack");
        sd_set_abort_callback(nullptr, nullptr);
        inputs[0].strength = 0; inputs[0].mask = {};
        require(stack.set(inputs, 2), "Zero weight rejected");
        int calls = 0;
        result = stack.compute(64, 48, [&](size_t, const auto& hint) {
            ++calls; require(hint.shape()[0] == 64 && hint.shape()[1] == 48, "New canvas reused old hint size");
            return std::optional<std::vector<sd::Tensor<float>>>{{sd::Tensor<float>({1, 1}, {2})}};
        });
        require(result && calls == 2 && (*result)[0].values()[0] == 3, "Zero strength skipped execution or changed weight sum");
        require(stack.set(nullptr, 0) && !stack.active(), "Disable retained inputs");
        require(!stack.set(inputs, 65), "Excessive layer count accepted");
        std::cout << "Multi-ControlNet residual composition passed\n";
    } catch (const std::exception& error) { sd_set_abort_callback(nullptr, nullptr); std::cerr << error.what() << '\n'; return 1; }
}
