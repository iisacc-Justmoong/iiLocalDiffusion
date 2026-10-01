#pragma once
#include <cstdint>
#include <span>
#include <stdexcept>

namespace iiLocalDiffusion::native_detail {
// A bounded protobuf wire walk, not a second ONNX semantic validator. Reject
// every tensor's external storage before the inference runtime can touch it.
// Field numbers follow onnx/onnx.proto; unknown non-storage fields are skipped.
inline void requireInlineOnnx(std::span<const char> bytes)
{
    enum class Message { Model, Graph, Node, Attribute, Tensor, Sparse, Function, Training };
    auto failure = [] { throw std::invalid_argument("Pose requires a well-formed inline ONNX model; external tensor files are forbidden."); };
    auto varint = [&](std::span<const char> &data) -> std::uint64_t {
        std::uint64_t result = 0;
        for (unsigned shift = 0; shift < 70; shift += 7) {
            if (data.empty()) failure();
            const auto value = static_cast<unsigned char>(data.front()); data = data.subspan(1);
            if (shift == 63 && value > 1) failure();
            result |= std::uint64_t(value & 127) << shift;
            if (!(value & 128)) return result;
        }
        failure(); return 0;
    };
    auto walk = [&](auto &&self, std::span<const char> data, Message kind, unsigned depth) -> void {
        if (depth > 64) failure();
        while (!data.empty()) {
            const auto tag = varint(data), field = tag >> 3, wire = tag & 7;
            if (field == 0 || field > 0x1fffffff) failure();
            std::uint64_t value = 0;
            std::span<const char> payload;
            if (wire == 0) value = varint(data);
            else if (wire == 2 || wire == 1 || wire == 5) {
                const auto length = wire == 2 ? varint(data) : wire == 1 ? 8 : 4;
                if (length > data.size()) failure();
                payload = data.first(static_cast<size_t>(length)); data = data.subspan(static_cast<size_t>(length));
            } else failure(); // Groups are not part of the ONNX schema.
            if (kind == Message::Tensor && (field == 13 || (field == 14 && (wire != 0 || value != 0)))) failure();
            auto nested = [&](Message next) {
                if (wire != 2) failure();
                self(self,payload,next,depth+1);
            };
            switch (kind) {
            case Message::Model:
                if (field == 7) nested(Message::Graph);
                if (field == 20) nested(Message::Training);
                if (field == 25) nested(Message::Function);
                break;
            case Message::Graph:
                if (field == 1) nested(Message::Node);
                if (field == 5) nested(Message::Tensor);
                if (field == 15) nested(Message::Sparse);
                break;
            case Message::Node:
                if (field == 5) nested(Message::Attribute);
                break;
            case Message::Attribute:
                if (field == 5 || field == 10) nested(Message::Tensor);
                if (field == 6 || field == 11) nested(Message::Graph);
                if (field == 22 || field == 23) nested(Message::Sparse);
                break;
            case Message::Sparse:
                if (field == 1 || field == 2) nested(Message::Tensor);
                break;
            case Message::Function:
                if (field == 7) nested(Message::Node);
                if (field == 11) nested(Message::Attribute);
                break;
            case Message::Training:
                if (field == 1 || field == 2) nested(Message::Graph);
                break;
            case Message::Tensor: break;
            }
        }
    };
    if (bytes.empty()) failure();
    walk(walk,bytes,Message::Model,0);
}
}
