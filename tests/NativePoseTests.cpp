#include "Generation/NativePose.hpp"
#include "Generation/OnnxInlineModel.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <fstream>
#include <future>
#include <iostream>
#include <thread>
using namespace iiLocalDiffusion;
namespace {
void require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
template<class F> void rejects(F f) {
    bool rejected = false;
    try { f(); } catch (const std::exception &) { rejected = true; }
    require(rejected,"Invalid ONNX model or cancelled request was accepted");
}
// Tiny test-only protobuf writer: executes actual ONNX Identity graphs, not a
// mocked inference function. Fixtures contain known outputs, not learned weights.
using Bytes = std::string;
Bytes varint(uint64_t n) { Bytes out; do { const auto low = n & 127; n >>= 7; out += char(low | (n ? 128 : 0)); } while(n); return out; }
Bytes integer(unsigned field, uint64_t n) { return varint(field*8)+varint(n); }
Bytes block(unsigned field, const Bytes &payload) { return varint(field*8+2)+varint(payload.size())+payload; }
Bytes dimensions(const std::vector<int64_t> &shape) {
    Bytes out; for (const auto d : shape) out += block(1,integer(1,d)); return out;
}
Bytes info(const Bytes &name, const std::vector<int64_t> &shape) {
    return block(1,name)+block(2,block(1,integer(1,1)+block(2,dimensions(shape))));
}
struct Output { Bytes name; std::vector<int64_t> shape; std::vector<float> values; };
Bytes model(std::vector<int64_t> shape, const std::vector<Output> &outputs, bool external = false)
{
    Bytes graph = block(2,"pose-contract")+block(11,info("image",shape));
    for (const auto &out : outputs) {
        Bytes tensor;
        for (const auto d : out.shape) tensor += integer(1,d);
        tensor += integer(2,1)+block(8,out.name+".weights");
        if (external) tensor += integer(14,1)+block(13,block(1,"location")+block(2,"forbidden.bin"));
        else {
            Bytes raw; raw.reserve(out.values.size()*4);
            for (const auto value : out.values) {
                const auto bits = std::bit_cast<uint32_t>(value);
                for (int byte = 0; byte < 4; ++byte) raw += char((bits >> (8*byte)) & 255);
            }
            tensor += block(9,raw);
        }
        graph += block(5,tensor);
        graph += block(1,block(1,out.name+".weights")+block(2,out.name)+block(4,"Identity"));
        graph += block(12,info(out.name,out.shape));
    }
    return integer(1,8)+block(2,"iiLocalDiffusion-tests")+block(7,graph)+block(8,integer(2,13));
}
void write(const std::filesystem::path &path, const Bytes &data) {
    std::ofstream file(path,std::ios::binary); file.write(data.data(),data.size()); require(bool(file),"Cannot write fixture");
}
}
int main(int argc, char **argv)
{
    try {
        require(argc == 2,"Pass a build-local fixture directory");
        const std::filesystem::path root(argv[1]); std::filesystem::create_directories(root);
        Output detection{"boxes",{1,8400,85},std::vector<float>(8400*85)};
        // Two non-overlapping people in the original 128x128 image.
        for (int i = 0; i < 2; ++i) {
            auto *p = detection.values.data()+i*85;
            p[0] = (i ? 480.f : 160.f)/8-i; p[1] = 320.f/8;
            p[2] = std::log(160.f/8); p[3] = std::log(320.f/8); p[4] = 1; p[5] = .9f;
        }
        Output sx{"simcc_x",{1,133,384},std::vector<float>(133*384)};
        Output sy{"simcc_y",{1,133,512},std::vector<float>(133*512)};
        // Body shoulders/elbows only, ensuring independently positioned people.
        for (int i : {5,6,7,8}) { sx.values[i*384+192] = .9f; sy.values[i*512+256] = .8f; }
        const auto det = model({1,3,640,640},{detection});
        const auto pose = model({1,3,256,192},{sx,sy});
        native_detail::requireInlineOnnx(det);
        rejects([&] { native_detail::requireInlineOnnx(model({1,3,640,640},{detection},true)); });
        // Also reject tensors hidden in graph attributes, functions and sparse values.
        const auto externalTensor = integer(14,1);
        const auto attribute = block(5,externalTensor);
        const auto node = block(5,attribute);
        for (const auto &bytes : {block(7,block(1,node)), block(25,block(11,attribute)),
                block(7,block(15,block(1,externalTensor))), block(20,block(2,block(5,externalTensor))),
                Bytes("\x3a\xff",2), Bytes("\x80",1), Bytes("\x00",1)})
            rejects([&] { native_detail::requireInlineOnnx(bytes); });
        const auto detectorPath = root/"detector.onnx", posePath = root/"pose.onnx";
        write(detectorPath,det); write(posePath,pose);
        std::atomic_bool cancelled{false};
        if (!nativePoseAvailable()) {
            rejects([&] { NativePoseSession session(detectorPath,posePath,cancelled); });
            std::cout << "Pose disabled-build contract passed\n";
            return 0;
        }
        NativePoseSession session(detectorPath,posePath,cancelled,2);
        require(session.modelBytes() == det.size()+pose.size() && session.threads() == 2,"Resident model ownership/accounting");
        NativeReferenceImage source{128,128,std::vector<std::uint8_t>(128*128*3,99)};
        releaseNativePoseCache();
        const auto firstRuntime = processNativePose(detectorPath,posePath,source,cancelled,2);
        require(!firstRuntime.modelCacheHit,"First runtime request unexpectedly reused models");
        // The sessions must never reopen either model during inference.
        std::filesystem::remove(detectorPath); std::filesystem::remove(posePath);
        const auto reusedRuntime = processNativePose(detectorPath,posePath,source,cancelled,2);
        require(reusedRuntime.modelCacheHit && reusedRuntime.image.rgb == firstRuntime.image.rgb,
            "Runtime reopened or evicted resident Pose models between requests");
        releaseNativePoseCache();
        rejects([&] { processNativePose(detectorPath,posePath,source,cancelled,2); });
        const auto output = session.process(source,cancelled);
        require(output.width == 128 && output.height == 128 && output.rgb.size() == source.rgb.size(),"Pose output dimensions");
        require(output.rgb != source.rgb && source.rgb == std::vector<std::uint8_t>(128*128*3,99),"RGB passthrough or source mutation");
        auto visible = [&](int x, int y) { return output.rgb[(y*128+x)*3] != 0 || output.rgb[(y*128+x)*3+1] != 0; };
        require(visible(32,64) && visible(96,64) && !visible(64,64),"Two-person native ONNX mapping");
        require(session.process(source,cancelled).rgb == output.rgb,"Resident repeated inference changed");
        auto control = std::make_shared<NativeExecutionControl>(); control->setPaused(true);
        auto waiting = std::async(std::launch::async,[&] { return session.process(source,cancelled,control); });
        const auto until = std::chrono::steady_clock::now()+std::chrono::seconds(3);
        while (!control->isWaiting() && std::chrono::steady_clock::now() < until) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        const bool parked = control->isWaiting(); control->setPaused(false);
        const auto resumed = waiting.get();
        require(parked && resumed.rgb == output.rgb,"Pause/resume lost resident session");
        cancelled = true; rejects([&] { session.process(source,cancelled); });
        cancelled = false; require(session.process(source,cancelled).rgb == output.rgb,"Cancellation poisoned session");
        write(detectorPath,model({1,3,640,640},{detection},true)); write(posePath,pose);
        rejects([&] { NativePoseSession external(detectorPath,posePath,cancelled,2); });
        write(detectorPath,model({1,3,64,64},{detection}));
        rejects([&] { NativePoseSession wrongShape(detectorPath,posePath,cancelled,2); });
        write(detectorPath,det);
        write(posePath,model({1,3,256,192},{sy,sx}));
        rejects([&] { NativePoseSession wrongAxes(detectorPath,posePath,cancelled,2); });
        std::cout << "Native in-memory ONNX Pose contracts passed\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
