#include "NativePose.hpp"
#include "NativeCachePolicy.hpp"
#include "PoseProcessing.hpp"
#include "OnnxInlineModel.hpp"
#include <fstream>
#include <thread>
#include <map>
#include <tuple>
#if IILD_HAS_POSE
#include <onnxruntime_cxx_api.h>
#endif

namespace iiLocalDiffusion {
#if IILD_HAS_POSE
namespace {
void runnable(const std::atomic_bool &cancelled, const std::shared_ptr<NativeExecutionControl> &control)
{
    if (cancelled || (control && !control->waitUntilRunnable(cancelled))) throw std::runtime_error("Pose preprocessing cancelled.");
}
std::vector<char> readModel(const std::filesystem::path &source, const std::atomic_bool &cancelled,
    const std::shared_ptr<NativeExecutionControl> &control)
{
    runnable(cancelled,control);
    const auto path = std::filesystem::canonical(source);
    if (!std::filesystem::is_regular_file(path) || path.extension() != ".onnx")
        throw std::runtime_error("Pose requires a regular local .onnx model.");
    const auto identity = native_detail::modelMetadataIdentity(path);
    const auto size = std::filesystem::file_size(path);
    // ONNX protobuf models are limited to 2 GiB; sidecar/external tensors are not supported.
    if (!size || size >= 2ull*1024*1024*1024) throw std::runtime_error("Pose model must be a nonempty inline ONNX model below 2 GiB.");
    std::ifstream input(path,std::ios::binary);
    if (!input) throw std::runtime_error("Cannot read the pose model.");
    std::vector<char> bytes(static_cast<size_t>(size));
    for (size_t offset = 0; offset < bytes.size();) {
        runnable(cancelled,control);
        const auto count = std::min<size_t>(4*1024*1024,bytes.size()-offset);
        if (!input.read(bytes.data()+offset,static_cast<std::streamsize>(count))) throw std::runtime_error("Short pose model read.");
        offset += count;
    }
    if (native_detail::modelMetadataIdentity(path) != identity) throw std::runtime_error("Pose model changed while loading.");
    runnable(cancelled,control);
    native_detail::requireInlineOnnx(bytes);
    return bytes;
}
std::vector<int64_t> shape(Ort::Session &session, bool input, size_t index)
{
    const auto info = input ? session.GetInputTypeInfo(index) : session.GetOutputTypeInfo(index);
    const auto tensor = info.GetTensorTypeAndShapeInfo();
    if (tensor.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) throw std::runtime_error("DWPose requires float32 tensors.");
    return tensor.GetShape();
}
struct Model {
    std::vector<char> bytes;
    Ort::Session session;
    std::string input;
    std::vector<std::string> output;
    std::vector<int64_t> inputShape;
    Model(Ort::Env &env, Ort::SessionOptions &options, const std::filesystem::path &path,
        const std::atomic_bool &cancelled, const std::shared_ptr<NativeExecutionControl> &control)
        : bytes(readModel(path,cancelled,control)), session(env,bytes.data(),bytes.size(),options)
    {
        runnable(cancelled,control);
        if (session.GetInputCount() != 1) throw std::runtime_error("DWPose must have one image input.");
        Ort::AllocatorWithDefaultOptions allocator;
        input = session.GetInputNameAllocated(0,allocator).get();
        for (size_t i = 0; i < session.GetOutputCount(); ++i) output.emplace_back(session.GetOutputNameAllocated(i,allocator).get());
        inputShape = shape(session,true,0);
    }
    std::vector<Ort::Value> run(std::vector<float> &chw, const std::atomic_bool &cancelled,
        const std::shared_ptr<NativeExecutionControl> &control)
    {
        runnable(cancelled,control);
        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
        auto tensor = Ort::Value::CreateTensor<float>(memory,chw.data(),chw.size(),inputShape.data(),inputShape.size());
        std::vector<const char *> names;
        for (const auto &name : output) names.push_back(name.c_str());
        const char *inputName = input.c_str();
        Ort::RunOptions options;
        // ORT supports thread-safe RunOptions termination; stop/join before the
        // options object is destroyed, including when Run throws.
        std::jthread monitor([&](std::stop_token stop) {
            while (!stop.stop_requested()) {
                if (cancelled) {
                    try { options.SetTerminate(); } catch (...) {}
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
        auto result = session.Run(options,&inputName,&tensor,1,names.data(),names.size());
        monitor.request_stop();
        runnable(cancelled,control);
        return result;
    }
};
std::span<const float> tensorSpan(Ort::Value &value, const std::vector<int64_t> &expected)
{
    const auto info = value.GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || info.GetShape() != expected)
        throw std::runtime_error("Unexpected DWPose output tensor.");
    return {value.GetTensorData<float>(),info.GetElementCount()};
}
}
struct NativePoseSession::Impl {
    // Destruction order releases sessions before their source buffers and Env.
    Ort::Env environment{ORT_LOGGING_LEVEL_WARNING,"iiLocalDiffusion.Pose"};
    Ort::SessionOptions options;
    unsigned threadCount;
    std::unique_ptr<Model> detector, pose;
    std::timed_mutex mutex;
    Impl(const std::filesystem::path &det, const std::filesystem::path &keypoints,
        const std::atomic_bool &cancelled, unsigned count, const std::shared_ptr<NativeExecutionControl> &control)
        : threadCount(count ? count : std::max(1u,std::thread::hardware_concurrency()))
    {
        if (threadCount > 1024) throw std::invalid_argument("Pose thread count exceeds 1024.");
        options.SetIntraOpNumThreads(static_cast<int>(threadCount));
        options.SetInterOpNumThreads(1);
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        options.AddConfigEntry("session.intra_op.allow_spinning","0");
        // CPU EP only: no CoreML compiled-model disk cache is created.
        detector = std::make_unique<Model>(environment,options,det,cancelled,control);
        if (detector->inputShape != std::vector<int64_t>{1,3,640,640} || detector->output.size() != 1
            || shape(detector->session,false,0) != std::vector<int64_t>{1,8400,85})
            throw std::runtime_error("Expected YOLOX-L detector input [1,3,640,640], output [1,8400,85].");
        pose = std::make_unique<Model>(environment,options,keypoints,cancelled,control);
        const auto &s = pose->inputShape;
        if (s.size() != 4 || s[0] != 1 || s[1] != 3 || !((s[2] == 384 && s[3] == 288) || (s[2] == 256 && s[3] == 192))
            || pose->output.size() != 2 || shape(pose->session,false,0) != std::vector<int64_t>{1,133,2*s[3]}
            || shape(pose->session,false,1) != std::vector<int64_t>{1,133,2*s[2]})
            throw std::runtime_error("Expected DWPose 133-joint SimCC model, 192x256 or 288x384.");
    }
};
NativePoseSession::NativePoseSession(const std::filesystem::path &detector, const std::filesystem::path &pose,
    const std::atomic_bool &cancelled, unsigned threads, const std::shared_ptr<NativeExecutionControl> &control)
    : impl(std::make_unique<Impl>(detector,pose,cancelled,threads,control)) {}
NativePoseSession::~NativePoseSession() = default;
bool nativePoseAvailable() noexcept { return true; }
std::uint64_t NativePoseSession::modelBytes() const noexcept { return impl->detector->bytes.size()+impl->pose->bytes.size(); }
unsigned NativePoseSession::threads() const noexcept { return impl->threadCount; }
NativeReferenceImage NativePoseSession::process(const NativeReferenceImage &image, const std::atomic_bool &cancelled,
    const std::shared_ptr<NativeExecutionControl> &control)
{
    std::unique_lock lock(impl->mutex,std::defer_lock);
    do { runnable(cancelled,control); } while (!lock.try_lock_for(std::chrono::milliseconds(20)));
    const native_detail::PosePixels pixels{image.rgb,image.width,image.height};
    auto tensor = native_detail::preparePoseDetector(pixels);
    auto detection = impl->detector->run(tensor,cancelled,control);
    const auto boxes = native_detail::decodePoseDetector(tensorSpan(detection[0],{1,8400,85}),image.width,image.height);
    std::vector<native_detail::WholeBodyPose> poses;
    const auto &s = impl->pose->inputShape;
    for (const auto &box : boxes) {
        runnable(cancelled,control);
        auto crop = native_detail::preparePoseCrop(pixels,box,int(s[3]),int(s[2]));
        auto values = impl->pose->run(crop.chw,cancelled,control);
        poses.push_back(native_detail::decodePoseSimcc(tensorSpan(values[0],{1,133,2*s[3]}),
            tensorSpan(values[1],{1,133,2*s[2]}),crop));
    }
    runnable(cancelled,control);
    NativeReferenceImage result{image.width,image.height,native_detail::drawPoseHint(poses,image.width,image.height)};
    runnable(cancelled,control);
    return result;
}
#else
struct NativePoseSession::Impl {};
NativePoseSession::NativePoseSession(const std::filesystem::path &, const std::filesystem::path &,
    const std::atomic_bool &, unsigned, const std::shared_ptr<NativeExecutionControl> &)
{ throw std::runtime_error("Native pose preprocessing is not enabled in this build."); }
NativePoseSession::~NativePoseSession() = default;
bool nativePoseAvailable() noexcept { return false; }
std::uint64_t NativePoseSession::modelBytes() const noexcept { return 0; }
unsigned NativePoseSession::threads() const noexcept { return 0; }
NativeReferenceImage NativePoseSession::process(const NativeReferenceImage &, const std::atomic_bool &,
    const std::shared_ptr<NativeExecutionControl> &)
{ throw std::runtime_error("Native pose preprocessing is not enabled in this build."); }
#endif

namespace {
struct PoseCache {
    std::timed_mutex mutex;
    std::atomic_uint64_t releaseEpoch{0};
    using Key = std::tuple<std::filesystem::path,std::filesystem::path,unsigned>;
    std::map<Key,std::unique_ptr<NativePoseSession>> sessions;
};
PoseCache &poseCache() { static PoseCache cache; return cache; }
}
void releaseNativePoseCache() noexcept
{
    auto &cache = poseCache();
    ++cache.releaseEpoch;
    std::unique_lock lock(cache.mutex,std::try_to_lock);
    if (lock.owns_lock()) cache.sessions.clear();
}
NativePoseResult processNativePose(const std::filesystem::path &detector,
    const std::filesystem::path &pose, const NativeReferenceImage &image,
    const std::atomic_bool &cancelled, unsigned threads, const std::shared_ptr<NativeExecutionControl> &control)
{
    if (!detector.is_absolute() || !pose.is_absolute()) throw std::invalid_argument("Pose models require absolute local paths.");
    auto ready = [&] {
        if (cancelled || (control && !control->waitUntilRunnable(cancelled))) throw std::runtime_error("Pose preprocessing cancelled.");
    };
    auto &cache = poseCache();
    std::unique_lock lock(cache.mutex,std::defer_lock);
    do { ready(); } while (!lock.try_lock_for(std::chrono::milliseconds(20)));
    struct DeferredRelease {
        PoseCache &cache; std::uint64_t epoch;
        ~DeferredRelease() { if (cache.releaseEpoch != epoch) cache.sessions.clear(); }
    } release{cache,cache.releaseEpoch.load()};
    const auto key = PoseCache::Key{detector.lexically_normal(),pose.lexically_normal(),threads};
    auto found = cache.sessions.find(key);
    NativePoseResult result;
    result.modelCacheHit = found != cache.sessions.end();
    if (found == cache.sessions.end()) {
        auto session = std::make_unique<NativePoseSession>(detector,pose,cancelled,threads,control);
        found = cache.sessions.emplace(key,std::move(session)).first;
    }
    result.modelBytes = found->second->modelBytes();
    result.threads = found->second->threads();
    result.image = found->second->process(image,cancelled,control);
    return result;
}
}
