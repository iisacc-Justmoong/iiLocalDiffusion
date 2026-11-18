#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace iild {

// Treat only the ASCII suffix case-insensitively; shard paths and tensor names
// are still interpreted exactly as recorded in the index.
[[nodiscard]] constexpr bool hasSafetensorsIndexSuffix(std::string_view path) noexcept {
    constexpr std::string_view suffix=".safetensors.index.json";
    if (path.size()<suffix.size()) return false;
    path.remove_prefix(path.size()-suffix.size());
    for (std::size_t i=0;i<suffix.size();++i) {
        const char c=path[i]>='A' && path[i]<='Z' ? static_cast<char>(path[i]-'A'+'a') : path[i];
        if (c!=suffix[i]) return false;
    }
    return true;
}

struct SafetensorsIndexShard {
    std::filesystem::path relativePath;
    std::filesystem::path resolvedPath;
    std::string identity;
    std::uint64_t sizeBytes = 0;
};
struct SafetensorsIndexTensor {
    std::string name;
    std::string dtype;
    std::vector<std::uint64_t> shape;
    std::size_t shard = 0;
};
struct SafetensorsIndex {
    std::filesystem::path sourcePath;
    std::filesystem::path resolvedPath;
    std::string identity;
    std::uint64_t sizeBytes = 0;
    std::uint64_t metadataBytesRead = 0;
    std::vector<SafetensorsIndexShard> shards;
    std::vector<SafetensorsIndexTensor> tensors;
    std::map<std::string, std::string> sharedMetadata;
};

// Inspect a standard Safetensors index without copying weights. Relative paths
// and exact tensor coverage are validated; resolved symlink targets are bound
// to the snapshot. Identity checks do not hash or execute tensor payloads.
[[nodiscard]] SafetensorsIndex inspectSafetensorsIndex(const std::filesystem::path &input);
[[nodiscard]] bool verifySafetensorsIndex(const SafetensorsIndex &index) noexcept;
// Index-only dependency discovery; it does not validate shard contents.
[[nodiscard]] std::vector<std::filesystem::path> safetensorsIndexShardPaths(const std::filesystem::path &input);
[[nodiscard]] std::string safetensorsIndexJson(const SafetensorsIndex &index);

struct PyTorchIndexShard {
    std::filesystem::path relativePath;
    std::filesystem::path resolvedPath;
    std::string identity;
    std::uint64_t sizeBytes = 0;
};
struct PyTorchIndexTensor {
    std::string name;
    std::size_t shard = 0;
};
struct PyTorchIndex {
    std::filesystem::path sourcePath;
    std::filesystem::path resolvedPath;
    std::string identity;
    std::uint64_t sizeBytes = 0;
    std::uint64_t metadataBytesRead = 0;
    std::vector<PyTorchIndexShard> shards;
    std::vector<PyTorchIndexTensor> tensors;
};

// Verifies bounded index structure, safe paths, shard signatures and source
// identities. Tensor membership and shapes require the safe Torch worker.
[[nodiscard]] PyTorchIndex inspectPyTorchIndex(const std::filesystem::path &input);
[[nodiscard]] bool verifyPyTorchIndex(const PyTorchIndex &index) noexcept;
[[nodiscard]] std::string pyTorchIndexJson(const PyTorchIndex &index);

struct WanDirectoryComponent {
    std::filesystem::path sourcePath;
    std::filesystem::path resolvedPath;
    std::vector<std::filesystem::path> dependencies;
    bool sharded = false;
};
struct WanDirectoryRoute {
    std::filesystem::path directory;
    std::filesystem::path manifest;
    std::string scheduler;
    WanDirectoryComponent model;
    WanDirectoryComponent textEncoder;
    WanDirectoryComponent videoVae;
    std::optional<WanDirectoryComponent> highNoiseModel;
    std::optional<double> moeBoundary;
};

// Resolve the declared local text-to-video pipeline without loading or copying
// tensor data. Only one unambiguous weight group per component is accepted.
// The result identifies files; the native engine validates the actual model.
[[nodiscard]] WanDirectoryRoute inspectWanDirectory(const std::filesystem::path &directory);
[[nodiscard]] std::string wanDirectoryJson(const WanDirectoryRoute &route);

// Additive interface for text and image-conditioned Wan directories. The legacy
// inspectWanDirectory contract remains limited to text-to-video pipelines.
struct WanPipelineDirectoryRoute {
    WanDirectoryRoute common;
    std::string pipeline;
    std::optional<WanDirectoryComponent> visionEncoder;
};
[[nodiscard]] WanPipelineDirectoryRoute inspectWanPipelineDirectory(const std::filesystem::path &directory);
[[nodiscard]] std::string wanPipelineDirectoryJson(const WanPipelineDirectoryRoute &route);

struct ModelTensorProbe {
    std::vector<std::string> matches;
    bool embeddedPackage = false;
};
// Checks exact original tensor names in a Safetensors file, validated shard
// index, or GGUF header. Package status is a metadata hint; neither check
// validates tensor values or compatibility.
[[nodiscard]] ModelTensorProbe probeModelTensorNames(const std::filesystem::path &input,
    std::span<const std::string_view> names);
[[nodiscard]] std::vector<std::string> findModelTensorNames(const std::filesystem::path &input,
    std::span<const std::string_view> names);
[[nodiscard]] bool modelContainsTensorNames(const std::filesystem::path &input,
    std::span<const std::string_view> names);

struct ModelPackagingProgress {
    std::string phase;
    std::string detail;
    std::uint64_t completedBytes = 0;
    std::uint64_t totalBytes = 0;
};
using ModelPackagingObserver = std::function<void(const ModelPackagingProgress &)>;

namespace detail {
// Internal inspection contract for legacy package routing. Index bytes and
// references are validated without materializing or loading tensor weights.
struct PackageShardIndex {
    std::string path;
    std::vector<std::pair<std::string, std::string>> weights;
};
[[nodiscard]] std::vector<PackageShardIndex> inspectPackageShardIndexes(const std::filesystem::path &input);
// Metadata-only discovery for native consumers. Standalone Safetensors return
// an empty set; corrupt manifests throw. It does not establish compatibility.
struct PackageAssets {
    std::set<std::string> roles;
    std::optional<float> wanMoeBoundary;
};
[[nodiscard]] PackageAssets inspectIncludedPackageAssets(const std::filesystem::path &input);
[[nodiscard]] bool hasIncludedPackageAsset(const std::filesystem::path &input, const std::string &role);
}

// Paths are relative to the input root. Empty role/position means automatic.
// Position is "main" or "component"; only a model can be main.
struct ModelPackagingAssignment {
    std::filesystem::path relativePath;
    std::string role;
    std::string position;
};

// All reports use iild-model-package-report-v1. Errors throw std::runtime_error;
// scan reports compatibility failures in ready/errors without writing any files.
[[nodiscard]] std::string scanModelFolder(const std::filesystem::path &input,
    std::span<const std::string> excluded = {}, std::stop_token stop = {},
    const ModelPackagingObserver &observer = {});
// Explicit, root-relative files or model directories. Only the selected items
// are read; provider control sidecars and redirected paths are rejected.
[[nodiscard]] std::string scanModelSelection(const std::filesystem::path &root,
    std::span<const std::filesystem::path> selected, std::span<const std::string> excluded = {},
    std::stop_token stop = {}, const ModelPackagingObserver &observer = {});
[[nodiscard]] std::string createModelPackage(const std::filesystem::path &input,
    const std::filesystem::path &output, std::span<const std::string> excluded = {},
    std::stop_token stop = {}, const ModelPackagingObserver &observer = {});
[[nodiscard]] std::string createModelPackageFromSelection(const std::filesystem::path &root,
    std::span<const std::filesystem::path> selected, const std::filesystem::path &output,
    std::span<const std::string> excluded = {}, std::stop_token stop = {},
    const ModelPackagingObserver &observer = {});
[[nodiscard]] std::string scanModelFolderWithAssignments(const std::filesystem::path &input,
    std::span<const ModelPackagingAssignment> assignments, std::span<const std::string> excluded = {},
    std::stop_token stop = {}, const ModelPackagingObserver &observer = {});
[[nodiscard]] std::string scanModelSelectionWithAssignments(const std::filesystem::path &root,
    std::span<const std::filesystem::path> selected, std::span<const ModelPackagingAssignment> assignments,
    std::span<const std::string> excluded = {}, std::stop_token stop = {}, const ModelPackagingObserver &observer = {});
[[nodiscard]] std::string createModelPackageWithAssignments(const std::filesystem::path &input,
    const std::filesystem::path &output, std::span<const ModelPackagingAssignment> assignments,
    std::span<const std::string> excluded = {}, std::stop_token stop = {}, const ModelPackagingObserver &observer = {});
[[nodiscard]] std::string createModelPackageFromSelectionWithAssignments(const std::filesystem::path &root,
    std::span<const std::filesystem::path> selected, const std::filesystem::path &output,
    std::span<const ModelPackagingAssignment> assignments, std::span<const std::string> excluded = {},
    std::stop_token stop = {}, const ModelPackagingObserver &observer = {});
[[nodiscard]] std::string verifyModelPackage(const std::filesystem::path &input,
    std::stop_token stop = {}, const ModelPackagingObserver &observer = {});
// Reconstructs included original files and duplicate aliases byte for byte.
// Destination must not exist; source package is never changed.
[[nodiscard]] std::string extractModelPackage(const std::filesystem::path &input,
    const std::filesystem::path &output, std::stop_token stop = {},
    const ModelPackagingObserver &observer = {});

struct ModelCalibrationOptions {
    // Explicit numeric repair; this is not a promise of improved image quality.
    bool replaceNonfiniteWithZero = false;
};

// Scan F16, BF16, F32 and F64 values. Other tensor types are counted as
// unchecked. The file is never changed; this does not execute the model.
[[nodiscard]] std::string inspectModelWeights(const std::filesystem::path &input,
    std::stop_token stop = {}, const ModelPackagingObserver &observer = {});

// Create a separate standalone Safetensors file with explicit non-finite
// numeric repair. Packaged models must use calibrateModelPackage instead.
[[nodiscard]] std::string calibrateModelFile(const std::filesystem::path &input,
    const std::filesystem::path &output, std::stop_token stop = {},
    const ModelPackagingObserver &observer = {});

// Verify and reconstruct an existing package, re-evaluate component roles, and
// publish a separately verified package. Source component bytes are preserved
// unless the caller explicitly requests the documented numeric repair.
// Input and existing destination are never overwritten; temporary files are
// removed on completion, failure or cancellation.
[[nodiscard]] std::string calibrateModelPackage(const std::filesystem::path &input,
    const std::filesystem::path &output, std::stop_token stop = {},
    const ModelPackagingObserver &observer = {}, ModelCalibrationOptions options = {});

// Reconstructs an unambiguous Safetensors/GGUF component or indexed shard group
// without reading unrelated weights. Original index, tensor and source SHA-256
// values and format structure are verified before publication or cache reuse.
[[nodiscard]] std::filesystem::path materializeModelPackageAsset(
    const std::filesystem::path &input, const std::filesystem::path &cacheDirectory,
    const std::string &role, std::stop_token stop = {},
    const ModelPackagingObserver &observer = {});

// Selects an exact original relative file path or Safetensors shard-index path,
// including when several different components have the same role.
[[nodiscard]] std::filesystem::path materializeModelPackageAssetByPath(
    const std::filesystem::path &input, const std::filesystem::path &cacheDirectory,
    const std::filesystem::path &relativeSource, std::stop_token stop = {},
    const ModelPackagingObserver &observer = {});

}
