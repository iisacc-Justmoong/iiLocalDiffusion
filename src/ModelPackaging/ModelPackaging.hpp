#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <span>
#include <stop_token>
#include <string>
#include <vector>

namespace iild {

struct ModelPackagingProgress {
    std::string phase;
    std::string detail;
    std::uint64_t completedBytes = 0;
    std::uint64_t totalBytes = 0;
};
using ModelPackagingObserver = std::function<void(const ModelPackagingProgress &)>;

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

// Reconstructs the selected safetensors or GGUF component without reading
// unrelated weights. SHA-256 and format structure are verified before
// publishing or reusing the reconstructed file.
[[nodiscard]] std::filesystem::path materializeModelPackageAsset(
    const std::filesystem::path &input, const std::filesystem::path &cacheDirectory,
    const std::string &role, std::stop_token stop = {},
    const ModelPackagingObserver &observer = {});

// Selects the exact original relative path, including when legacy manifests
// incorrectly assigned several different components the same role.
[[nodiscard]] std::filesystem::path materializeModelPackageAssetByPath(
    const std::filesystem::path &input, const std::filesystem::path &cacheDirectory,
    const std::filesystem::path &relativeSource, std::stop_token stop = {},
    const ModelPackagingObserver &observer = {});

}
