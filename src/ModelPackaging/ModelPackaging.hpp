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

// All reports use iild-model-package-report-v1. Errors throw std::runtime_error;
// scan reports compatibility failures in ready/errors without writing any files.
[[nodiscard]] std::string scanModelFolder(const std::filesystem::path &input,
    std::span<const std::string> excluded = {}, std::stop_token stop = {},
    const ModelPackagingObserver &observer = {});
[[nodiscard]] std::string createModelPackage(const std::filesystem::path &input,
    const std::filesystem::path &output, std::span<const std::string> excluded = {},
    std::stop_token stop = {}, const ModelPackagingObserver &observer = {});
[[nodiscard]] std::string verifyModelPackage(const std::filesystem::path &input,
    std::stop_token stop = {}, const ModelPackagingObserver &observer = {});
// Reconstructs included original files and duplicate aliases byte for byte.
// Destination must not exist; source package is never changed.
[[nodiscard]] std::string extractModelPackage(const std::filesystem::path &input,
    const std::filesystem::path &output, std::stop_token stop = {},
    const ModelPackagingObserver &observer = {});

}
