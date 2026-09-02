#pragma once

#include "ModelManifest/ModelManifest.hpp"

#include <json-c/json.h>

#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace iild::detail
{

using JsonObject = std::unique_ptr<json_object, decltype(&json_object_put)>;

[[noreturn]] void fail(ModelManifestErrorCode code, const std::string &message);

[[nodiscard]] std::string displayString(std::string_view value);
[[nodiscard]] std::filesystem::path canonicalModelRoot(
    const std::filesystem::path &modelRoot);
[[nodiscard]] JsonObject parseObject(
    const std::filesystem::path &path,
    const std::string &description);
[[nodiscard]] std::string requiredString(
    json_object *object,
    const char *name,
    const std::string &description);

void requireDirectory(const std::filesystem::path &path, const std::string &description);
void requireFile(const std::filesystem::path &path, const std::string &description);
void requireSupportedString(
    json_object *object,
    const char *name,
    std::string_view expected,
    const std::string &description);
void requireSupportedInteger(
    json_object *object,
    const char *name,
    std::int64_t expected,
    const std::string &description);
void requireSupportedBoolean(
    json_object *object,
    const char *name,
    bool expected,
    const std::string &description);
void requireSupportedNumber(
    json_object *object,
    const char *name,
    double expected,
    const std::string &description);
void requireOptionalSupportedIntegerOrNull(
    json_object *object,
    const char *name,
    std::int64_t expected,
    const std::string &description);
void requireOptionalSupportedIntegerArray(
    json_object *object,
    const char *name,
    std::initializer_list<std::int64_t> expected,
    const std::string &description);
void requireSupportedIntegerArray(
    json_object *object,
    const char *name,
    std::initializer_list<std::int64_t> expected,
    const std::string &description);
void requireSupportedStringArray(
    json_object *object,
    const char *name,
    std::initializer_list<std::string_view> expected,
    const std::string &description);
void requirePipelineComponent(
    json_object *modelIndex,
    const char *name,
    std::string_view expectedLibrary,
    std::string_view expectedClass);

[[nodiscard]] std::vector<std::filesystem::path> findSafetensorFiles(
    const std::filesystem::path &directory,
    const std::string &description,
    std::string_view canonicalBaseName);
[[nodiscard]] ModelComponentManifest loadClipTokenizer(
    const std::filesystem::path &root,
    std::string_view componentName);
[[nodiscard]] ModelComponentManifest loadClipTextEncoder(
    const std::filesystem::path &root,
    std::string_view componentName,
    std::int64_t hiddenSize,
    std::optional<std::int64_t> projectionDimension);

} // namespace iild::detail
