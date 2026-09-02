#include "ModelManifest/ModelManifestParser.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <set>
#include <system_error>
#include <utility>

namespace
{

namespace fs = std::filesystem;

constexpr std::uintmax_t maximumConfigurationSize = 1024U * 1024U;

using JsonTokener = std::unique_ptr<json_tokener, decltype(&json_tokener_free)>;

struct SafetensorShardPosition
{
    std::size_t index;
    std::size_t count;
};

bool isRegularFile(const fs::path &path)
{
    std::error_code error;
    const bool result = fs::is_regular_file(fs::status(path, error));
    return !error && result;
}

std::string displayPath(const fs::path &path)
{
    return iild::detail::displayString(path.generic_string());
}

json_object *requiredMember(
    json_object *object,
    const char *name,
    const std::string &description)
{
    json_object *member = nullptr;
    if (!json_object_object_get_ex(object, name, &member) || member == nullptr)
    {
        iild::detail::fail(
            iild::ModelManifestErrorCode::invalidPackage,
            description + " is missing required member '" + name + "'");
    }
    return member;
}

std::string jsonStringValue(json_object *value)
{
    const auto *characters = json_object_get_string(value);
    const auto length = json_object_get_string_len(value);
    return std::string{characters, static_cast<std::size_t>(length)};
}

std::int64_t requiredInteger(
    json_object *object,
    const char *name,
    const std::string &description)
{
    json_object *member = requiredMember(object, name, description);
    if (!json_object_is_type(member, json_type_int))
    {
        iild::detail::fail(
            iild::ModelManifestErrorCode::invalidPackage,
            description + " member '" + name + "' must be an integer");
    }
    return json_object_get_int64(member);
}

double requiredNumber(
    json_object *object,
    const char *name,
    const std::string &description)
{
    json_object *member = requiredMember(object, name, description);
    if (!json_object_is_type(member, json_type_double) &&
        !json_object_is_type(member, json_type_int))
    {
        iild::detail::fail(
            iild::ModelManifestErrorCode::invalidPackage,
            description + " member '" + name + "' must be a number");
    }
    return json_object_get_double(member);
}

bool requiredBoolean(
    json_object *object,
    const char *name,
    const std::string &description)
{
    json_object *member = requiredMember(object, name, description);
    if (!json_object_is_type(member, json_type_boolean))
    {
        iild::detail::fail(
            iild::ModelManifestErrorCode::invalidPackage,
            description + " member '" + name + "' must be a boolean");
    }
    return json_object_get_boolean(member) != 0;
}

void requireNonEmptyFile(const fs::path &path, const std::string &description)
{
    iild::detail::requireFile(path, description);
    std::error_code sizeError;
    const auto size = fs::file_size(path, sizeError);
    if (sizeError || size == 0)
    {
        iild::detail::fail(
            iild::ModelManifestErrorCode::invalidPackage,
            description + " is empty or its size cannot be read: " + displayPath(path));
    }
}

std::optional<SafetensorShardPosition> parseSafetensorShardPosition(
    const std::string_view filename,
    const std::string_view canonicalBaseName)
{
    constexpr std::string_view suffix = ".safetensors";
    if (!filename.ends_with(suffix))
    {
        return std::nullopt;
    }

    const std::array prefixes{
        std::string{canonicalBaseName} + "-",
        std::string{canonicalBaseName} + ".fp16-"};
    std::string_view numbering;
    for (const auto &prefix : prefixes)
    {
        if (filename.starts_with(prefix))
        {
            numbering = filename.substr(
                prefix.size(),
                filename.size() - prefix.size() - suffix.size());
            break;
        }
    }

    constexpr std::string_view separator = "-of-";
    if (numbering.size() != 14 || numbering.substr(5, separator.size()) != separator)
    {
        return std::nullopt;
    }

    const auto parseFiveDigits = [](const std::string_view digits) -> std::optional<std::size_t> {
        std::size_t result = 0;
        for (const char character : digits)
        {
            if (character < '0' || character > '9')
            {
                return std::nullopt;
            }
            result = result * 10U + static_cast<std::size_t>(character - '0');
        }
        return result;
    };

    const auto index = parseFiveDigits(numbering.substr(0, 5));
    const auto count = parseFiveDigits(numbering.substr(9, 5));
    if (!index.has_value() || !count.has_value() || *index == 0 || *count == 0 || *index > *count)
    {
        return std::nullopt;
    }
    return SafetensorShardPosition{*index, *count};
}

std::vector<fs::path> indexedSafetensorFiles(
    const fs::path &directory,
    const fs::path &indexPath,
    const std::string &description,
    const std::string_view canonicalBaseName)
{
    const auto indexObject = iild::detail::parseObject(
        indexPath,
        description + " sharded weight index");
    json_object *weightMap = requiredMember(
        indexObject.get(),
        "weight_map",
        description + " sharded weight index");
    if (!json_object_is_type(weightMap, json_type_object))
    {
        iild::detail::fail(
            iild::ModelManifestErrorCode::invalidPackage,
            description + " sharded weight index member 'weight_map' must be an object");
    }
    if (json_object_object_length(weightMap) == 0)
    {
        iild::detail::fail(
            iild::ModelManifestErrorCode::invalidPackage,
            description + " sharded weight index member 'weight_map' must not be empty");
    }

    std::set<fs::path> referencedFiles;
    std::set<std::size_t> referencedShardIndices;
    std::optional<std::size_t> shardCount;
    json_object_object_foreach(weightMap, tensorName, shardValue)
    {
        static_cast<void>(tensorName);
        if (!json_object_is_type(shardValue, json_type_string))
        {
            iild::detail::fail(
                iild::ModelManifestErrorCode::invalidPackage,
                description + " sharded weight index values must be strings");
        }

        const fs::path shardName{jsonStringValue(shardValue)};
        if (shardName.empty() || shardName.is_absolute() || shardName.has_parent_path() ||
            shardName.filename() != shardName || shardName.extension() != ".safetensors")
        {
            iild::detail::fail(
                iild::ModelManifestErrorCode::invalidPackage,
                description + " sharded weight index contains an unsafe shard path: " +
                    iild::detail::displayString(shardName.generic_string()));
        }

        const auto position = parseSafetensorShardPosition(
            shardName.generic_string(), canonicalBaseName);
        if (!position.has_value())
        {
            iild::detail::fail(
                iild::ModelManifestErrorCode::invalidPackage,
                description + " sharded weight index contains a non-canonical shard name: " +
                    iild::detail::displayString(shardName.generic_string()));
        }
        if (shardCount.has_value() && *shardCount != position->count)
        {
            iild::detail::fail(
                iild::ModelManifestErrorCode::invalidPackage,
                description + " sharded weight index contains mismatched shard counts");
        }
        shardCount = position->count;
        referencedShardIndices.insert(position->index);
        referencedFiles.insert(directory / shardName);
    }

    if (!shardCount.has_value() || referencedShardIndices.size() != *shardCount)
    {
        iild::detail::fail(
            iild::ModelManifestErrorCode::invalidPackage,
            description + " sharded weight index does not reference every shard");
    }
    for (std::size_t index = 1; index <= *shardCount; ++index)
    {
        if (!referencedShardIndices.contains(index))
        {
            iild::detail::fail(
                iild::ModelManifestErrorCode::invalidPackage,
                description + " sharded weight index does not reference every shard");
        }
    }

    std::vector<fs::path> result;
    result.reserve(referencedFiles.size());
    for (const auto &path : referencedFiles)
    {
        requireNonEmptyFile(path, description + " referenced shard");
        result.push_back(path);
    }
    return result;
}

template<typename Value>
void requireArrayTypeAndLength(
    json_object *member,
    const char *name,
    const std::initializer_list<Value> expected,
    const std::string &description)
{
    if (!json_object_is_type(member, json_type_array) ||
        json_object_array_length(member) != expected.size())
    {
        iild::detail::fail(
            iild::ModelManifestErrorCode::invalidPackage,
            description + " member '" + name + "' must be an array with " +
                std::to_string(expected.size()) + " entries");
    }
}

} // namespace

namespace iild
{

ModelManifestError::ModelManifestError(
    const ModelManifestErrorCode code,
    std::string message)
    : std::runtime_error{std::move(message)}, code_{code}
{
}

ModelManifestErrorCode ModelManifestError::code() const noexcept
{
    return code_;
}

} // namespace iild

namespace iild::detail
{

[[noreturn]] void fail(const ModelManifestErrorCode code, const std::string &message)
{
    throw ModelManifestError{code, message};
}

std::string displayString(const std::string_view value)
{
    std::string result;
    result.reserve(value.size());
    constexpr char hexadecimal[] = "0123456789abcdef";
    for (const char rawCharacter : value)
    {
        const auto character = static_cast<unsigned char>(rawCharacter);
        if (character >= 0x20 && character <= 0x7e && character != '\\' && character != '\'')
        {
            result.push_back(static_cast<char>(character));
        }
        else if (character == '\\' || character == '\'')
        {
            result.push_back('\\');
            result.push_back(static_cast<char>(character));
        }
        else
        {
            result += "\\x";
            result.push_back(hexadecimal[character >> 4U]);
            result.push_back(hexadecimal[character & 0x0fU]);
        }
    }
    return result;
}

void requireDirectory(const fs::path &path, const std::string &description)
{
    std::error_code error;
    const auto status = fs::status(path, error);
    if (error || !fs::exists(status) || !fs::is_directory(status))
    {
        fail(
            ModelManifestErrorCode::invalidPackage,
            description + " is missing or is not a directory: " + displayPath(path));
    }
}

void requireFile(const fs::path &path, const std::string &description)
{
    if (!isRegularFile(path))
    {
        fail(
            ModelManifestErrorCode::invalidPackage,
            description + " is missing or is not a regular file: " + displayPath(path));
    }
}

JsonObject parseObject(const fs::path &path, const std::string &description)
{
    requireFile(path, description);

    std::error_code sizeError;
    const auto size = fs::file_size(path, sizeError);
    if (sizeError || size == 0 || size > maximumConfigurationSize)
    {
        fail(
            ModelManifestErrorCode::invalidPackage,
            description + " must be between 1 byte and 1 MiB: " + displayPath(path));
    }

    std::ifstream input{path, std::ios::binary};
    if (!input)
    {
        fail(
            ModelManifestErrorCode::invalidPackage,
            description + " is not readable: " + displayPath(path));
    }

    std::string contents(static_cast<std::size_t>(maximumConfigurationSize + 1U), '\0');
    input.read(contents.data(), static_cast<std::streamsize>(contents.size()));
    const auto bytesRead = input.gcount();
    if (input.bad())
    {
        fail(
            ModelManifestErrorCode::invalidPackage,
            description + " could not be read: " + displayPath(path));
    }
    if (bytesRead <= 0 || static_cast<std::uintmax_t>(bytesRead) > maximumConfigurationSize)
    {
        fail(
            ModelManifestErrorCode::invalidPackage,
            description + " must be between 1 byte and 1 MiB: " + displayPath(path));
    }
    contents.resize(static_cast<std::size_t>(bytesRead));

    JsonTokener tokener{json_tokener_new(), &json_tokener_free};
    if (!tokener)
    {
        throw std::bad_alloc{};
    }
    json_tokener_set_flags(tokener.get(), JSON_TOKENER_STRICT | JSON_TOKENER_VALIDATE_UTF8);

    json_object *parsed = json_tokener_parse_ex(
        tokener.get(),
        contents.data(),
        static_cast<int>(contents.size()));
    const auto parseError = json_tokener_get_error(tokener.get());
    const auto parseEnd = json_tokener_get_parse_end(tokener.get());

    JsonObject object{parsed, &json_object_put};
    const bool parseEndIsValid = parseEnd <= contents.size();
    const bool hasOnlyTrailingWhitespace = parseEndIsValid && std::all_of(
        contents.begin() + static_cast<std::ptrdiff_t>(parseEnd),
        contents.end(),
        [](const unsigned char character) { return std::isspace(character) != 0; });

    if (parseError != json_tokener_success || !object || !hasOnlyTrailingWhitespace)
    {
        fail(
            ModelManifestErrorCode::invalidPackage,
            description + " is not valid JSON: " + displayPath(path));
    }
    if (!json_object_is_type(object.get(), json_type_object))
    {
        fail(
            ModelManifestErrorCode::invalidPackage,
            description + " must contain a JSON object: " + displayPath(path));
    }
    return object;
}

std::string requiredString(
    json_object *object,
    const char *name,
    const std::string &description)
{
    json_object *member = requiredMember(object, name, description);
    if (!json_object_is_type(member, json_type_string))
    {
        fail(
            ModelManifestErrorCode::invalidPackage,
            description + " member '" + name + "' must be a string");
    }
    return jsonStringValue(member);
}

void requireSupportedString(
    json_object *object,
    const char *name,
    const std::string_view expected,
    const std::string &description)
{
    const auto actual = requiredString(object, name, description);
    if (actual != expected)
    {
        fail(
            ModelManifestErrorCode::unsupportedModel,
            description + " member '" + name + "' must be '" +
                std::string{expected} + "', found '" + displayString(actual) + "'");
    }
}

void requireSupportedInteger(
    json_object *object,
    const char *name,
    const std::int64_t expected,
    const std::string &description)
{
    const auto actual = requiredInteger(object, name, description);
    if (actual != expected)
    {
        fail(
            ModelManifestErrorCode::unsupportedModel,
            description + " member '" + name + "' must be " +
                std::to_string(expected) + ", found " + std::to_string(actual));
    }
}

void requireSupportedBoolean(
    json_object *object,
    const char *name,
    const bool expected,
    const std::string &description)
{
    const auto actual = requiredBoolean(object, name, description);
    if (actual != expected)
    {
        fail(
            ModelManifestErrorCode::unsupportedModel,
            description + " member '" + name + "' has an unsupported value");
    }
}

void requireSupportedNumber(
    json_object *object,
    const char *name,
    const double expected,
    const std::string &description)
{
    const auto actual = requiredNumber(object, name, description);
    if (std::abs(actual - expected) > 1.0e-12)
    {
        fail(
            ModelManifestErrorCode::unsupportedModel,
            description + " member '" + name + "' must be " +
                std::to_string(expected) + ", found " + std::to_string(actual));
    }
}

void requireOptionalSupportedIntegerOrNull(
    json_object *object,
    const char *name,
    const std::int64_t expected,
    const std::string &description)
{
    json_object *member = nullptr;
    if (!json_object_object_get_ex(object, name, &member) || member == nullptr ||
        json_object_is_type(member, json_type_null))
    {
        return;
    }
    if (!json_object_is_type(member, json_type_int))
    {
        fail(
            ModelManifestErrorCode::invalidPackage,
            description + " member '" + name + "' must be an integer or null");
    }
    const auto actual = json_object_get_int64(member);
    if (actual != expected)
    {
        fail(
            ModelManifestErrorCode::unsupportedModel,
            description + " member '" + name + "' must be " +
                std::to_string(expected) + " when specified, found " + std::to_string(actual));
    }
}

void requireOptionalSupportedIntegerArray(
    json_object *object,
    const char *name,
    const std::initializer_list<std::int64_t> expected,
    const std::string &description)
{
    json_object *member = nullptr;
    if (!json_object_object_get_ex(object, name, &member) || member == nullptr)
    {
        return;
    }
    requireArrayTypeAndLength(member, name, expected, description);

    std::size_t index = 0;
    for (const auto expectedValue : expected)
    {
        json_object *value = json_object_array_get_idx(member, index);
        if (!json_object_is_type(value, json_type_int))
        {
            fail(
                ModelManifestErrorCode::invalidPackage,
                description + " member '" + name + "' entries must be integers");
        }
        if (json_object_get_int64(value) != expectedValue)
        {
            fail(
                ModelManifestErrorCode::unsupportedModel,
                description + " member '" + name + "' must contain the canonical integer array");
        }
        ++index;
    }
}

void requireSupportedIntegerArray(
    json_object *object,
    const char *name,
    const std::initializer_list<std::int64_t> expected,
    const std::string &description)
{
    json_object *member = requiredMember(object, name, description);
    requireArrayTypeAndLength(member, name, expected, description);

    std::size_t index = 0;
    for (const auto expectedValue : expected)
    {
        json_object *value = json_object_array_get_idx(member, index);
        if (!json_object_is_type(value, json_type_int))
        {
            fail(
                ModelManifestErrorCode::invalidPackage,
                description + " member '" + name + "' entries must be integers");
        }
        if (json_object_get_int64(value) != expectedValue)
        {
            fail(
                ModelManifestErrorCode::unsupportedModel,
                description + " member '" + name + "' must contain the canonical integer array");
        }
        ++index;
    }
}

void requireSupportedStringArray(
    json_object *object,
    const char *name,
    const std::initializer_list<std::string_view> expected,
    const std::string &description)
{
    json_object *member = requiredMember(object, name, description);
    requireArrayTypeAndLength(member, name, expected, description);

    std::size_t index = 0;
    for (const auto expectedValue : expected)
    {
        json_object *value = json_object_array_get_idx(member, index);
        if (!json_object_is_type(value, json_type_string))
        {
            fail(
                ModelManifestErrorCode::invalidPackage,
                description + " member '" + name + "' entries must be strings");
        }
        const auto actual = jsonStringValue(value);
        if (actual != expectedValue)
        {
            fail(
                ModelManifestErrorCode::unsupportedModel,
                description + " member '" + name + "' must contain the canonical string array");
        }
        ++index;
    }
}

void requirePipelineComponent(
    json_object *modelIndex,
    const char *name,
    const std::string_view expectedLibrary,
    const std::string_view expectedClass)
{
    json_object *component = requiredMember(modelIndex, name, "model_index.json");
    if (!json_object_is_type(component, json_type_array) ||
        json_object_array_length(component) != 2)
    {
        fail(
            ModelManifestErrorCode::invalidPackage,
            "model_index.json member '" + std::string{name} +
                "' must be a [library, class] array");
    }

    json_object *library = json_object_array_get_idx(component, 0);
    json_object *className = json_object_array_get_idx(component, 1);
    if (!json_object_is_type(library, json_type_string) ||
        !json_object_is_type(className, json_type_string))
    {
        fail(
            ModelManifestErrorCode::invalidPackage,
            "model_index.json member '" + std::string{name} +
                "' must contain two strings");
    }

    const std::string actualLibrary = jsonStringValue(library);
    const std::string actualClass = jsonStringValue(className);
    if (actualLibrary != expectedLibrary || actualClass != expectedClass)
    {
        fail(
            ModelManifestErrorCode::unsupportedModel,
            "model_index.json member '" + std::string{name} + "' must be [\"" +
                std::string{expectedLibrary} + "\", \"" + std::string{expectedClass} +
                "\"]");
    }
}

std::vector<fs::path> findSafetensorFiles(
    const fs::path &directory,
    const std::string &description,
    const std::string_view canonicalBaseName)
{
    const std::string base{canonicalBaseName};
    const std::array singleNames{
        base + ".safetensors",
        base + ".fp16.safetensors"};
    const std::array indexNames{
        base + ".safetensors.index.json",
        base + ".fp16.safetensors.index.json",
        base + ".safetensors.index.fp16.json"};

    std::set<fs::path> artifacts;
    bool foundIndex = false;
    for (const auto &indexName : indexNames)
    {
        const auto indexPath = directory / indexName;
        if (!isRegularFile(indexPath))
        {
            continue;
        }
        foundIndex = true;
        for (const auto &path : indexedSafetensorFiles(
                 directory,
                 indexPath,
                 description,
                 canonicalBaseName))
        {
            artifacts.insert(path);
        }
    }

    for (const auto &singleName : singleNames)
    {
        const auto singlePath = directory / singleName;
        if (isRegularFile(singlePath))
        {
            requireNonEmptyFile(singlePath, description + " weight file");
            artifacts.insert(singlePath);
        }
    }

    if (!foundIndex && artifacts.empty())
    {
        fail(
            ModelManifestErrorCode::invalidPackage,
            description + " weight file is missing; expected canonical single-file weights or " +
                "a sharded safetensors index in " + displayPath(directory));
    }
    return {artifacts.begin(), artifacts.end()};
}

fs::path canonicalModelRoot(const fs::path &modelRoot)
{
    if (modelRoot.empty())
    {
        fail(ModelManifestErrorCode::inputUnavailable, "model root is empty");
    }

    std::error_code statusError;
    const auto rootStatus = fs::status(modelRoot, statusError);
    if (statusError || !fs::exists(rootStatus) || !fs::is_directory(rootStatus))
    {
        fail(
            ModelManifestErrorCode::inputUnavailable,
            "model root is unavailable or is not a directory: " + displayPath(modelRoot));
    }

    std::error_code canonicalError;
    const auto root = fs::canonical(modelRoot, canonicalError);
    if (canonicalError)
    {
        fail(
            ModelManifestErrorCode::inputUnavailable,
            "model root cannot be resolved: " + displayPath(modelRoot));
    }
    return root;
}

ModelComponentManifest loadClipTokenizer(
    const fs::path &root,
    const std::string_view componentName)
{
    const std::string component{componentName};
    const auto directory = root / component;
    const auto configuration = directory / "tokenizer_config.json";
    const auto vocabulary = directory / "vocab.json";
    const auto merges = directory / "merges.txt";

    requireDirectory(directory, component + " component");
    const auto config = parseObject(configuration, component + "/tokenizer_config.json");
    requireSupportedInteger(config.get(), "model_max_length", 77, component + " configuration");
    requireSupportedString(
        config.get(), "tokenizer_class", "CLIPTokenizer", component + " configuration");
    requireFile(vocabulary, component + " vocabulary");
    requireFile(merges, component + " merges");

    return {directory, configuration, {vocabulary, merges}};
}

ModelComponentManifest loadClipTextEncoder(
    const fs::path &root,
    const std::string_view componentName,
    const std::int64_t hiddenSize,
    const std::optional<std::int64_t> projectionDimension)
{
    const std::string component{componentName};
    const auto directory = root / component;
    const auto configuration = directory / "config.json";

    requireDirectory(directory, component + " component");
    const auto config = parseObject(configuration, component + "/config.json");
    requireSupportedInteger(config.get(), "hidden_size", hiddenSize, component + " configuration");
    requireSupportedInteger(
        config.get(), "max_position_embeddings", 77, component + " configuration");
    if (projectionDimension.has_value())
    {
        requireSupportedInteger(
            config.get(), "projection_dim", *projectionDimension, component + " configuration");
    }

    return {
        directory,
        configuration,
        findSafetensorFiles(directory, component, "model")};
}

} // namespace iild::detail
