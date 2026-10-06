#include "ModelPackaging/ModelPackaging.hpp"

#include <json-c/json.h>
#include <openssl/evp.h>
#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>
#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#endif

namespace iild {
namespace {
namespace fs = std::filesystem;
using U64 = std::uint64_t;
constexpr U64 headerLimit = 64 * 1024 * 1024;
constexpr std::size_t chunkSize = 8 * 1024 * 1024;
constexpr const char *schema = "iild-safetensors-package-v1";
constexpr const char *reportSchema = "iild-model-package-report-v1";
constexpr const char *emptyDigest = "0000000000000000000000000000000000000000000000000000000000000000";

[[noreturn]] void fail(const std::string &message) { throw std::runtime_error(message); }
void check(std::stop_token stop) { if (stop.stop_requested()) fail("Packaging cancelled"); }
void progress(const ModelPackagingObserver &observer, std::stop_token stop,
              std::string phase, std::string detail, U64 done = 0, U64 total = 0) {
    check(stop);
    if (observer) observer({std::move(phase), std::move(detail), done, total});
    check(stop);
}
U64 add(U64 a, U64 b) {
    const auto maximum = static_cast<U64>(std::numeric_limits<std::int64_t>::max());
    if (b > maximum || a > maximum - b)
        fail("Model size exceeds supported 64-bit file offsets");
    return a + b;
}
std::string lower(std::string value) {
    for (auto &c : value) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return value;
}
bool has(const std::string &value, const std::string &needle) { return value.find(needle) != std::string::npos; }
bool safeRelative(const fs::path &value) {
    if (value.empty() || value.is_absolute() || value.has_root_name()) return false;
    for (const auto &part : value)
        if (part == ".." || part == "." || part.empty() || has(part.string(), "\\") || has(part.string(), ":"))
            return false;
    return value.generic_string().find('\0') == std::string::npos;
}

class J {
    std::shared_ptr<json_object> object_;
public:
    explicit J(json_object *object) : object_(object, json_object_put) {
        if (!object) fail("Missing JSON value");
    }
    static J object() { return J(json_object_new_object()); }
    static J array() { return J(json_object_new_array()); }
    json_object *get() const { return object_.get(); }
    bool contains(const std::string &key) const {
        json_object *child = nullptr;
        return json_object_object_get_ex(get(), key.c_str(), &child) && child;
    }
    J at(const std::string &key) const {
        json_object *child = nullptr;
        if (!json_object_object_get_ex(get(), key.c_str(), &child) || !child) fail("Missing JSON member: " + key);
        return J(json_object_get(child));
    }
    J at(std::size_t index) const {
        auto *child = json_object_array_get_idx(get(), index);
        if (!child) fail("Missing JSON array element");
        return J(json_object_get(child));
    }
    bool is(json_type type) const { return json_object_is_type(get(), type); }
    std::size_t size() const { return json_object_array_length(get()); }
    std::string str() const {
        if (!is(json_type_string)) fail("Expected a JSON string");
        return {json_object_get_string(get()), static_cast<std::size_t>(json_object_get_string_len(get()))};
    }
    U64 number() const {
        if (!is(json_type_int) || json_object_get_int64(get()) < 0) fail("Expected a nonnegative 64-bit integer");
        return static_cast<U64>(json_object_get_int64(get()));
    }
    bool boolean() const {
        if (!is(json_type_boolean)) fail("Expected a JSON boolean");
        return json_object_get_boolean(get()) != 0;
    }
    std::string dump() const { return json_object_to_json_string_ext(get(), JSON_C_TO_STRING_PLAIN | JSON_C_TO_STRING_NOSLASHESCAPE); }
    void set(const std::string &key, const J &value) { json_object_object_add(get(), key.c_str(), json_object_get(value.get())); }
    void set(const std::string &key, const std::string &value) {
        json_object_object_add(get(), key.c_str(), json_object_new_string_len(value.data(), static_cast<int>(value.size())));
    }
    void set(const std::string &key, const char *value) { set(key, std::string(value)); }
    void set(const std::string &key, U64 value) { json_object_object_add(get(), key.c_str(), json_object_new_uint64(value)); }
    void set(const std::string &key, bool value) { json_object_object_add(get(), key.c_str(), json_object_new_boolean(value)); }
    void push(const J &value) { json_object_array_add(get(), json_object_get(value.get())); }
    void push(const std::string &value) { json_object_array_add(get(), json_object_new_string_len(value.data(), static_cast<int>(value.size()))); }
    void push(U64 value) { json_object_array_add(get(), json_object_new_uint64(value)); }
    std::vector<std::pair<std::string, J>> members() const {
        if (!is(json_type_object)) fail("Expected a JSON object");
        std::vector<std::pair<std::string, J>> result;
        json_object_object_foreach(get(), key, value) {
            if (!value) fail("Null JSON member: " + std::string(key));
            result.emplace_back(key, J(json_object_get(value)));
        }
        return result;
    }
};

// json-c overwrites duplicate object keys; reject them before accepting headers.
void rejectDuplicateKeys(const std::string &text) {
    struct Scope { bool object; std::set<std::string> keys; };
    std::vector<Scope> scopes;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        if (c == '{' || c == '[') scopes.push_back({c == '{', {}});
        else if (c == '}' || c == ']') { if (!scopes.empty()) scopes.pop_back(); }
        else if (c == '"') {
            const auto begin = i++;
            while (i < text.size() && text[i] != '"') { if (text[i] == '\\') ++i; ++i; }
            auto next = i + 1;
            while (next < text.size() && std::isspace(static_cast<unsigned char>(text[next]))) ++next;
            if (next < text.size() && text[next] == ':' && !scopes.empty() && scopes.back().object) {
                const auto raw = text.substr(begin, i - begin + 1);
                auto *key = json_tokener_parse(raw.c_str());
                if (!key) fail("Invalid JSON object key");
                const std::string decoded(json_object_get_string(key), static_cast<std::size_t>(json_object_get_string_len(key)));
                json_object_put(key);
                if (!scopes.back().keys.insert(decoded).second) fail("Duplicate JSON key: " + decoded);
            }
        }
    }
}
J parse(const std::string &text) {
    if (text.size() > headerLimit) fail("JSON exceeds the 64 MiB metadata limit");
    rejectDuplicateKeys(text);
    auto tokener = std::unique_ptr<json_tokener, decltype(&json_tokener_free)>(json_tokener_new(), json_tokener_free);
    json_tokener_set_flags(tokener.get(), JSON_TOKENER_STRICT);
    auto *value = json_tokener_parse_ex(tokener.get(), text.data(), static_cast<int>(text.size()));
    const auto error = json_tokener_get_error(tokener.get());
    auto end = json_tokener_get_parse_end(tokener.get());
    while (end < text.size() && std::isspace(static_cast<unsigned char>(text[end]))) ++end;
    if (error != json_tokener_success || end != text.size() || !value) {
        if (value) json_object_put(value);
        fail("Invalid or incomplete JSON metadata");
    }
    return J(value);
}
std::string readSmall(const fs::path &path) {
    const auto size = fs::file_size(path);
    if (size > headerLimit) fail("Configuration exceeds the metadata limit: " + path.filename().string());
    std::ifstream file(path, std::ios::binary);
    std::string data(static_cast<std::size_t>(size), '\0');
    if (!file.read(data.data(), static_cast<std::streamsize>(size))) fail("Could not read " + path.string());
    return data;
}
U64 readLE(std::istream &file, unsigned count) {
    std::array<unsigned char, 8> bytes{};
    if (!file.read(reinterpret_cast<char *>(bytes.data()), count)) fail("Truncated model file");
    U64 value = 0;
    for (unsigned i = 0; i < count; ++i) value |= static_cast<U64>(bytes[i]) << (i * 8);
    return value;
}
std::array<char, 8> sizeBytes(U64 value) {
    std::array<char, 8> bytes{};
    for (unsigned i = 0; i < 8; ++i) bytes[i] = static_cast<char>((value >> (i * 8)) & 255);
    return bytes;
}
class Hash {
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context_{EVP_MD_CTX_new(), EVP_MD_CTX_free};
public:
    Hash() { if (!context_ || EVP_DigestInit_ex(context_.get(), EVP_sha256(), nullptr) != 1) fail("SHA-256 initialization failed"); }
    void update(const char *data, std::size_t size) {
        if (EVP_DigestUpdate(context_.get(), data, size) != 1) fail("SHA-256 update failed");
    }
    void update(const std::string &value) { update(value.data(), value.size()); }
    std::string finish() {
        unsigned char bytes[EVP_MAX_MD_SIZE]; unsigned length = 0;
        if (EVP_DigestFinal_ex(context_.get(), bytes, &length) != 1 || length != 32) fail("SHA-256 finalization failed");
        constexpr char hex[] = "0123456789abcdef";
        std::string result; result.reserve(64);
        for (unsigned i = 0; i < length; ++i) { result += hex[bytes[i] >> 4]; result += hex[bytes[i] & 15]; }
        return result;
    }
};
std::string hashRange(const fs::path &path, U64 offset, U64 size, std::stop_token stop,
                      const ModelPackagingObserver &observer, const std::string &detail, Hash *whole = nullptr) {
    std::ifstream file(path, std::ios::binary);
    file.seekg(static_cast<std::streamoff>(offset));
    std::vector<char> buffer(chunkSize); Hash hash;
    U64 done = 0;
    while (done < size) {
        progress(observer, stop, "deduplicate", detail, done, size);
        const auto count = static_cast<std::size_t>(std::min<U64>(size - done, buffer.size()));
        if (!file.read(buffer.data(), static_cast<std::streamsize>(count))) fail("Source is truncated or unreadable: " + path.string());
        hash.update(buffer.data(), count); if (whole) whole->update(buffer.data(), count);
        done += count;
    }
    check(stop);
    return hash.finish();
}

struct Tensor {
    std::string key, dtype, outputKey, digest = emptyDigest;
    std::vector<U64> shape;
    U64 begin = 0, end = 0;
};
struct Source {
    fs::path logical, actual;
    std::string relative, role, format, status = "included", reason, group, header, digest = emptyDigest;
    U64 size = 0, base = 0;
    fs::file_time_type time;
    J metadata = J::object();
    std::vector<Tensor> tensors;
    bool optional = false;
};
unsigned dtypeBits(const std::string &dtype) {
    static const std::map<std::string, unsigned> bits{
        {"BOOL",8},{"U8",8},{"I8",8},{"I16",16},{"U16",16},{"I32",32},{"U32",32},
        {"I64",64},{"U64",64},{"F16",16},{"BF16",16},{"F32",32},{"F64",64},
        {"F8_E4M3",8},{"F8_E5M2",8},{"F8_E4M3FN",8},{"F8_E4M3FNUZ",8},{"F8_E5M2FNUZ",8},{"F8_E8M0",8},
        {"F4",4},{"F6_E2M3",6},{"F6_E3M2",6}
    };
    const auto found = bits.find(dtype);
    if (found == bits.end()) fail("Unsupported Safetensors dtype: " + dtype);
    return found->second;
}
void readSafetensors(Source &source) {
    std::ifstream file(source.actual, std::ios::binary);
    const auto size = readLE(file, 8);
    if (!size || size > headerLimit || size > source.size || size > source.size - std::min<U64>(source.size, 8))
        fail("Invalid Safetensors header length");
    source.header.resize(static_cast<std::size_t>(size));
    if (!file.read(source.header.data(), static_cast<std::streamsize>(size))) fail("Truncated Safetensors header");
    if (source.header.front() != '{') fail("Safetensors header must begin with an object");
    const auto header = parse(source.header);
    source.base = 8 + size;
    if (header.contains("__metadata__")) {
        source.metadata = header.at("__metadata__");
        for (const auto &[key, value] : source.metadata.members()) { (void)key; (void)value.str(); }
    }
    for (const auto &[key, value] : header.members()) {
        if (key == "__metadata__") continue;
        if (key.empty() || key.find('\0') != std::string::npos) fail("Invalid tensor name");
        Tensor tensor; tensor.key = key; tensor.dtype = value.at("dtype").str();
        const auto offsets = value.at("data_offsets"), shape = value.at("shape");
        if (!offsets.is(json_type_array) || offsets.size() != 2 || !shape.is(json_type_array))
            fail("Invalid tensor shape or data offsets");
        tensor.begin = offsets.at(0).number(); tensor.end = offsets.at(1).number();
        U64 count = 1;
        for (std::size_t i = 0; i < shape.size(); ++i) {
            const auto dimension = shape.at(i).number();
            if (dimension && count > std::numeric_limits<U64>::max() / dimension) fail("Tensor shape overflows");
            count *= dimension; tensor.shape.push_back(dimension);
        }
        const auto bits = dtypeBits(tensor.dtype);
        if (count > (std::numeric_limits<U64>::max() - 7) / bits) fail("Tensor storage size overflows");
        const auto bytes = (count * bits + 7) / 8;
        if (tensor.end < tensor.begin || tensor.end - tensor.begin != bytes ||
            tensor.end > source.size - source.base) fail("Tensor shape does not match the file data range: " + key);
        source.tensors.push_back(std::move(tensor));
    }
    if (source.tensors.empty()) fail("Safetensors contains no model tensors");
    std::sort(source.tensors.begin(), source.tensors.end(), [](const auto &a, const auto &b) {
        return std::tie(a.begin,a.end,a.key) < std::tie(b.begin,b.end,b.key);
    });
    U64 end = 0;
    for (const auto &tensor : source.tensors) {
        if (tensor.begin == tensor.end) continue;
        if (tensor.begin != end) fail("Safetensors ranges overlap or contain a gap");
        end = tensor.end;
    }
    if (end != source.size - source.base) fail("Safetensors does not cover the complete file");
}
std::string ggufString(std::istream &file) {
    const auto length = readLE(file, 8);
    if (length > headerLimit) fail("GGUF metadata string exceeds the limit");
    std::string text(static_cast<std::size_t>(length), '\0');
    if (!file.read(text.data(), static_cast<std::streamsize>(length))) fail("Truncated GGUF string");
    return text;
}
void skipGGUF(std::istream &file, unsigned type, unsigned depth = 0) {
    if (depth > 8) fail("GGUF metadata nesting is too deep");
    if (file.tellg() < 0 || static_cast<U64>(file.tellg()) > headerLimit) fail("GGUF metadata exceeds the header limit");
    if (type == 8) { (void)ggufString(file); return; }
    if (type == 9) {
        const auto element = static_cast<unsigned>(readLE(file, 4)); const auto count = readLE(file, 8);
        if (count > 10'000'000) fail("GGUF array exceeds the limit");
        for (U64 i = 0; i < count; ++i) skipGGUF(file, element, depth + 1);
        return;
    }
    static constexpr std::array<unsigned, 13> sizes{1,1,2,2,4,4,4,1,0,0,8,8,8};
    if (type >= sizes.size() || sizes[type] == 0) fail("Invalid GGUF metadata type");
    (void)readLE(file, sizes[type]);
}
void readGGUF(Source &source) {
    std::ifstream file(source.actual, std::ios::binary);
    if (readLE(file, 4) != 0x46554747) fail("Invalid GGUF magic");
    const auto version = readLE(file, 4);
    if (version != 2 && version != 3) fail("Unsupported GGUF version");
    const auto tensors = readLE(file, 8), fields = readLE(file, 8);
    if (!tensors || tensors > 1'000'000 || fields > 1'000'000) fail("Invalid GGUF tensor or metadata count");
    U64 alignment = 32;
    for (U64 i = 0; i < fields; ++i) {
        const auto key = ggufString(file); const auto type = static_cast<unsigned>(readLE(file, 4));
        if (type == 8) source.metadata.set(key, ggufString(file));
        else if (key == "general.alignment" && type == 4) alignment = readLE(file, 4);
        else skipGGUF(file, type);
    }
    if (!alignment || alignment > 4096 || (alignment & (alignment - 1))) fail("Invalid GGUF alignment");
    // Sizes follow the GGML on-disk block definitions, independently of inference.
    static const std::map<U64,std::pair<U64,U64>> encoding{
        {0,{1,4}},{1,{1,2}},{2,{32,18}},{3,{32,20}},{6,{32,22}},{7,{32,24}},
        {8,{32,34}},{9,{32,36}},{10,{256,84}},{11,{256,110}},{12,{256,144}},
        {13,{256,176}},{14,{256,210}},{15,{256,292}},{16,{256,66}},
        {17,{256,74}},{18,{256,98}},{19,{256,50}},{20,{32,18}},
        {21,{256,110}},{22,{256,82}},{23,{256,136}},{24,{1,1}},{25,{1,2}},
        {26,{1,4}},{27,{1,8}},{28,{1,8}},{29,{256,56}},{30,{1,2}},
        {34,{256,54}},{35,{256,66}},{39,{32,17}},{40,{64,36}},{41,{128,18}},
        {42,{64,18}},{43,{1,1}},{44,{1,1}}
    };
    std::set<std::string> names; std::vector<std::pair<U64,U64>> ranges;
    for (U64 i = 0; i < tensors; ++i) {
        if (file.tellg() < 0 || static_cast<U64>(file.tellg()) > headerLimit) fail("GGUF header exceeds the metadata limit");
        const auto name = ggufString(file);
        if (name.empty() || !names.insert(name).second) fail("Invalid or duplicate GGUF tensor name");
        const auto dimensions = readLE(file, 4);
        if (!dimensions || dimensions > 4) fail("Invalid GGUF tensor dimensions");
        U64 count=1,first=0;
        for (U64 d = 0; d < dimensions; ++d) {
            const auto size=readLE(file,8);
            if (!size || count>std::numeric_limits<U64>::max()/size) fail("Invalid GGUF tensor shape");
            if (!d) first=size;
            count*=size;
        }
        const auto type=readLE(file,4),offset=readLE(file,8);
        if (!encoding.contains(type)) fail("Unsupported GGUF tensor encoding: "+std::to_string(type));
        const auto [block,bytes]=encoding.at(type);
        if (first%block || count/block>std::numeric_limits<U64>::max()/bytes) fail("Invalid GGUF quantization block shape");
        ranges.emplace_back(offset,add(offset,count/block*bytes));
    }
    const auto position = file.tellg();
    if (position < 0) fail("Invalid GGUF header");
    const auto base = (static_cast<U64>(position) + alignment - 1) & ~(alignment - 1);
    if (base >= source.size) fail("GGUF has no tensor data");
    std::sort(ranges.begin(),ranges.end());
    U64 end=0;
    for (const auto &[begin,next] : ranges) {
        if (begin<end || begin%alignment || next>source.size-base) fail("Invalid or truncated GGUF tensor range");
        end=next;
    }
    source.metadata.set("gguf.version", version);
}
bool prefix(const Source &source, const std::string &value) {
    return std::any_of(source.tensors.begin(), source.tensors.end(), [&](const auto &tensor) { return tensor.key.starts_with(value); });
}
std::string inferRole(const Source &source) {
    const auto path = lower(source.relative), config = source.metadata.contains("config") ? lower(source.metadata.at("config").str()) : "";
    if (has(path,"lora") || prefix(source,"lora_") ||
        std::any_of(source.tensors.begin(),source.tensors.end(),[](const auto &t){return has(t.key,".lora_A.") || has(t.key,".lora_down.");})) return "lora";
    if (has(path,"upscaler") || has(config,"latentupsampler")) return "upscaler";
    if (prefix(source,"model.diffusion_model.") || prefix(source,"diffusion_model.") || has(path,"transformer/") || has(path,"unet/")) return "model";
    if (has(path,"audio_vae") || prefix(source,"audio_vae.") || prefix(source,"vocoder.")) return "audio_vae";
    if (has(path,"projection") || prefix(source,"text_embedding_projection.")) return "projections";
    if (has(path,"vae") || has(config,"autoencoder")) return "video_vae";
    if (source.format == "gguf" || has(path,"text_encoder") || prefix(source,"text_model.")) return "text_encoder";
    return source.format == "safetensors" ? "model" : "other";
}
fs::path resolvedSource(const fs::path &path) {
#ifdef _WIN32
    const auto handle = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
        OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    if (handle == INVALID_HANDLE_VALUE) fail("Cannot resolve model source");
    struct Close { HANDLE value; ~Close() { CloseHandle(value); } } close{handle};
    const auto length = GetFinalPathNameByHandleW(handle, nullptr, 0, FILE_NAME_NORMALIZED);
    if (!length) fail("Cannot resolve model source name");
    std::wstring name(length, L'\0');
    const auto written = GetFinalPathNameByHandleW(handle, name.data(), length, FILE_NAME_NORMALIZED);
    if (!written || written >= length) fail("Cannot resolve model source name");
    name.resize(written);
    return fs::path(name);
#else
    return fs::canonical(path);
#endif
}
void stamp(const Source &source) {
    const auto name = source.relative.empty() ? source.logical.filename().string() : source.relative;
    if (resolvedSource(source.logical) != source.actual) fail("Source changed during packaging (identity): " + name);
    if (fs::file_size(source.actual) != source.size) fail("Source changed during packaging (size): " + name);
    if (fs::last_write_time(source.actual) != source.time) fail("Source changed during packaging (modified time): " + name);
}
struct Entry { std::size_t source, tensor; };
struct Plan {
    fs::path input;
    std::vector<Source> files;
    std::vector<Entry> entries;
    std::vector<std::string> errors;
    std::vector<std::size_t> primary;
    std::string family = "Model", name;
    bool ltx = false;
    U64 payload = 0;
};
void blob(Source &source) {
    Tensor tensor; tensor.key = ""; tensor.dtype = "U8"; tensor.shape = {source.size}; tensor.end = source.size;
    source.tensors.push_back(std::move(tensor));
}
std::string roleKey(const Source &file, const Tensor &tensor) {
    if (file.role == "video_vae") return tensor.key.starts_with("vae.") ? tensor.key : "vae." + tensor.key;
    if (file.role == "audio_vae") return tensor.key.starts_with("audio_vae.") || tensor.key.starts_with("vocoder.")
        ? tensor.key : "audio_vae." + tensor.key;
    if (file.role == "projections") return tensor.key.starts_with("text_embedding_projection.")
        ? tensor.key : "text_embedding_projection." + tensor.key;
    return tensor.key;
}

Plan scan(const fs::path &input, std::span<const std::string> excluded, std::stop_token stop,
          const ModelPackagingObserver &observer) {
    if (!fs::is_directory(input)) fail("Choose an existing model folder");
    Plan plan; plan.input = fs::absolute(input).lexically_normal(); plan.name = input.filename().string();
    std::vector<fs::path> paths;
    for (const auto &entry : fs::recursive_directory_iterator(plan.input)) {
        check(stop);
        const auto relative = entry.path().lexically_relative(plan.input);
        if (std::any_of(relative.begin(),relative.end(),[](const auto &part){return part.string().starts_with(".");})) continue;
        if (entry.is_regular_file()) paths.push_back(entry.path());
        if (paths.size() > 10'000) fail("Model folder contains more than 10,000 files");
    }
    std::sort(paths.begin(),paths.end());
    for (const auto &path : paths) {
        progress(observer,stop,"scan",path.filename().string(),plan.files.size(),paths.size());
        Source source; source.logical = path; source.actual = resolvedSource(path);
        source.relative = path.lexically_relative(plan.input).generic_string();
        if (!safeRelative(fs::path(source.relative))) fail("Invalid source path");
        source.size = fs::file_size(source.actual); source.time = fs::last_write_time(source.actual);
        const auto extension = lower(path.extension().string());
        try {
            if (extension == ".safetensors" || extension == ".safetensor") {
                source.format = "safetensors"; readSafetensors(source);
            } else if (extension == ".gguf") {
                source.format = "gguf"; readGGUF(source); blob(source);
            } else {
                static const std::set<std::string> assets{".json",".txt",".model",".tiktoken",".vocab",".merges",".yaml",".yml",".ini",".cfg",".bin",".pt",".pth",".ckpt",".onnx",".npz",".md"};
                if (!assets.contains(extension) && !has(lower(path.filename().string()),"license") && !has(lower(path.filename().string()),"notice")) {
                    source.status="ignored"; source.reason="Unsupported supporting file"; source.format="other";
                } else {
                    source.format="asset";
                    if (extension == ".json") (void)parse(readSmall(path));
                    blob(source);
                }
            }
            source.role = inferRole(source); source.optional = source.role == "upscaler" || source.role == "lora";
            source.group = source.relative;
            if (std::find(excluded.begin(),excluded.end(),source.relative) != excluded.end()) {
                if (!source.optional) plan.errors.push_back("Required component cannot be excluded: " + source.relative);
                else { source.status="excluded"; source.reason="Optional component excluded"; }
            }
        } catch (const std::exception &error) {
            source.status="invalid"; source.reason=error.what(); source.tensors.clear();
            if (source.size <= 4096) {
                const auto contents = readSmall(path);
                if (has(contents,"Auth failed: credentials expired")) source.reason="Auth failed: credentials expired";
            }
        }
        plan.files.push_back(std::move(source));
    }
    for (const auto &file : plan.files)
        if (file.status == "invalid" && fs::path(file.relative).extension() == ".json")
            plan.errors.push_back("Invalid model configuration or index: " + file.relative + " · " + file.reason);
    std::map<std::string,std::size_t> byPath;
    for (std::size_t i=0;i<plan.files.size();++i) byPath[plan.files[i].relative]=i;
    for (auto &index : plan.files) if (index.relative.ends_with(".safetensors.index.json") && index.status=="included") {
        try {
            const auto mapping=parse(readSmall(index.actual)).at("weight_map");
            for (const auto &[key,value] : mapping.members()) {
                const fs::path reference(value.str());
                if (!safeRelative(reference)) fail("Unsafe shard path in " + index.relative);
                const auto relative=(fs::path(index.relative).parent_path()/reference).generic_string();
                const auto found=byPath.find(relative);
                if (found==byPath.end() || plan.files[found->second].status!="included") fail("Missing or invalid shard: " + relative);
                auto &file=plan.files[found->second]; file.group=index.relative;
                if (file.format!="safetensors" || std::none_of(file.tensors.begin(),file.tensors.end(),[&](const auto &t){return t.key==key;}))
                    fail("Shard is missing indexed tensor: " + key);
            }
        } catch (const std::exception &error) { plan.errors.push_back(error.what()); }
    }
    std::map<std::string,U64> groups;
    for (const auto &file : plan.files) if (file.status=="included" && file.role=="model") groups[file.group]=add(groups[file.group],file.size);
    std::string primaryGroup;
    if (!groups.empty()) primaryGroup=std::max_element(groups.begin(),groups.end(),[](const auto &a,const auto &b){return a.second<b.second;})->first;
    for (const auto &file : plan.files) {
        if (file.status=="invalid" || !file.metadata.contains("model_version")) continue;
        const auto version=file.metadata.at("model_version").str();
        if ((has(lower(file.relative),"ltx") || has(lower(file.metadata.dump()),"ltx")) && version.starts_with("2.")) {
            plan.ltx=true; plan.family="LTX "+version.substr(0,version.find('.',version.find('.')+1));
            plan.name=plan.family+" "+(has(lower(plan.input.filename().string()),"uncensored")?"Uncensored":"Model");
        }
    }
    for (std::size_t i=0;i<plan.files.size();++i)
        if (plan.files[i].status=="included" && plan.files[i].role=="model" && plan.files[i].group==primaryGroup) plan.primary.push_back(i);
    if (plan.primary.empty() && !plan.ltx) {
        for (std::size_t i=0;i<plan.files.size();++i) if (plan.files[i].status=="included" && plan.files[i].format=="gguf") {
            plan.primary.push_back(i); plan.files[i].role="model"; break;
        }
    }
    if (plan.primary.empty()) plan.errors.push_back("No valid main model was found in this folder");
    std::map<std::string,std::size_t> outputKeys;
    auto append=[&](std::size_t i, std::size_t t, std::string key) {
        auto &tensor=plan.files[i].tensors[t];
        if (!outputKeys.emplace(key,plan.entries.size()).second) { plan.errors.push_back("Tensor key conflict: "+key); return; }
        tensor.outputKey=std::move(key); plan.entries.push_back({i,t}); plan.payload=add(plan.payload,tensor.end-tensor.begin);
    };
    for (const auto i : plan.primary) {
        auto &file=plan.files[i];
        const auto version=file.metadata.contains("model_version")?file.metadata.at("model_version").str():"";
        if ((has(lower(file.relative),"ltx") || has(lower(file.metadata.dump()),"ltx")) && version.starts_with("2.")) {
            plan.ltx=true; plan.family="LTX "+version.substr(0,version.find('.',version.find('.')+1)); plan.name=plan.family+" "+(has(lower(plan.input.filename().string()),"uncensored")?"Uncensored":"Model");
        }
        for (std::size_t t=0;t<file.tensors.size();++t) {
            const auto key=file.format=="safetensors"?file.tensors[t].key:"__iild_assets__/model/"+file.relative;
            if (key.starts_with("__iild_assets__/") && file.format=="safetensors") plan.errors.push_back("Input already contains reserved model-package keys");
            append(i,t,key);
        }
    }
    for (std::size_t i=0;i<plan.files.size();++i) {
        auto &file=plan.files[i];
        if (file.status!="included" || std::find(plan.primary.begin(),plan.primary.end(),i)!=plan.primary.end()) continue;
        const bool embeddedRole=file.role=="video_vae" || file.role=="audio_vae" || file.role=="projections";
        std::size_t matches=0;
        if (embeddedRole) for (const auto &tensor : file.tensors) if (outputKeys.contains(roleKey(file,tensor))) ++matches;
        if (matches) {
            bool identical=matches==file.tensors.size();
            Hash fileHash; const auto bytes=sizeBytes(file.header.size()); fileHash.update(bytes.data(),bytes.size()); fileHash.update(file.header);
            for (auto &tensor : file.tensors) {
                check(stop);
                const auto found=outputKeys.find(roleKey(file,tensor));
                const auto candidate=hashRange(file.actual,file.base+tensor.begin,tensor.end-tensor.begin,stop,observer,file.relative,&fileHash);
                if (found==outputKeys.end()) { identical=false; continue; }
                const auto entry=plan.entries[found->second]; auto &base=plan.files[entry.source].tensors[entry.tensor];
                auto &baseFile=plan.files[entry.source];
                if (base.dtype!=tensor.dtype || base.shape!=tensor.shape) { identical=false; continue; }
                if (base.digest==emptyDigest) base.digest=hashRange(baseFile.actual,baseFile.base+base.begin,base.end-base.begin,stop,observer,"Checking embedded "+file.role);
                if (candidate!=base.digest) identical=false;
                tensor.outputKey=base.outputKey; tensor.digest=candidate;
            }
            stamp(file); file.digest=fileHash.finish();
            if (identical) { file.status="duplicate"; file.reason="Identical tensors already in the main checkpoint"; }
            else { file.status="conflict"; file.reason="Component conflicts with embedded main-model tensors"; plan.errors.push_back("Component conflict: "+file.relative); }
            continue;
        }
        for (std::size_t t=0;t<file.tensors.size();++t) {
            std::string key;
            if (file.format!="safetensors") key="__iild_assets__/"+file.role+"/"+file.relative;
            else if (plan.ltx && embeddedRole) key=roleKey(file,file.tensors[t]);
            else key="components/"+file.role+"/"+file.relative+"/"+file.tensors[t].key;
            append(i,t,key);
        }
    }
    auto roleAvailable=[&](const std::string &role,const std::string &tensorPrefix) {
        for (const auto &file : plan.files) if ((file.status=="included" || file.status=="duplicate") &&
            (file.role==role || prefix(file,tensorPrefix))) return true;
        return false;
    };
    if (plan.ltx) {
        for (const auto &[role,key,label] : std::array<std::array<std::string,3>,4>{{
            {"video_vae","vae.","video decoder"},{"audio_vae","audio_vae.","audio decoder"},
            {"projections","text_embedding_projection.","text projections"},{"text_encoder","text_encoder.","text encoder"}}})
            if (!roleAvailable(role,key)) plan.errors.push_back("Required "+label+" is missing");
    }
    for (const auto &file : plan.files) if (file.relative=="model_index.json" && file.status=="included") {
        const auto index=parse(readSmall(file.actual));
        for (const auto &[name,value] : index.members()) {
            if (name.starts_with("_") || !value.is(json_type_array) || value.size()!=2) continue;
            auto *classValue=json_object_array_get_idx(value.get(),1);
            if (!classValue || json_object_is_type(classValue,json_type_null)) continue;
            if (!safeRelative(fs::path(name)) || name.find('/') != std::string::npos)
                fail("Unsafe Diffusers component name: " + name);
            const bool weightsRequired = name == "unet" || name == "transformer" || name == "vae"
                || name.starts_with("text_encoder") || name.starts_with("controlnet");
            const bool exists=std::any_of(plan.files.begin(),plan.files.end(),[&](const auto &f){
                if (!f.relative.starts_with(name+"/") || f.status!="included") return false;
                if (!weightsRequired) return true;
                const auto extension = lower(fs::path(f.relative).extension().string());
                return f.format == "safetensors" || f.format == "gguf" ||
                    (f.size && (extension == ".bin" || extension == ".pt" || extension == ".ckpt" || extension == ".onnx"));
            });
            if (!exists) plan.errors.push_back("Required Diffusers component is missing: "+name);
        }
    }
    for (const auto &file : plan.files) if (file.status=="included" || file.status=="duplicate") stamp(file);
    return plan;
}

J manifest(const Plan &plan) {
    auto result=J::object(), files=J::array();
    result.set("schema",schema); result.set("family",plan.family); result.set("name",plan.name);
    for (const auto &source : plan.files) {
        auto file=J::object(), tensors=J::array();
        file.set("path",source.relative); file.set("format",source.format); file.set("role",source.role);
        file.set("status",source.status); file.set("reason",source.reason); file.set("size",source.size);
        file.set("sha256",source.digest); file.set("metadata",source.metadata); file.set("header",source.header);
        if (source.status=="included" || source.status=="duplicate") for (const auto &tensor : source.tensors) {
            auto value=J::object();
            value.set("key",tensor.key); value.set("output_key",tensor.outputKey); value.set("sha256",tensor.digest);
            value.set("begin",tensor.begin); value.set("end",tensor.end); tensors.push(value);
        }
        file.set("tensors",tensors); files.push(file);
    }
    result.set("files",files); return result;
}
std::string outputHeader(const Plan &plan) {
    auto header=J::object(), metadata=J::object(); U64 offset=0;
    for (const auto &entry : plan.entries) {
        const auto &tensor=plan.files[entry.source].tensors[entry.tensor];
        auto value=J::object(), shape=J::array(), offsets=J::array();
        for (const auto dimension : tensor.shape) shape.push(dimension);
        offsets.push(offset); offset=add(offset,tensor.end-tensor.begin); offsets.push(offset);
        value.set("dtype",tensor.dtype); value.set("shape",shape); value.set("data_offsets",offsets); header.set(tensor.outputKey,value);
    }
    if (!plan.primary.empty() && plan.files[plan.primary.front()].format=="safetensors")
        metadata=plan.files[plan.primary.front()].metadata;
    // Copy the original metadata instead of mutating the source record.
    metadata=parse(metadata.dump());
    metadata.set("iild_package_schema",schema); metadata.set("iild_package_manifest",manifest(plan).dump());
    header.set("__metadata__",metadata);
    auto text=header.dump();
    if (text.size()>headerLimit-8) fail("Package metadata exceeds the Safetensors header limit");
    text.append((8-text.size()%8)%8,' '); return text;
}
J components(const Plan &plan) {
    auto result=J::array();
    auto push=[&](std::string id,std::string label,std::string description,std::string icon,U64 size,bool optional,bool included) {
        auto value=J::object(); value.set("id",id); value.set("label",label); value.set("description",description);
        value.set("icon",icon); value.set("size",size); value.set("optional",optional); value.set("included",included); result.push(value);
    };
    U64 primarySize=0;
    for (const auto i : plan.primary) primarySize=add(primarySize,plan.files[i].size);
    if (!plan.primary.empty()) {
        bool fp8=false;
        for (const auto i : plan.primary) for (const auto &t : plan.files[i].tensors) if (t.dtype.starts_with("F8")) fp8=true;
        push("model",plan.ltx?"Video + audio model":"Main model",plan.family+(fp8?" · FP8 mixed":"")+" · Main checkpoint","toolWindowModelChecker",primarySize,false,true);
    }
    if (plan.ltx) {
        const bool encoderAvailable=std::any_of(plan.files.begin(),plan.files.end(),[](const auto &file) {
            return file.role=="text_encoder" && file.status=="included";
        });
        if (!encoderAvailable) push("text_encoder","Text encoder","Required text encoder was not found","toolWindowModelChecker",0,false,false);
        for (const auto &[role,key,label] : std::array<std::array<std::string,3>,3>{{
            {"video_vae","vae.","Video decoder"},{"audio_vae","audio_vae.","Audio decoder + vocoder"},
            {"projections","text_embedding_projection.","Text projections"}}}) {
            bool embedded=false,available=false; U64 size=0;
            for (const auto i : plan.primary) if (prefix(plan.files[i],key)) embedded=true;
            for (const auto &file : plan.files) if (file.role==role && (file.status=="included" || file.status=="duplicate")) { available=true; size=add(size,file.status=="included"?file.size:0); }
            push(role,label,embedded?"Already inside the main checkpoint":available?"Included from the model folder":"Required component was not found",
                role=="projections"?"toolWindowModelChecker":"localSwiftPackageDependency",embedded?0:size,false,embedded||available);
        }
    }
    for (std::size_t i=0;i<plan.files.size();++i) {
        const auto &file=plan.files[i];
        if (std::find(plan.primary.begin(),plan.primary.end(),i)!=plan.primary.end() ||
            (plan.ltx && (file.role=="video_vae" || file.role=="audio_vae" || file.role=="projections")) ||
            file.status=="duplicate" || file.status=="invalid" || file.status=="ignored" || file.status=="conflict") continue;
        const auto label=file.role=="text_encoder"?"Text encoder":file.role=="upscaler"?"Spatial upscaler":file.role=="lora"?"LoRA":file.role=="model"?"Additional checkpoint":file.role=="video_vae"?"Video decoder":file.relative;
        const auto detail=file.format=="gguf"?"GGUF · Original format and quantization retained":
            file.role=="upscaler"?"2× resolution · Optional":file.role=="lora"?"Separate adapter · Original weights retained":file.relative;
        push(file.relative,label,detail,"toolWindowModelChecker",file.size,file.optional,file.status=="included");
    }
    return result;
}
J report(const Plan &plan, const std::string &operation) {
    auto result=J::object(), files=J::array(), errors=J::array(); U64 included=0,duplicates=0,invalid=0,duplicateSize=0;
    for (const auto &source : plan.files) {
        auto file=J::object(); file.set("id",source.relative); file.set("path",source.logical.string());
        file.set("relative_path",source.relative); file.set("role",source.role); file.set("format",source.format);
        file.set("size",source.size); file.set("status",source.status); file.set("reason",source.reason); file.set("optional",source.optional); files.push(file);
        if (source.status=="included") ++included;
        if (source.status=="duplicate") { ++duplicates; duplicateSize=add(duplicateSize,source.size); }
        if (source.status=="invalid") ++invalid;
    }
    for (const auto &error : plan.errors) errors.push(error);
    const auto list=components(plan); U64 count=0;
    for (std::size_t i=0;i<list.size();++i) if (list.at(i).at("included").boolean()) ++count;
    result.set("schema",reportSchema); result.set("operation",operation); result.set("ready",plan.errors.empty());
    result.set("source_directory",plan.input.string()); result.set("family",plan.family); result.set("suggested_name",plan.name);
    result.set("files",files); result.set("components",list); result.set("errors",errors);
    result.set("scanned_file_count",static_cast<U64>(plan.files.size())); result.set("included_file_count",included);
    result.set("duplicate_file_count",duplicates); result.set("invalid_file_count",invalid); result.set("duplicate_bytes",duplicateSize);
    result.set("component_count",count); result.set("payload_bytes",plan.payload);
    result.set("estimated_size_bytes",add(add(8,outputHeader(plan).size()),plan.payload)); return result;
}
fs::path temporaryPath(const fs::path &target) {
    std::random_device random;
    std::string suffix;
    constexpr char hex[]="0123456789abcdef";
    for (unsigned i=0;i<32;++i) suffix+=hex[random()&15];
    return target.parent_path()/("."+target.filename().string()+".partial-"+suffix);
}
struct Temporary {
    fs::path path;
    bool directory=false;
    ~Temporary() { std::error_code error; if (directory) fs::remove_all(path,error); else fs::remove(path,error); }
};
void createExclusive(const fs::path &path) {
#ifdef _WIN32
    const auto handle=CreateFileW(path.c_str(),GENERIC_WRITE,0,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
    if (handle==INVALID_HANDLE_VALUE) fail("Could not create temporary output"); CloseHandle(handle);
#else
    const int descriptor=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
    if (descriptor<0) fail("Could not create temporary output: "+path.string()); ::close(descriptor);
#endif
}
void syncFile(const fs::path &path) {
#ifdef _WIN32
    const auto handle=CreateFileW(path.c_str(),GENERIC_WRITE,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
    if (handle==INVALID_HANDLE_VALUE) fail("Could not synchronize the output");
    const auto ok=FlushFileBuffers(handle); CloseHandle(handle); if (!ok) fail("Output synchronization failed");
#else
    const auto descriptor=::open(path.c_str(),O_RDONLY);
    if (descriptor<0) fail("Could not synchronize the output");
    const auto result=::fsync(descriptor); ::close(descriptor); if (result!=0) fail("Output synchronization failed");
#endif
}
void commit(const fs::path &temporary,const fs::path &target) {
#ifdef _WIN32
    if (!MoveFileExW(temporary.c_str(),target.c_str(),MOVEFILE_WRITE_THROUGH)) fail("Could not save without overwriting an existing output");
#elif defined(__APPLE__)
    if (::renamex_np(temporary.c_str(),target.c_str(),RENAME_EXCL)!=0) fail("Could not save without overwriting an existing output");
#else
    fs::create_hard_link(temporary,target); fs::remove(temporary);
#endif
}
void destination(const fs::path &target,U64 bytes) {
    if (target.empty() || fs::exists(target) || fs::is_symlink(fs::symlink_status(target))) fail("Output already exists or is invalid; choose a new name");
    if (!fs::is_directory(target.parent_path())) fail("Choose an existing output folder");
    if (fs::space(target.parent_path()).available<bytes) fail("Not enough free space in the output folder");
}

struct Verified {
    Source file;
    J manifest=J::object();
    std::string digest;
};
Verified verify(const fs::path &input,std::stop_token stop,const ModelPackagingObserver &observer) {
    Verified value;
    value.file.logical=fs::absolute(input); value.file.actual=resolvedSource(input);
    value.file.size=fs::file_size(input); value.file.time=fs::last_write_time(input); readSafetensors(value.file);
    if (!value.file.metadata.contains("iild_package_schema") || value.file.metadata.at("iild_package_schema").str()!=schema)
        fail("This file is not an iiLocalDiffusion model package");
    value.manifest=parse(value.file.metadata.at("iild_package_manifest").str());
    if (value.manifest.at("schema").str()!=schema) fail("Unsupported model-package manifest");
    std::map<std::string,std::string> digests;
    std::map<std::string,const Tensor *> tensors;
    for (const auto &tensor : value.file.tensors) tensors[tensor.key]=&tensor;
    const auto files=value.manifest.at("files");
    if (!files.is(json_type_array) || files.size()>10'000) fail("Invalid package file list");
    std::set<std::string> paths;
    for (std::size_t i=0;i<files.size();++i) {
        const auto source=files.at(i); const auto relative=source.at("path").str();
        if (!safeRelative(fs::path(relative)) || !paths.insert(relative).second) fail("Unsafe or duplicate package source path");
        const auto status=source.at("status").str();
        if (status!="included" && status!="duplicate") continue;
        const auto original=source.at("tensors"); U64 sourceCursor=0;
        for (std::size_t t=0;t<original.size();++t) {
            const auto record=original.at(t);
            const auto key=record.at("output_key").str(), digest=record.at("sha256").str();
            const auto begin=record.at("begin").number(),end=record.at("end").number();
            if (!tensors.contains(key) || digest.size()!=64 || (end!=begin && begin!=sourceCursor) || end<begin ||
                end-begin!=tensors[key]->end-tensors[key]->begin) fail("Invalid source mapping in package");
            sourceCursor=std::max(sourceCursor,end);
            if (digests.contains(key) && digests[key]!=digest) fail("Conflicting package checksums");
            digests[key]=digest;
        }
        const auto base=source.at("format").str()=="safetensors"?add(8,source.at("header").str().size()):0;
        if (add(base,sourceCursor)!=source.at("size").number()) fail("Incomplete source mapping in package");
    }
    if (digests.size()!=tensors.size()) fail("Package manifest does not cover all tensor data");
    std::ifstream file(input,std::ios::binary); Hash whole;
    const auto prefixBytes=sizeBytes(value.file.header.size()); whole.update(prefixBytes.data(),prefixBytes.size()); whole.update(value.file.header);
    file.seekg(static_cast<std::streamoff>(value.file.base));
    std::vector<char> buffer(chunkSize); U64 done=0;
    for (const auto &tensor : value.file.tensors) {
        Hash hash; U64 size=tensor.end-tensor.begin,position=0;
        while (position<size) {
            progress(observer,stop,"verify","Checking "+tensor.key,done,value.file.size-value.file.base);
            const auto count=static_cast<std::size_t>(std::min<U64>(size-position,buffer.size()));
            if (!file.read(buffer.data(),static_cast<std::streamsize>(count))) fail("Package data is truncated");
            hash.update(buffer.data(),count); whole.update(buffer.data(),count); position+=count; done+=count;
        }
        if (hash.finish()!=digests.at(tensor.key)) fail("Package checksum mismatch: "+tensor.key);
    }
    stamp(value.file); value.digest=whole.finish(); return value;
}
}

std::string scanModelFolder(const fs::path &input,std::span<const std::string> excluded,
                           std::stop_token stop,const ModelPackagingObserver &observer) {
    return report(scan(input,excluded,stop,observer),"scan").dump();
}
std::string createModelPackage(const fs::path &input,const fs::path &output,std::span<const std::string> excluded,
                              std::stop_token stop,const ModelPackagingObserver &observer) {
    const auto target=fs::absolute(output).lexically_normal();
    destination(target,0);
    auto plan=scan(input,excluded,stop,observer);
    if (!plan.errors.empty()) fail(plan.errors.front());
    auto header=outputHeader(plan);
    destination(target,add(add(8,header.size()),plan.payload));
    Temporary temporary{temporaryPath(target)};
    createExclusive(temporary.path);
    std::fstream file(temporary.path,std::ios::binary|std::ios::in|std::ios::out);
    if (!file) fail("Could not write the temporary output");
    const auto prefixBytes=sizeBytes(header.size());
    file.write(prefixBytes.data(),prefixBytes.size()); file.write(header.data(),static_cast<std::streamsize>(header.size()));
    std::vector<char> buffer(chunkSize); U64 done=0;
    std::size_t current=std::numeric_limits<std::size_t>::max();
    std::ifstream source; std::unique_ptr<Hash> originalHash;
    auto finishSource=[&] {
        if (current!=std::numeric_limits<std::size_t>::max()) {
            plan.files[current].digest=originalHash->finish(); stamp(plan.files[current]);
        }
    };
    progress(observer,stop,"write","Writing the model package",0,plan.payload);
    for (const auto &entry : plan.entries) {
        auto &inputFile=plan.files[entry.source]; auto &tensor=inputFile.tensors[entry.tensor];
        if (current!=entry.source) {
            finishSource(); current=entry.source; stamp(inputFile); source.close(); source.clear(); source.open(inputFile.actual,std::ios::binary);
            originalHash=std::make_unique<Hash>();
            if (inputFile.format=="safetensors") {
                const auto bytes=sizeBytes(inputFile.header.size()); originalHash->update(bytes.data(),bytes.size()); originalHash->update(inputFile.header);
            }
        }
        source.seekg(static_cast<std::streamoff>(inputFile.base+tensor.begin)); Hash hash; U64 position=0;
        while (position<tensor.end-tensor.begin) {
            progress(observer,stop,"write",inputFile.relative,done,plan.payload);
            const auto count=static_cast<std::size_t>(std::min<U64>(tensor.end-tensor.begin-position,buffer.size()));
            if (!source.read(buffer.data(),static_cast<std::streamsize>(count))) fail("Source is truncated: "+inputFile.relative);
            file.write(buffer.data(),static_cast<std::streamsize>(count));
            if (!file) fail("Output write failed; check free space and storage");
            hash.update(buffer.data(),count); originalHash->update(buffer.data(),count); position+=count; done+=count;
        }
        const auto digest=hash.finish();
        if (tensor.digest!=emptyDigest && tensor.digest!=digest) fail("Source tensor changed after duplicate validation: "+tensor.key);
        tensor.digest=digest;
    }
    finishSource();
    for (const auto &inputFile : plan.files) if (inputFile.status=="included" || inputFile.status=="duplicate") stamp(inputFile);
    const auto finalHeader=outputHeader(plan);
    if (finalHeader.size()!=header.size()) fail("Internal package header reservation changed");
    file.seekp(8); file.write(finalHeader.data(),static_cast<std::streamsize>(finalHeader.size())); file.flush();
    if (!file) fail("Could not finalize the package metadata"); file.close();
    const auto verified=verify(temporary.path,stop,observer);
    progress(observer,stop,"save","Saving the verified model package",done,plan.payload);
    syncFile(temporary.path); check(stop); commit(temporary.path,target);
    auto result=report(plan,"create"); result.set("verified",true); result.set("output",target.string());
    result.set("size",static_cast<U64>(fs::file_size(target))); result.set("sha256",verified.digest);
    return result.dump();
}
std::string verifyModelPackage(const fs::path &input,std::stop_token stop,const ModelPackagingObserver &observer) {
    const auto verified=verify(input,stop,observer);
    auto result=J::object(); result.set("schema",reportSchema); result.set("operation","verify");
    result.set("verified",true); result.set("output",fs::absolute(input).string());
    result.set("size",verified.file.size); result.set("sha256",verified.digest); return result.dump();
}
std::string extractModelPackage(const fs::path &input,const fs::path &output,std::stop_token stop,
                               const ModelPackagingObserver &observer) {
    const auto target=fs::absolute(output).lexically_normal(); destination(target,0);
    const auto verified=verify(input,stop,observer); const auto files=verified.manifest.at("files");
    U64 total=0;
    for (std::size_t i=0;i<files.size();++i) {
        const auto file=files.at(i);
        const auto status=file.at("status").str();
        if (status=="included" || status=="duplicate") total=add(total,file.at("size").number());
    }
    destination(target,total); Temporary temporary{temporaryPath(target),true};
    fs::create_directory(temporary.path);
    std::map<std::string,const Tensor *> tensors;
    for (const auto &tensor : verified.file.tensors) tensors[tensor.key]=&tensor;
    std::ifstream source(input,std::ios::binary); std::vector<char> buffer(chunkSize); U64 done=0,countFiles=0;
    for (std::size_t i=0;i<files.size();++i) {
        const auto record=files.at(i);
        const auto status=record.at("status").str();
        if (status!="included" && status!="duplicate") continue;
        const auto path=temporary.path/fs::path(record.at("path").str()); fs::create_directories(path.parent_path());
        std::ofstream file(path,std::ios::binary); Hash hash;
        if (record.at("format").str()=="safetensors") {
            const auto header=record.at("header").str();
            const auto bytes=sizeBytes(header.size());
            file.write(bytes.data(),bytes.size()); file.write(header.data(),static_cast<std::streamsize>(header.size()));
            hash.update(bytes.data(),bytes.size()); hash.update(header);
        }
        const auto mapping=record.at("tensors");
        for (std::size_t t=0;t<mapping.size();++t) {
            const auto *tensor=tensors.at(mapping.at(t).at("output_key").str());
            source.clear(); source.seekg(static_cast<std::streamoff>(verified.file.base+tensor->begin)); U64 position=0;
            while (position<tensor->end-tensor->begin) {
                progress(observer,stop,"extract",record.at("path").str(),done,total);
                const auto size=static_cast<std::size_t>(std::min<U64>(tensor->end-tensor->begin-position,buffer.size()));
                if (!source.read(buffer.data(),static_cast<std::streamsize>(size))) fail("Package changed during extraction");
                file.write(buffer.data(),static_cast<std::streamsize>(size)); hash.update(buffer.data(),size); position+=size; done+=size;
                if (!file) fail("Could not write extracted model");
            }
        }
        file.close();
        if (hash.finish()!=record.at("sha256").str()) fail("Reconstructed source checksum mismatch");
        ++countFiles;
    }
    stamp(verified.file); check(stop);
    if (fs::exists(target)) fail("Extraction destination already exists");
#ifdef _WIN32
    if (!MoveFileExW(temporary.path.c_str(),target.c_str(),MOVEFILE_WRITE_THROUGH)) fail("Could not save extracted directory");
#elif defined(__APPLE__)
    if (::renamex_np(temporary.path.c_str(),target.c_str(),RENAME_EXCL)!=0) fail("Could not save extracted directory");
#else
    // renameat2 is not available on every Unix platform; an empty reserved
    // directory prevents accidental replacement of existing user data.
    if (!fs::create_directory(target)) fail("Extraction destination already exists");
    try { fs::rename(temporary.path,target); } catch (...) { fs::remove(target); throw; }
#endif
    auto result=J::object(); result.set("schema",reportSchema); result.set("operation","extract"); result.set("verified",true);
    result.set("output",target.string()); result.set("file_count",countFiles); return result.dump();
}
}
