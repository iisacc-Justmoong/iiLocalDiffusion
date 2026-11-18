#include <iiFileProviderStorage.h>
#include <iiFileProviderPresentation.h>
#include "ModelPackaging/ModelPackaging.hpp"
#include "ModelPackaging/ModelInspection.hpp"
#include "ModelGGUFInspection.hpp"
#include "ModelWanNames.hpp"
#include "ModelTorchCandidate.hpp"

#include <json-c/json.h>
#include <openssl/evp.h>
#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <csignal>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <string_view>
#include <stdexcept>
#include <tuple>
#include <utility>
#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
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
bool forbiddenModelExtension(const fs::path &path) {
    return lower(path.extension().string()) == ".onnx";
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
    std::vector<char> inputBuffer(1024 * 1024);
    std::ifstream file;
    file.rdbuf()->pubsetbuf(inputBuffer.data(), static_cast<std::streamsize>(inputBuffer.size()));
    file.open(path, std::ios::binary);
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
    progress(observer, stop, "deduplicate", detail, done, size);
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
    std::string position = "component", detectedRole, roleSource = "detected", positionSource = "automatic";
    U64 size = 0, base = 0;
    fs::file_time_type time;
    J metadata = J::object();
    std::vector<Tensor> tensors;
    bool optional = false, shardIndex = false, indexAlias = false;
};

// Inspect only the container prefix. Full structure and data ranges are still
// validated by the selected reader before a source can enter a package.
std::string modelContentFormat(const fs::path &path, U64 size) {
    std::ifstream file(path, std::ios::binary);
    if (!file) fail("Cannot read model source: " + path.string());
    std::array<unsigned char, 9> prefix{};
    file.read(reinterpret_cast<char *>(prefix.data()), static_cast<std::streamsize>(prefix.size()));
    const auto count = file.gcount();
    if (count >= 4 && prefix[0] == 'G' && prefix[1] == 'G' && prefix[2] == 'U' && prefix[3] == 'F')
        return "gguf";
    if (count < static_cast<std::streamsize>(prefix.size()) || size < prefix.size()) return {};
    U64 header = 0;
    for (std::size_t i = 0; i < 8; ++i) header |= U64(prefix[i]) << (8 * i);
    // A plausible but truncated header still belongs to the Safetensors reader,
    // which reports it as invalid instead of silently storing it as an asset.
    if (header >= 2 && header <= headerLimit && prefix[8] == '{') return "safetensors";
    return {};
}

// An index can be renamed independently of its shards. Bound this cheap
// candidate check before asking the index parser to examine any JSON.
bool looksLikeJsonObject(const fs::path &path, U64 size) {
    if (!size || size>headerLimit) return false;
    std::ifstream file(path,std::ios::binary);
    if (!file) return false;
    std::array<char,4096> bytes{};
    file.read(bytes.data(),static_cast<std::streamsize>(bytes.size()));
    for (std::streamsize i=0;i<file.gcount();++i) {
        const char c=bytes[static_cast<std::size_t>(i)];
        if (c==' ' || c=='\t' || c=='\n' || c=='\r') continue;
        return c=='{';
    }
    return false;
}

std::optional<unsigned> knownDtypeBits(const std::string &dtype) {
    static const std::map<std::string, unsigned> bits{
        {"BOOL",8},{"U8",8},{"I8",8},{"I16",16},{"U16",16},{"I32",32},{"U32",32},
        {"I64",64},{"U64",64},{"F16",16},{"BF16",16},{"F32",32},{"F64",64},{"C64",64},
        {"F8_E4M3",8},{"F8_E5M2",8},{"F8_E4M3FN",8},{"F8_E4M3FNUZ",8},{"F8_E5M2FNUZ",8},{"F8_E8M0",8},
        {"F4",4},{"F6_E2M3",6},{"F6_E3M2",6}
    };
    const auto found = bits.find(dtype);
    if (found == bits.end()) return std::nullopt;
    return found->second;
}
void readSafetensors(Source &source, std::string *preloadedHeader=nullptr, bool allowUnknownDtype=false) {
    std::ifstream file;
    U64 size=0;
    if (preloadedHeader) size=preloadedHeader->size();
    else {file.open(source.actual,std::ios::binary);size=readLE(file,8);}
    if (!size || size > headerLimit || size > source.size || size > source.size - std::min<U64>(source.size, 8))
        fail("Invalid Safetensors header length");
    if (preloadedHeader) source.header=std::move(*preloadedHeader);
    else {
        source.header.resize(static_cast<std::size_t>(size));
        if (!file.read(source.header.data(), static_cast<std::streamsize>(size))) fail("Truncated Safetensors header");
    }
    if (source.header.front() != '{') fail("Safetensors header must begin with an object");
    const auto header = parse(source.header);
    source.base = 8 + size;
    if (header.contains("__metadata__")) {
        source.metadata = header.at("__metadata__");
        for (const auto &[key, value] : source.metadata.members()) { (void)key; (void)value.str(); }
    }
    const bool namesOnly=allowUnknownDtype && !source.metadata.contains("iild_package_schema")
        && !source.metadata.contains("iild_package_manifest");
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
        if (tensor.dtype.empty() || tensor.dtype.find('\0')!=std::string::npos) fail("Invalid Safetensors dtype");
        if (tensor.end < tensor.begin || tensor.end > source.size - source.base)
            fail("Tensor shape does not match the file data range: " + key);
        const auto known=knownDtypeBits(tensor.dtype);
        if (!known && !namesOnly) fail("Unsupported Safetensors dtype: " + tensor.dtype);
        if (known) {
            const auto bits=*known;
            if (count > (std::numeric_limits<U64>::max() - 7) / bits) fail("Tensor storage size overflows");
            const auto bytes = (count * bits + 7) / 8;
            if (tensor.end - tensor.begin != bytes) fail("Tensor shape does not match the file data range: " + key);
        } else if ((count==0)!=(tensor.begin==tensor.end)) {
            fail("Tensor shape does not match the file data range: " + key);
        }
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
// GGUF may be a complete file or a bounded U8 asset inside a package. Keep
// offsets relative to the asset and never let metadata reads cross its extent.
class GgufInput {
    std::ifstream file_;
    U64 length_, position_=0;
public:
    GgufInput(const fs::path &path,U64 offset,U64 length):file_(path,std::ios::binary),length_(length) {
        if (!file_ || offset>static_cast<U64>(std::numeric_limits<std::streamoff>::max()))
            fail("Cannot open GGUF source");
        file_.seekg(static_cast<std::streamoff>(offset));
        if (!file_) fail("Cannot seek to GGUF source");
    }
    U64 position() const {return position_;}
    void read(char *data,U64 count) {
        if (position_>length_ || count>length_-position_ || position_>headerLimit || count>headerLimit-position_ ||
            count>static_cast<U64>(std::numeric_limits<std::streamsize>::max()))
            fail("GGUF metadata exceeds the asset or header limit");
        if (count && !file_.read(data,static_cast<std::streamsize>(count))) fail("Truncated GGUF metadata");
        position_+=count;
    }
    U64 little(unsigned count) {
        std::array<unsigned char,8> bytes{};
        if (count>bytes.size()) fail("Invalid GGUF integer size");
        read(reinterpret_cast<char *>(bytes.data()),count);
        U64 result=0;for(unsigned i=0;i<count;++i) result|=static_cast<U64>(bytes[i])<<(i*8);
        return result;
    }
};
std::string ggufString(GgufInput &file) {
    const auto length = file.little(8);
    if (length > headerLimit - file.position()) fail("GGUF metadata string exceeds the limit");
    std::string text(static_cast<std::size_t>(length), '\0');
    file.read(text.data(),length);
    return text;
}
void skipGGUF(GgufInput &file, unsigned type, unsigned depth = 0) {
    if (depth > 8) fail("GGUF metadata nesting is too deep");
    if (type == 8) { (void)ggufString(file); return; }
    if (type == 9) {
        const auto element = static_cast<unsigned>(file.little(4)); const auto count = file.little(8);
        if (count > 10'000'000) fail("GGUF array exceeds the limit");
        for (U64 i = 0; i < count; ++i) skipGGUF(file, element, depth + 1);
        return;
    }
    static constexpr std::array<unsigned, 13> sizes{1,1,2,2,4,4,4,1,0,0,8,8,8};
    if (type >= sizes.size() || sizes[type] == 0) fail("Invalid GGUF metadata type");
    (void)file.little(sizes[type]);
}
void readGGUF(Source &source, detail::GgufCatalog *catalog = nullptr,U64 assetOffset=0) {
    GgufInput file(source.actual,assetOffset,source.size);
    if (file.little(4) != 0x46554747) fail("Invalid GGUF magic");
    const auto version = file.little(4);
    if (version != 2 && version != 3) fail("Unsupported GGUF version");
    const auto tensors = file.little(8), fields = file.little(8);
    if (!tensors || tensors > 1'000'000 || fields > 1'000'000) fail("Invalid GGUF tensor or metadata count");
    U64 alignment = 32;
    std::set<std::string> metadataKeys;
    for (U64 i = 0; i < fields; ++i) {
        const auto key = ggufString(file); const auto type = static_cast<unsigned>(file.little(4));
        if (key.empty() || !metadataKeys.insert(key).second) fail("Invalid or duplicate GGUF metadata key");
        if (type == 8) source.metadata.set(key, ggufString(file));
        else if (key == "general.alignment" && type == 4) alignment = file.little(4);
        else if (key == "society.modalities" && type == 9) {
            const auto element = file.little(4), count = file.little(8);
            if (element != 8 || count > 4) fail("Invalid GGUF society modalities");
            auto values = J::array();
            for (U64 j = 0; j < count; ++j) values.push(ggufString(file));
            source.metadata.set(key, values);
        }
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
        const auto name = ggufString(file);
        if (name.empty() || !names.insert(name).second) fail("Invalid or duplicate GGUF tensor name");
        if (catalog) catalog->tensorNames.push_back(name);
        const auto dimensions = file.little(4);
        if (!dimensions || dimensions > 4) fail("Invalid GGUF tensor dimensions");
        U64 count=1,first=0;
        for (U64 d = 0; d < dimensions; ++d) {
            const auto size=file.little(8);
            if (!size || count>std::numeric_limits<U64>::max()/size) fail("Invalid GGUF tensor shape");
            if (!d) first=size;
            count*=size;
        }
        const auto type=file.little(4),offset=file.little(8);
        if (!encoding.contains(type)) {
            if (!catalog) fail("Unsupported GGUF tensor encoding: "+std::to_string(type));
            // The encoding's payload extent is unknown; check that its aligned
            // starting byte exists, without claiming that the payload is valid.
            ranges.emplace_back(offset, add(offset, 1));
            continue;
        }
        const auto [block,bytes]=encoding.at(type);
        if (first%block || count/block>std::numeric_limits<U64>::max()/bytes) fail("Invalid GGUF quantization block shape");
        ranges.emplace_back(offset,add(offset,count/block*bytes));
    }
    const auto position = file.position();
    const auto base = (position + alignment - 1) & ~(alignment - 1);
    if (base >= source.size) fail("GGUF has no tensor data");
    std::sort(ranges.begin(),ranges.end());
    U64 end=0;
    for (const auto &[begin,next] : ranges) {
        if (begin<end || begin%alignment || next>source.size-base) fail("Invalid or truncated GGUF tensor range");
        end=next;
    }
    source.metadata.set("gguf.version", version);
    if (catalog) {
        catalog->metadataJson = source.metadata.dump();
        catalog->metadataBytesRead = position;
    }
}
bool prefix(const Source &source, const std::string &value) {
    return std::any_of(source.tensors.begin(), source.tensors.end(), [&](const auto &tensor) { return tensor.key.starts_with(value); });
}
bool tensorNamed(const Source &source, const std::string &value) {
    return std::any_of(source.tensors.begin(), source.tensors.end(), [&](const auto &tensor) { return tensor.key == value; });
}
std::string inferRole(const Source &source) {
    const auto path = lower(source.relative), config = source.metadata.contains("config") ? lower(source.metadata.at("config").str()) : "";
    if (has(path,"lora") || prefix(source,"lora_") ||
        std::any_of(source.tensors.begin(),source.tensors.end(),[](const auto &t){return has(t.key,".lora_A.") || has(t.key,".lora_down.");})) return "lora";
    if (has(path,"upscaler") || has(config,"latentupsampler")) return "upscaler";
    if (prefix(source,"model.diffusion_model.") || prefix(source,"diffusion_model.") || has(path,"transformer/") || has(path,"unet/")) return "model";
    // A complete MOSS transformer can coexist with language-model tensors.
    // Check the transformer signature before classifying an isolated Qwen encoder.
    if (tensorNamed(source,"video_embedder.weight") && tensorNamed(source,"final_linear_video.weight")
        && tensorNamed(source,"layers.0.proj.qkv.weight")) return "model";
    if (has(path,"audio_vae") || prefix(source,"audio_vae.") || prefix(source,"vocoder.")) return "audio_vae";
    if (has(path,"projection") || prefix(source,"text_embedding_projection.")) return "projections";
    if (has(path,"vae") || has(config,"autoencoder")) return "video_vae";
    if (has(path,"text_encoder") || has(path,"text encoder/")
        || path.starts_with("llm/") || has(path,"/llm/") || prefix(source,"text_model.")
        || prefix(source,"model.language_model.")) return "text_encoder";
    if (source.format == "gguf") {
        if (source.metadata.contains("general.architecture") &&
            detail::isDiffusionGgufArchitecture(source.metadata.at("general.architecture").str())) return "model";
        return "text_encoder";
    }
    return source.format == "safetensors" ? "model" : "other";
}
fs::path resolvedSource(const fs::path &path) {
#ifdef _WIN32
    const auto handle = iiFileProvider::native::createFile(path.c_str(), FILE_READ_ATTRIBUTES,
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
    std::optional<double> wanMoeBoundary;
    U64 payload = 0;
};
void requireNoOnnxManifest(const J &manifest) {
    const auto files = manifest.at("files");
    if (!files.is(json_type_array) || files.size() > 10'000) fail("Invalid package file list");
    for (std::size_t i = 0; i < files.size(); ++i) {
        const auto source = files.at(i);
        const auto status = source.at("status").str();
        if (status != "included" && status != "duplicate") continue;
        const fs::path path(source.at("path").str());
        const auto format = source.at("format").str();
        if (!safeRelative(path) || forbiddenModelExtension(path) || lower(format) == "onnx")
            fail("Package contains an ONNX model asset: " + path.generic_string());
    }
}
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
          const ModelPackagingObserver &observer, std::span<const fs::path> selected = {},
          std::span<const ModelPackagingAssignment> assignments = {}) {
    if (!fs::is_directory(input)) fail("Choose an existing model folder");
    Plan plan; plan.input = fs::absolute(input).lexically_normal(); plan.name = input.filename().string();
    std::vector<fs::path> paths;
    const auto internal = [](const fs::path &relative) {
        return lower(relative.filename().string()).ends_with(".iisacc.json")
            || std::any_of(relative.begin(),relative.end(),[](const auto &part){return part.string().starts_with(".");});
    };
    const auto collect = [&](const fs::path &directory, bool rejectRedirects) {
        for (const auto &entry : fs::recursive_directory_iterator(directory)) {
            check(stop);
            const auto relative = entry.path().lexically_relative(plan.input);
            if (internal(relative)) continue;
            if (rejectRedirects && entry.is_symlink()) fail("A selected model contains a redirected path: " + relative.generic_string());
            if (entry.is_regular_file()) paths.push_back(entry.path());
            if (paths.size() > 10'000) fail("Model selection contains more than 10,000 files");
        }
    };
    if (selected.empty()) collect(plan.input, false);
    else {
        if (fs::is_symlink(fs::symlink_status(plan.input))) fail("The model library is redirected");
        for (const auto &relative : selected) {
            check(stop);
            if (!safeRelative(relative) || internal(relative)) fail("Invalid selected model path: " + relative.generic_string());
            auto current = plan.input;
            for (const auto &part : relative) {
                current /= part;
                const auto status = fs::symlink_status(current);
                if (!fs::exists(status) || fs::is_symlink(status)) fail("Selected model is unavailable or redirected: " + relative.generic_string());
            }
            if (fs::is_directory(current)) collect(current, true);
            else if (fs::is_regular_file(current)) paths.push_back(current);
            else fail("Selected model is not a regular file or directory: " + relative.generic_string());
            if (paths.size() > 10'000) fail("Model selection contains more than 10,000 files");
        }
    }
    std::sort(paths.begin(),paths.end());
    paths.erase(std::unique(paths.begin(),paths.end()),paths.end());
    for (const auto &path : paths) {
        progress(observer,stop,"scan",path.filename().string(),plan.files.size(),paths.size());
        Source source; source.logical = path; source.actual = resolvedSource(path);
        source.relative = path.lexically_relative(plan.input).generic_string();
        if (!safeRelative(fs::path(source.relative))) fail("Invalid source path");
        source.size = fs::file_size(source.actual); source.time = fs::last_write_time(source.actual);
        const auto extension = lower(path.extension().string());
        try {
            if (forbiddenModelExtension(path)) fail("ONNX model files are not supported: " + source.relative);
            auto format = modelContentFormat(source.actual, source.size);
            if (format.empty()) source.shardIndex=hasSafetensorsIndexSuffix(source.relative) ||
                (looksLikeJsonObject(source.actual,source.size) && detectModelContainer(source.logical)=="safetensors-index");
            if (format.empty() && !source.shardIndex && (extension == ".safetensors" || extension == ".safetensor")) format = "safetensors";
            if (format.empty() && !source.shardIndex && extension == ".gguf") format = "gguf";
            if (format == "safetensors") {
                source.format = "safetensors"; readSafetensors(source);
                if (source.metadata.contains("iild_package_schema")) {
                    if (!source.metadata.contains("iild_package_manifest")) fail("Invalid nested model package");
                    requireNoOnnxManifest(parse(source.metadata.at("iild_package_manifest").str()));
                }
            } else if (format == "gguf") {
                source.format = "gguf"; readGGUF(source); blob(source);
            } else {
                static const std::set<std::string> assets{".json",".txt",".model",".tiktoken",".vocab",".merges",".yaml",".yml",".ini",".cfg",".bin",".pt",".pth",".ckpt",".npz",".md"};
                if (!source.shardIndex && !assets.contains(extension) && !has(lower(path.filename().string()),"license") && !has(lower(path.filename().string()),"notice")) {
                    source.status="ignored"; source.reason="Unsupported supporting file"; source.format="other";
                } else {
                    source.format="asset";
                    if (extension == ".json" || source.shardIndex) (void)parse(readSmall(path));
                    blob(source);
                }
            }
            source.role = inferRole(source); source.detectedRole=source.role; source.optional = source.role == "upscaler" || source.role == "lora";
            source.group = source.relative;
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
        else if (file.status == "invalid" && (forbiddenModelExtension(file.relative) || has(file.reason, "ONNX model asset")))
            plan.errors.push_back(file.reason);
    std::map<std::string,std::size_t> byPath;
    for (std::size_t i=0;i<plan.files.size();++i) byPath[plan.files[i].relative]=i;
    std::map<std::pair<fs::path,fs::path>,std::size_t> indexAliases;
    for (std::size_t i=0;i<plan.files.size();++i) {
        auto &candidate=plan.files[i];
        if (!candidate.shardIndex || candidate.status!="included") continue;
        const auto key=std::pair{candidate.logical.parent_path().lexically_normal(),candidate.actual};
        const auto [found,inserted]=indexAliases.emplace(key,i);
        if (inserted) continue;
        auto &prior=plan.files[found->second];
        if (!hasSafetensorsIndexSuffix(prior.relative) && hasSafetensorsIndexSuffix(candidate.relative)) {
            prior.shardIndex=false;prior.indexAlias=true;found->second=i;
        } else {candidate.shardIndex=false;candidate.indexAlias=true;}
    }
    for (auto &index : plan.files) if (index.shardIndex && index.status=="included") {
        try {
            const auto mapping=parse(readSmall(index.actual)).at("weight_map");
            std::map<std::size_t,std::set<std::string>> indexedNames;
            const auto entries=mapping.members();
            if (entries.empty() || entries.size()>1'000'000) fail("Shard index has an invalid tensor count: " + index.relative);
            for (const auto &[key,value] : entries) {
                const fs::path reference(value.str());
                if (!safeRelative(reference)) fail("Unsafe shard path in " + index.relative);
                const auto relative=(fs::path(index.relative).parent_path()/reference).generic_string();
                const auto found=byPath.find(relative);
                if (found==byPath.end() || plan.files[found->second].status!="included") fail("Missing or invalid shard: " + relative);
                auto &file=plan.files[found->second];
                if (file.group!=file.relative && file.group!=index.relative)
                    fail("A shard is claimed by multiple indexes: " + relative);
                file.group=index.relative;
                if (file.format!="safetensors" || std::none_of(file.tensors.begin(),file.tensors.end(),[&](const auto &t){return t.key==key;}))
                    fail("Shard is missing indexed tensor: " + key);
                indexedNames[found->second].insert(key);
            }
            for (const auto &[position,names]:indexedNames)
                if (names.size()!=plan.files[position].tensors.size())
                    fail("Shard contains unindexed tensors: " + plan.files[position].relative);
        } catch (const std::exception &error) { plan.errors.push_back(error.what()); }
    }
    static const std::set<std::string> allowedRoles{"model","text_encoder","video_vae","audio_vae","projections","upscaler","lora","other","high_noise_model","vision_encoder"};
    std::map<std::string,std::pair<std::string,std::string>> groupChoices;
    std::set<std::string> assignedPaths;
    for (const auto &assignment : assignments) {
        const auto path=assignment.relativePath.generic_string();
        if (!safeRelative(assignment.relativePath) || !assignedPaths.insert(path).second)
            fail("Invalid or repeated assignment path: " + path);
        const auto found=byPath.find(path);
        if (found==byPath.end()) fail("Assigned source is not selected: " + path);
        const auto &file=plan.files[found->second];
        if (file.status!="included" || (file.format!="safetensors" && file.format!="gguf"))
            fail("Assigned source is not a valid model file: " + path);
        if (!assignment.role.empty() && !allowedRoles.contains(assignment.role)) fail("Unknown model role: " + assignment.role);
        if (!assignment.position.empty() && assignment.position!="main" && assignment.position!="component")
            fail("Unknown model position: " + assignment.position);
        auto &choice=groupChoices[file.group];
        if (!assignment.role.empty()) {
            if (!choice.first.empty() && choice.first!=assignment.role) fail("Conflicting roles in a sharded model: " + file.group);
            choice.first=assignment.role;
        }
        if (!assignment.position.empty()) {
            if (!choice.second.empty() && choice.second!=assignment.position) fail("Conflicting positions in a sharded model: " + file.group);
            choice.second=assignment.position;
        }
    }
    // A Wan Diffusers index identifies its experts and, for image-to-video,
    // its vision encoder. File size cannot determine a component's role.
    std::map<std::string,std::string> indexedComponentRoles;
    std::string indexedMain;
    std::size_t wanIndexes=0;
    for (const auto &indexFile : plan.files) {
        if (fs::path(indexFile.relative).filename()!="model_index.json" || indexFile.status!="included") continue;
        const auto index=parse(readSmall(indexFile.actual));
        if (!index.contains("_class_name") || !index.at("_class_name").is(json_type_string)) continue;
        const auto pipeline=index.at("_class_name").str();
        if (pipeline!="WanPipeline" && pipeline!="WanImageToVideoPipeline") continue;
        const bool imageToVideo=pipeline=="WanImageToVideoPipeline";
        const auto declaresExpert=[&](const std::string &name) {
            if (!index.contains(name)) return false;
            const auto value=index.at(name);
            if (!value.is(json_type_array) || value.size()!=2) return false;
            auto *library=json_object_array_get_idx(value.get(),0), *type=json_object_array_get_idx(value.get(),1);
            return library && type && json_object_is_type(library,json_type_string) && json_object_is_type(type,json_type_string)
                && std::string(json_object_get_string(library))=="diffusers"
                && std::string(json_object_get_string(type))=="WanTransformer3DModel";
        };
        if (!declaresExpert("transformer")) {
            if (imageToVideo) plan.errors.push_back("Unsupported Wan transformer declaration");
            continue;
        }
        const bool dual=declaresExpert("transformer_2");
        if (!dual && !imageToVideo) continue;
        if (++wanIndexes>1) { plan.errors.push_back("Multiple Wan model indexes were selected"); continue; }
        if (index.contains("transformer_2") && !dual) {
            plan.errors.push_back("Unsupported Wan transformer_2 declaration");continue;
        }
        if (dual) {
            if (!index.contains("boundary_ratio") ||
                (!index.at("boundary_ratio").is(json_type_double) && !index.at("boundary_ratio").is(json_type_int))) {
                plan.errors.push_back("Dual-expert Wan model_index.json requires a numeric boundary_ratio");
                continue;
            }
            const auto boundary=json_object_get_double(index.at("boundary_ratio").get());
            if (!std::isfinite(boundary) || boundary<0 || boundary>1) {
                plan.errors.push_back("Wan boundary_ratio must be between 0 and 1");
                continue;
            }
            plan.wanMoeBoundary=boundary;
        } else if (index.contains("boundary_ratio")) {
            plan.errors.push_back("Wan boundary_ratio requires a second expert");continue;
        }
        const auto parent=fs::path(indexFile.relative).parent_path();
        const auto selectRole=[&](const std::string &directory,const std::string &role) {
            const auto prefix=(parent/directory).generic_string()+"/";
            std::set<std::string> candidates;
            for (const auto &file : plan.files)
                if (file.status=="included" && file.relative.starts_with(prefix)
                    && (file.format=="safetensors" || file.format=="gguf")) candidates.insert(file.group);
            if (candidates.size()!=1) {
                plan.errors.push_back("Wan "+directory+" must contain exactly one model weight group");
                return;
            }
            indexedComponentRoles.emplace(*candidates.begin(),role);
            if (role=="model") indexedMain=*candidates.begin();
        };
        selectRole("transformer",dual ? "high_noise_model" : "model");
        if (dual) selectRole("transformer_2","model");
        if (imageToVideo) {
            const auto visionDeclaration=[&](const char *name,const char *type) -> std::optional<bool> {
                if (!index.contains(name)) return {};
                const auto value=index.at(name);
                if (!value.is(json_type_array) || value.size()!=2) return {};
                auto *library=json_object_array_get_idx(value.get(),0), *declared=json_object_array_get_idx(value.get(),1);
                if (!library && !declared) return false;
                if (!library || !declared || !json_object_is_type(library,json_type_string) ||
                    !json_object_is_type(declared,json_type_string) ||
                    std::string(json_object_get_string(library))!="transformers" ||
                    std::string(json_object_get_string(declared))!=type) return {};
                return true;
            };
            const auto encoder=visionDeclaration("image_encoder","CLIPVisionModelWithProjection");
            const auto processor=visionDeclaration("image_processor","CLIPProcessor");
            if (!encoder || !processor || *encoder!=*processor)
                plan.errors.push_back("Invalid Wan image_encoder or image_processor declaration");
            else if (*encoder) selectRole("image_encoder","vision_encoder");
        }
    }
    std::string requestedMain=indexedMain;
    for (auto &file : plan.files) {
        const auto choice=groupChoices.find(file.group);
        const auto indexed=indexedComponentRoles.find(file.group);
        if (indexed!=indexedComponentRoles.end() && (file.format=="safetensors" || file.format=="gguf")) {
            const auto expectedPosition=indexed->second=="model" ? "main" : "component";
            if (choice!=groupChoices.end() && ((!choice->second.first.empty() && choice->second.first!=indexed->second)
                || (!choice->second.second.empty() && choice->second.second!=expectedPosition)))
                fail("Assignment conflicts with Wan model_index.json: "+file.relative);
            file.role=indexed->second; file.roleSource="model_index"; file.positionSource="model_index";
        }
        if (choice!=groupChoices.end() && (file.format=="safetensors" || file.format=="gguf")) {
            if (!choice->second.first.empty()) { file.role=choice->second.first; file.roleSource="user"; }
            if (!choice->second.second.empty()) {
                if (choice->second.second=="main" && file.role!="model") fail("Only a model can be the main position: " + file.relative);
                file.positionSource="user";
                if (choice->second.second=="main") {
                    if (!requestedMain.empty() && requestedMain!=file.group) fail("Only one model group can be the main model");
                    requestedMain=file.group;
                }
            }
        }
        file.optional=file.role=="upscaler" || file.role=="lora";
    }
    std::map<std::string,std::string> indexedRoles;
    std::set<std::string> ambiguousGroups;
    std::set<std::string> indexPaths;
    for (const auto &file:plan.files) if (file.shardIndex && file.status=="included") indexPaths.insert(file.relative);
    for (const auto &file : plan.files) {
        if (file.status!="included" || file.format!="safetensors" || file.group==file.relative
            || !indexPaths.contains(file.group)) continue;
        const auto [found,inserted]=indexedRoles.emplace(file.group,file.role);
        if (!inserted && found->second!=file.role) ambiguousGroups.insert(file.group);
    }
    for (const auto &group : ambiguousGroups)
        plan.errors.push_back("Inconsistent roles in sharded model; assign one shared role: " + group);
    std::set<std::string> excludedGroups;
    for (const auto &path : excluded) {
        const auto found=byPath.find(path);
        if (found==byPath.end()) continue;
        const auto &file=plan.files[found->second];
        if (!file.optional) plan.errors.push_back("Required component cannot be excluded: " + file.relative);
        else excludedGroups.insert(file.group);
    }
    for (auto &file : plan.files) {
        const bool shardIndex=file.format=="asset" && file.relative==file.group && file.shardIndex;
        if (file.status=="included" && excludedGroups.contains(file.group) && (file.optional || shardIndex)) {
            file.status="excluded"; file.reason="Optional component excluded";
        }
    }
    std::map<std::string,U64> groups;
    for (const auto &file : plan.files) if (file.status=="included" && file.role=="model" &&
        (!groupChoices.contains(file.group) || groupChoices.at(file.group).second!="component")) groups[file.group]=add(groups[file.group],file.size);
    std::string primaryGroup=requestedMain;
    if (primaryGroup.empty() && !groups.empty()) primaryGroup=std::max_element(groups.begin(),groups.end(),[](const auto &a,const auto &b){return a.second<b.second;})->first;
    for (const auto &file : plan.files) {
        if (file.status=="invalid" || !file.metadata.contains("model_version")) continue;
        if (!assignments.empty() && (file.role!="model" || file.group!=primaryGroup)) continue;
        const auto version=file.metadata.at("model_version").str();
        if ((has(lower(file.relative),"ltx") || has(lower(file.metadata.dump()),"ltx")) && version.starts_with("2.")) {
            plan.ltx=true; plan.family="LTX "+version.substr(0,version.find('.',version.find('.')+1));
            plan.name=plan.family+" "+(has(lower(plan.input.filename().string()),"uncensored")?"Uncensored":"Model");
        }
    }
    for (std::size_t i=0;i<plan.files.size();++i)
        if (plan.files[i].status=="included" && plan.files[i].role=="model" && plan.files[i].group==primaryGroup) plan.primary.push_back(i);
    const bool modelDeclared=std::any_of(plan.files.begin(),plan.files.end(),[](const auto &file) {
        return file.status=="included" && file.role=="model";
    });
    if (plan.primary.empty() && !plan.ltx && !modelDeclared) {
        for (std::size_t i=0;i<plan.files.size();++i) if (plan.files[i].status=="included" && plan.files[i].format=="gguf") {
            const auto choice=groupChoices.find(plan.files[i].group);
            if (choice!=groupChoices.end() && (!choice->second.first.empty() || choice->second.second=="component")) continue;
            plan.primary.push_back(i); plan.files[i].role="model"; plan.files[i].detectedRole="model"; break;
        }
    }
    if (plan.primary.empty()) plan.errors.push_back("No valid main model was found in this folder");
    for (const auto i : plan.primary) plan.files[i].position="main";
    detail::WanDiffusersSignature wanDiffusers;
    for (const auto i : plan.primary) if (plan.files[i].format=="safetensors")
        for (const auto &tensor : plan.files[i].tensors) wanDiffusers.add(tensor.key);
    const auto wanPrefix=wanDiffusers.prefix();
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
            auto key=file.format=="safetensors"?file.tensors[t].key:"__iild_assets__/model/"+file.relative;
            if (file.format=="safetensors" && wanPrefix)
                if (const auto mapped=detail::wanDiffusersTensorName(key,*wanPrefix)) key="model.diffusion_model."+*mapped;
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
    for (const auto &file : plan.files) if (fs::path(file.relative).filename()=="model_index.json" && file.status=="included") {
        const auto index=parse(readSmall(file.actual));
        if (index.contains("schema") && index.at("schema").is(json_type_string)
            && index.at("schema").str() == "iild-unified-model-v1") continue;
        const auto parent = fs::path(file.relative).parent_path();
        json_object_object_foreach(index.get(), memberName, memberValue) {
            if (!memberValue) continue;
            const std::string name(memberName);
            const J value(json_object_get(memberValue));
            if (name.starts_with("_") || !value.is(json_type_array) || value.size()!=2) continue;
            auto *classValue=json_object_array_get_idx(value.get(),1);
            if (!classValue || json_object_is_type(classValue,json_type_null)) continue;
            if (!safeRelative(fs::path(name)) || name.find('/') != std::string::npos)
                fail("Unsafe Diffusers component name: " + name);
            const bool weightsRequired = name == "unet" || name.starts_with("transformer") || name == "vae"
                || name.starts_with("text_encoder") || name.starts_with("controlnet");
            const auto componentPrefix = (parent / name).generic_string() + "/";
            const bool exists=std::any_of(plan.files.begin(),plan.files.end(),[&](const auto &f){
                if (!f.relative.starts_with(componentPrefix) || f.status!="included") return false;
                if (!weightsRequired) return true;
                const auto extension = lower(fs::path(f.relative).extension().string());
                return f.format == "safetensors" || f.format == "gguf" ||
                    (f.size && (extension == ".bin" || extension == ".pt" || extension == ".pth" || extension == ".ckpt"));
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
    if (plan.wanMoeBoundary) result.set("wan_moe_boundary",J(json_object_new_double(*plan.wanMoeBoundary)));
    for (const auto &source : plan.files) {
        auto file=J::object(), tensors=J::array();
        file.set("path",source.relative); file.set("format",source.format); file.set("role",source.role);
        file.set("group",source.group);
        if (source.shardIndex || source.indexAlias) file.set("shard_index",source.shardIndex);
        file.set("position",source.position); file.set("role_source",source.roleSource); file.set("position_source",source.positionSource);
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
        const auto label=file.role=="text_encoder"?"Text encoder":file.role=="upscaler"?"Spatial upscaler":file.role=="lora"?"LoRA":
            file.role=="model"?"Additional checkpoint":file.role=="video_vae"?"VAE / decoder":
            file.role=="audio_vae"?"Audio decoder":file.role=="projections"?"Text projections":file.relative;
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
        file.set("detected_role",source.detectedRole);
        file.set("position",source.position); file.set("group",source.group);
        file.set("role_source",source.roleSource); file.set("position_source",source.positionSource);
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
#ifdef _WIN32
    const auto pid = GetCurrentProcessId();
#else
    const auto pid = ::getpid();
#endif
    return target.parent_path()/("."+target.filename().string()+".partial-"+std::to_string(pid)+"-"+suffix);
}
void removeAbandonedAssets(const fs::path &target) {
    const auto prefix = "." + target.filename().string() + ".partial-";
    for (const auto &entry : fs::directory_iterator(target.parent_path())) {
        const auto name = entry.path().filename().string();
        if (!name.starts_with(prefix) || entry.is_symlink() || !entry.is_regular_file()) continue;
        const auto start = name.data() + prefix.size();
        const auto end = name.data() + name.size();
        std::uint32_t pid = 0;
        const auto parsed = std::from_chars(start, end, pid);
        if (parsed.ec != std::errc{} || !pid || parsed.ptr == end || *parsed.ptr != '-'
            || end - parsed.ptr != 33 || !std::all_of(parsed.ptr + 1, end, [](char ch) {
                return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'); })) continue;
#ifdef _WIN32
        const auto process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
        bool abandoned = !process && GetLastError() == ERROR_INVALID_PARAMETER;
        if (process) {
            DWORD code = STILL_ACTIVE;
            abandoned = GetExitCodeProcess(process, &code) && code != STILL_ACTIVE;
            CloseHandle(process);
        }
#else
        if (pid > static_cast<std::uint32_t>(std::numeric_limits<pid_t>::max())) continue;
        const bool abandoned = ::kill(static_cast<pid_t>(pid), 0) == -1 && errno == ESRCH;
#endif
        if (abandoned) { std::error_code ignored; fs::remove(entry.path(), ignored); }
    }
}
struct Temporary {
    fs::path path;
    bool directory=false;
    ~Temporary() { std::error_code error; if (directory) fs::remove_all(path,error); else fs::remove(path,error); }
};
void createExclusive(const fs::path &path) {
#ifdef _WIN32
    const auto handle=iiFileProvider::native::createFile(path.c_str(),GENERIC_WRITE,0,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
    if (handle==INVALID_HANDLE_VALUE) fail("Could not create temporary output"); CloseHandle(handle);
#else
    const int descriptor=iiFileProvider::native::openFile(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
    if (descriptor<0) fail("Could not create temporary output: "+path.string()); ::close(descriptor);
#endif
}
void syncFile(const fs::path &path) {
#ifdef _WIN32
    const auto handle=iiFileProvider::native::createFile(path.c_str(),GENERIC_WRITE,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
    if (handle==INVALID_HANDLE_VALUE) fail("Could not synchronize the output");
    const auto ok=FlushFileBuffers(handle); CloseHandle(handle); if (!ok) fail("Output synchronization failed");
#else
    const auto descriptor=iiFileProvider::native::openFile(path.c_str(),O_RDONLY);
    if (descriptor<0) fail("Could not synchronize the output");
    const auto result=::fsync(descriptor); ::close(descriptor); if (result!=0) fail("Output synchronization failed");
#endif
}
void commit(const fs::path &temporary,const fs::path &target) {
#ifdef _WIN32
    if (!iiFileProvider::native::moveFile(temporary.c_str(),target.c_str(),MOVEFILE_WRITE_THROUGH)) fail("Could not save without overwriting an existing output");
#elif defined(__APPLE__)
    if (::renamex_np(temporary.c_str(),target.c_str(),RENAME_EXCL)!=0) fail("Could not save without overwriting an existing output");
#else
    iiFileProvider::native::createHardLink(temporary,target); fs::remove(temporary);
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
    requireNoOnnxManifest(value.manifest);
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

detail::GgufCatalog detail::inspectGgufCatalog(const fs::path &path) {
    Source source;
    source.logical = fs::absolute(path); source.actual = resolvedSource(path);
    source.size = fs::file_size(source.actual); source.time = fs::last_write_time(source.actual);
    detail::GgufCatalog result;
    readGGUF(source, &result);
    return result;
}

namespace {
struct ObjectIdentity {
    std::string stamp;
    U64 size = 0;
    bool operator==(const ObjectIdentity &) const = default;
};
ObjectIdentity objectIdentity(const fs::path &path) {
    const auto size=fs::file_size(path);
    auto token=iiFileProvider::native::Presentation::cacheChangeToken(path);
#ifdef _WIN32
    const auto handle=iiFileProvider::native::createFile(path.c_str(),FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,nullptr,OPEN_EXISTING,FILE_FLAG_BACKUP_SEMANTICS,nullptr);
    if (handle==INVALID_HANDLE_VALUE) fail("Cannot identify a Safetensors source");
    struct Close {HANDLE value; ~Close(){CloseHandle(value);}} close{handle};
    BY_HANDLE_FILE_INFORMATION info{};
    if (!GetFileInformationByHandle(handle,&info)) fail("Cannot identify a Safetensors source");
    token += ':'+std::to_string(info.dwVolumeSerialNumber)+':'+std::to_string(info.nFileIndexHigh)+':'+std::to_string(info.nFileIndexLow);
#else
    struct stat info{};
    if (::stat(path.c_str(),&info)!=0) fail("Cannot identify a Safetensors source");
    token += ':'+std::to_string(info.st_dev)+':'+std::to_string(info.st_ino);
#endif
    token += ':'+std::to_string(static_cast<long long>(fs::last_write_time(path).time_since_epoch().count()));
    return {std::move(token),size};
}
fs::path indexUtf8Path(const std::string &value) {
    return fs::path(std::u8string(reinterpret_cast<const char8_t *>(value.data()), value.size()));
}
std::string indexUtf8(const fs::path &path) {
    const auto value=path.generic_u8string(); return {value.begin(),value.end()};
}
struct IndexDescription {
    fs::path source, resolved;
    ObjectIdentity identity;
    std::map<std::string,fs::path> weights;
    U64 metadataBytes = 0;
};
IndexDescription readIndex(const fs::path &input) {
    IndexDescription result;
    result.source=fs::absolute(input).lexically_normal();
    result.resolved=resolvedSource(result.source);
    // The index's selected path, not its resolved blob path, defines the
    // loader's relative shard namespace (e.g. a Hugging Face snapshot).
    result.identity=objectIdentity(result.resolved);
    const auto contents=readSmall(result.resolved);
    result.metadataBytes=contents.size();
    const auto mapping=parse(contents).at("weight_map");
    if (!mapping.is(json_type_object)) fail("Safetensors index weight_map must be an object");
    const auto entries=mapping.members();
    if (entries.empty() || entries.size()>1'000'000) fail("Safetensors index has an invalid tensor count");
    std::set<fs::path> uniqueShards;
    for (const auto &[name,value] : entries) {
        if (name.empty() || name.find('\0')!=std::string::npos) fail("Invalid indexed tensor name");
        const auto reference=indexUtf8Path(value.str());
        if (!safeRelative(reference) || forbiddenModelExtension(reference))
            fail("Unsafe or unsupported shard path in Safetensors index");
        result.weights.emplace(name,reference);
        uniqueShards.insert(reference);
        if (uniqueShards.size()>4096) fail("Safetensors index has too many shards");
    }
    if (resolvedSource(result.source)!=result.resolved ||
        objectIdentity(result.resolved)!=result.identity)
        fail("Safetensors index changed during inspection");
    return result;
}
}

std::vector<fs::path> safetensorsIndexShardPaths(const fs::path &input) {
    const auto source=fs::absolute(input).lexically_normal();
    const auto resolved=resolvedSource(source);
    const auto snapshot=objectIdentity(resolved);
    struct Cached { fs::path resolved; ObjectIdentity identity; std::vector<fs::path> paths; };
    static std::mutex mutex;
    static std::map<fs::path,Cached> cache;
    {
        std::lock_guard lock(mutex);
        const auto found=cache.find(source);
        if (found!=cache.end() && found->second.resolved==resolved && found->second.identity==snapshot)
            return found->second.paths;
    }
    const auto description=readIndex(source);
    std::set<fs::path> unique;
    for (const auto &[name,path] : description.weights) { (void)name; unique.insert(path); }
    std::vector<fs::path> paths;
    paths.reserve(unique.size());
    for (const auto &path : unique) paths.push_back(description.source.parent_path()/path);
    {
        std::lock_guard lock(mutex);
        if (cache.size()>=128 && !cache.contains(source)) cache.erase(cache.begin());
        cache[source]={description.resolved,description.identity,paths};
    }
    return paths;
}

bool verifySafetensorsIndex(const SafetensorsIndex &index) noexcept {
    try {
        if (resolvedSource(index.sourcePath)!=index.resolvedPath ||
            objectIdentity(index.resolvedPath)!=ObjectIdentity{index.identity,index.sizeBytes}) return false;
        for (const auto &shard : index.shards) {
            const auto logical=index.sourcePath.parent_path()/shard.relativePath;
            if (resolvedSource(logical)!=shard.resolvedPath ||
                objectIdentity(shard.resolvedPath)!=ObjectIdentity{shard.identity,shard.sizeBytes}) return false;
        }
        return true;
    } catch (...) { return false; }
}

SafetensorsIndex inspectSafetensorsIndex(const fs::path &input) {
    const auto description=readIndex(input);
    SafetensorsIndex result;
    result.sourcePath=description.source; result.resolvedPath=description.resolved;
    result.identity=description.identity.stamp; result.sizeBytes=description.identity.size;
    result.metadataBytesRead=description.metadataBytes;
    std::map<fs::path,std::set<std::string>> expected;
    for (const auto &[name,path] : description.weights) expected[path].insert(name);
    std::set<std::string> metadataConflicts;
    for (const auto &[relative,names] : expected) {
        Source shard;
        shard.logical=description.source.parent_path()/relative;
        shard.actual=resolvedSource(shard.logical);
        const auto identity=objectIdentity(shard.actual);
        shard.size=identity.size; shard.time=fs::last_write_time(shard.actual);
        readSafetensors(shard);
        result.metadataBytesRead=add(result.metadataBytesRead,8+shard.header.size());
        if (result.metadataBytesRead>256ull*1024*1024) fail("Combined shard headers exceed the inspection limit");
        if (shard.tensors.size()!=names.size()) fail("Shard index does not exactly cover its tensors");
        const auto shardNumber=result.shards.size();
        for (const auto &tensor : shard.tensors) {
            if (!names.contains(tensor.key)) fail("Shard contains an unindexed or wrongly indexed tensor: "+tensor.key);
            result.tensors.push_back({tensor.key,tensor.dtype,tensor.shape,shardNumber});
        }
        for (const auto &[key,value] : shard.metadata.members()) {
            const auto text=value.str();
            if (metadataConflicts.contains(key)) continue;
            const auto [found,inserted]=result.sharedMetadata.emplace(key,text);
            if (!inserted && found->second!=text) {result.sharedMetadata.erase(found);metadataConflicts.insert(key);}
        }
        result.shards.push_back({relative,shard.actual,identity.stamp,identity.size});
    }
    if (result.tensors.size()!=description.weights.size() || !verifySafetensorsIndex(result))
        fail("Safetensors index or a shard changed during inspection");
    return result;
}

std::string safetensorsIndexJson(const SafetensorsIndex &index) {
    auto result=J::object(); result.set("schema","iild-safetensors-index-v1");
    result.set("path",indexUtf8(index.sourcePath)); result.set("resolved_path",indexUtf8(index.resolvedPath));
    result.set("identity",index.identity); result.set("size_bytes",index.sizeBytes);
    result.set("metadata_bytes_read",index.metadataBytesRead);
    auto shards=J::array();
    for (const auto &shard : index.shards) {
        auto value=J::object(); value.set("relative_path",indexUtf8(shard.relativePath));
        value.set("resolved_path",indexUtf8(shard.resolvedPath)); value.set("identity",shard.identity);
        value.set("size_bytes",shard.sizeBytes); shards.push(value);
    }
    result.set("shards",shards);
    auto tensors=J::array();
    for (const auto &tensor : index.tensors) {
        auto value=J::object(); value.set("name",tensor.name); value.set("dtype",tensor.dtype);
        value.set("shard",static_cast<U64>(tensor.shard)); auto shape=J::array();
        for (const auto dimension : tensor.shape) shape.push(dimension);
        value.set("shape",shape); tensors.push(value);
    }
    result.set("tensors",tensors);
    auto metadata=J::object();
    for (const auto &[key,value] : index.sharedMetadata) metadata.set(key,value);
    result.set("metadata",metadata);
    return result.dump();
}

bool verifyPyTorchIndex(const PyTorchIndex &index) noexcept {
    try {
        if (resolvedSource(index.sourcePath)!=index.resolvedPath ||
            objectIdentity(index.resolvedPath)!=ObjectIdentity{index.identity,index.sizeBytes}) return false;
        for (const auto &shard : index.shards) {
            const auto logical=index.sourcePath.parent_path()/shard.relativePath;
            if (resolvedSource(logical)!=shard.resolvedPath ||
                objectIdentity(shard.resolvedPath)!=ObjectIdentity{shard.identity,shard.sizeBytes}) return false;
        }
        return true;
    } catch (...) { return false; }
}

PyTorchIndex inspectPyTorchIndex(const fs::path &input) {
    const auto description=readIndex(input);
    PyTorchIndex result;
    result.sourcePath=description.source;result.resolvedPath=description.resolved;
    result.identity=description.identity.stamp;result.sizeBytes=description.identity.size;
    result.metadataBytesRead=description.metadataBytes;
    std::map<fs::path,std::size_t> shardNumbers;
    for (const auto &[name,relative] : description.weights) {
        auto found=shardNumbers.find(relative);
        if (found==shardNumbers.end()) {
            const auto logical=description.source.parent_path()/relative;
            const auto resolved=resolvedSource(logical);
            const auto identity=objectIdentity(resolved);
            if (!modelContentFormat(resolved,identity.size).empty() || !detail::isPyTorchCandidate(resolved,identity.size))
                fail("Indexed shard is not a PyTorch checkpoint candidate");
            const auto number=result.shards.size();
            result.shards.push_back({relative,resolved,identity.stamp,identity.size});
            found=shardNumbers.emplace(relative,number).first;
        }
        result.tensors.push_back({name,found->second});
    }
    if (!verifyPyTorchIndex(result)) fail("PyTorch index or a shard changed during inspection");
    return result;
}

std::string pyTorchIndexJson(const PyTorchIndex &index) {
    auto result=J::object();result.set("schema","iild-pytorch-index-v1");
    result.set("path",indexUtf8(index.sourcePath));result.set("resolved_path",indexUtf8(index.resolvedPath));
    result.set("identity",index.identity);result.set("size_bytes",index.sizeBytes);
    result.set("metadata_bytes_read",index.metadataBytesRead);
    auto shards=J::array();
    for (const auto &shard : index.shards) {
        auto value=J::object();value.set("relative_path",indexUtf8(shard.relativePath));
        value.set("resolved_path",indexUtf8(shard.resolvedPath));value.set("identity",shard.identity);
        value.set("size_bytes",shard.sizeBytes);shards.push(value);
    }
    result.set("shards",shards);
    auto tensors=J::array();
    for (const auto &tensor : index.tensors) {
        auto value=J::object();value.set("name",tensor.name);value.set("shard",static_cast<U64>(tensor.shard));tensors.push(value);
    }
    result.set("tensors",tensors);return result.dump();
}

WanPipelineDirectoryRoute inspectWanPipelineDirectory(const fs::path &directory) {
    WanPipelineDirectoryRoute pipeline;
    auto &route=pipeline.common;
    route.directory=fs::absolute(directory).lexically_normal();
    if (!fs::is_directory(route.directory)) fail("Wan pipeline requires a local directory");
    route.manifest=route.directory/"model_index.json";
    const auto manifestResolved=resolvedSource(route.manifest);
    const auto manifestBefore=objectIdentity(manifestResolved);
    if (manifestBefore.size>1024*1024) fail("Wan model_index.json exceeds the 1 MiB limit");
    const auto index=parse(readSmall(manifestResolved));
    if (!index.is(json_type_object) || !index.contains("_class_name"))
        fail("The directory is not a supported Wan pipeline");
    pipeline.pipeline=index.at("_class_name").str();
    if (pipeline.pipeline!="WanPipeline" && pipeline.pipeline!="WanImageToVideoPipeline")
        fail("The directory is not a supported Wan pipeline");
    const bool imageToVideo=pipeline.pipeline=="WanImageToVideoPipeline";
    const auto declared=[&](const char *name,const char *library,const std::set<std::string> &types) {
        if (!index.contains(name)) fail("Missing Wan component: "+std::string(name));
        const auto component=index.at(name);
        if (!component.is(json_type_array) || component.size()!=2 || component.at(0).str()!=library
            || !types.contains(component.at(1).str()))
            fail("Unsupported Wan component: "+std::string(name));
        return component.at(1).str();
    };
    route.scheduler=declared("scheduler","diffusers",{"UniPCMultistepScheduler","FlowMatchEulerDiscreteScheduler"});
    (void)declared("text_encoder","transformers",{"UMT5EncoderModel"});
    (void)declared("tokenizer","transformers",{"T5Tokenizer","T5TokenizerFast"});
    (void)declared("transformer","diffusers",{"WanTransformer3DModel"});
    (void)declared("vae","diffusers",{"AutoencoderKLWan"});
    const auto nullableDeclared=[&](const char *name,const char *library,const char *type) {
        if (!index.contains(name)) fail("Missing Wan component: "+std::string(name));
        const auto value=index.at(name);
        if (!value.is(json_type_array) || value.size()!=2) fail("Unsupported Wan component: "+std::string(name));
        auto *kind=json_object_array_get_idx(value.get(),0);
        auto *className=json_object_array_get_idx(value.get(),1);
        if (!kind && !className) return false;
        if (!kind || !className || !json_object_is_type(kind,json_type_string) ||
            !json_object_is_type(className,json_type_string) || std::string(json_object_get_string(kind))!=library ||
            std::string(json_object_get_string(className))!=type)
            fail("Unsupported Wan component: "+std::string(name));
        return true;
    };
    bool vision=false;
    if (imageToVideo) {
        vision=nullableDeclared("image_encoder","transformers","CLIPVisionModelWithProjection");
        if (nullableDeclared("image_processor","transformers","CLIPProcessor")!=vision)
            fail("Wan image_encoder and image_processor must both be present or both be null");
    }
    const bool dual=index.contains("transformer_2");
    if (dual) {
        (void)declared("transformer_2","diffusers",{"WanTransformer3DModel"});
        if (!index.contains("boundary_ratio")) fail("Dual-expert Wan requires boundary_ratio");
        const auto boundary=index.at("boundary_ratio");
        if (!boundary.is(json_type_double) && !boundary.is(json_type_int)) fail("Invalid Wan boundary_ratio");
        route.moeBoundary=json_object_get_double(boundary.get());
        if (!std::isfinite(*route.moeBoundary) || *route.moeBoundary<=0 || *route.moeBoundary>=1)
            fail("Wan boundary_ratio must be between 0 and 1");
    } else if (index.contains("boundary_ratio")) fail("Wan boundary_ratio requires a second transformer");
    std::set<std::string> allowed{"scheduler","text_encoder","tokenizer","transformer","transformer_2","vae","boundary_ratio"};
    if (imageToVideo) {allowed.insert("image_encoder");allowed.insert("image_processor");}
    json_object_object_foreach(index.get(),key,value) {
        (void)value;
        if (key[0]!='_' && !allowed.contains(key)) fail("Unsupported Wan pipeline setting: "+std::string(key));
    }
    struct Snapshot {fs::path source,resolved; ObjectIdentity identity;};
    std::vector<Snapshot> snapshots{{route.manifest,manifestResolved,manifestBefore}};
    const auto component=[&](const std::string &name) {
        const auto folder=route.directory/name;
        if (!fs::is_directory(folder)) fail("Missing Wan component directory: "+name);
        struct Candidate {fs::path source; std::optional<SafetensorsIndex> index;};
        std::vector<Candidate> candidates;
        std::set<fs::path> indexedShards;
        std::size_t entries=0;
        for (const auto &entry:fs::directory_iterator(folder)) {
            if (++entries>10000) fail("Too many files in Wan component: "+name);
            if (!entry.is_regular_file() || forbiddenModelExtension(entry.path())) continue;
            const auto path=fs::absolute(entry.path()).lexically_normal();
            const auto format=detectModelContainer(path);
            if (format=="safetensors-index" ||
                (hasSafetensorsIndexSuffix(indexUtf8(path.filename())) && format!="safetensors" && format!="gguf")) {
                auto shardIndex=inspectSafetensorsIndex(path);
                for (const auto &shard:shardIndex.shards) indexedShards.insert(shard.resolvedPath);
                // A snapshot may expose both an index and a same-directory
                // link to it. They have the same logical shard base and are
                // one group; prefer the conventional public index name.
                auto same=std::find_if(candidates.begin(),candidates.end(),[&](const Candidate &candidate) {
                    return candidate.index && candidate.source.parent_path()==path.parent_path() &&
                        candidate.index->resolvedPath==shardIndex.resolvedPath;
                });
                if (same==candidates.end()) candidates.push_back({path,std::move(shardIndex)});
                else if (!hasSafetensorsIndexSuffix(indexUtf8(same->source.filename())) &&
                         hasSafetensorsIndexSuffix(indexUtf8(path.filename()))) *same={path,std::move(shardIndex)};
            } else if (format=="safetensors" || format=="gguf") candidates.push_back({path,{}});
        }
        candidates.erase(std::remove_if(candidates.begin(),candidates.end(),[&](const Candidate &candidate) {
            return !candidate.index && indexedShards.contains(resolvedSource(candidate.source));
        }),candidates.end());
        if (candidates.empty()) fail("Missing Wan weight group for "+name);
        if (candidates.size()!=1) fail("Wan "+name+" has multiple ambiguous weight groups");
        auto &chosen=candidates.front();
        WanDirectoryComponent result;
        result.sourcePath=chosen.source; result.resolvedPath=resolvedSource(chosen.source);
        result.sharded=chosen.index.has_value();
        const auto remember=[&](const fs::path &path) {
            const auto resolved=resolvedSource(path);
            snapshots.push_back({path,resolved,objectIdentity(resolved)});
            result.dependencies.push_back(path);
        };
        remember(chosen.source);
        if (chosen.index) {
            for (const auto &shard:chosen.index->shards)
                remember(chosen.index->sourcePath.parent_path()/shard.relativePath);
            if (!verifySafetensorsIndex(*chosen.index)) fail("Wan component changed during inspection: "+name);
        }
        return result;
    };
    route.model=component(dual?"transformer_2":"transformer");
    if (dual) route.highNoiseModel=component("transformer");
    route.textEncoder=component("text_encoder");
    route.videoVae=component("vae");
    if (vision) pipeline.visionEncoder=component("image_encoder");
    for (const auto &item:snapshots)
        if (resolvedSource(item.source)!=item.resolved || objectIdentity(item.resolved)!=item.identity)
            fail("Wan pipeline changed during inspection");
    return pipeline;
}

WanDirectoryRoute inspectWanDirectory(const fs::path &directory) {
    auto result=inspectWanPipelineDirectory(directory);
    if (result.pipeline!="WanPipeline") fail("The directory is not a supported Wan text-to-video pipeline");
    return std::move(result.common);
}

std::string wanDirectoryJson(const WanDirectoryRoute &route) {
    auto result=J::object(); result.set("schema","iild-wan-directory-v1");
    result.set("directory",indexUtf8(route.directory)); result.set("manifest",indexUtf8(route.manifest));
    result.set("scheduler",route.scheduler);
    const auto executionPath=[](const WanDirectoryComponent &component) -> const fs::path & {
        return component.sharded ? component.sourcePath : component.resolvedPath;
    };
    result.set("model",indexUtf8(executionPath(route.model)));
    result.set("text_encoder",indexUtf8(executionPath(route.textEncoder)));
    result.set("video_vae",indexUtf8(executionPath(route.videoVae)));
    if (route.highNoiseModel) result.set("high_noise_model",indexUtf8(executionPath(*route.highNoiseModel)));
    else json_object_object_add(result.get(),"high_noise_model",nullptr);
    if (route.moeBoundary) json_object_object_add(result.get(),"moe_boundary",json_object_new_double(*route.moeBoundary));
    else json_object_object_add(result.get(),"moe_boundary",nullptr);
    std::set<std::string> dependencies{indexUtf8(route.manifest)};
    const auto add=[&](const WanDirectoryComponent &component) {
        for (const auto &path:component.dependencies) dependencies.insert(indexUtf8(path));
    };
    add(route.model); add(route.textEncoder); add(route.videoVae);
    if (route.highNoiseModel) add(*route.highNoiseModel);
    auto paths=J::array(); for (const auto &path:dependencies) paths.push(path);
    result.set("dependencies",paths);
    return result.dump();
}

std::string wanPipelineDirectoryJson(const WanPipelineDirectoryRoute &route) {
    auto result=parse(wanDirectoryJson(route.common));
    result.set("pipeline",route.pipeline);
    if (route.visionEncoder) {
        const auto &vision=*route.visionEncoder;
        result.set("vision_encoder",indexUtf8(vision.sharded ? vision.sourcePath : vision.resolvedPath));
        std::set<std::string> dependencies;
        const auto current=result.at("dependencies");
        for (std::size_t i=0;i<current.size();++i) dependencies.insert(current.at(i).str());
        for (const auto &path:vision.dependencies) dependencies.insert(indexUtf8(path));
        auto paths=J::array();for (const auto &path:dependencies) paths.push(path);
        result.set("dependencies",paths);
    } else json_object_object_add(result.get(),"vision_encoder",nullptr);
    return result.dump();
}

ModelTensorProbe probeModelTensorNames(const fs::path &input,std::span<const std::string_view> names) {
    if (names.empty() || names.size()>4096) fail("Choose between one and 4096 tensor names");
    std::set<std::string,std::less<>> missing;
    for (const auto name:names) {
        if (name.empty() || name.size()>4096 || name.find('\0')!=name.npos)
            fail("Invalid tensor name");
        missing.emplace(name);
    }
    if (forbiddenModelExtension(input)) fail("ONNX model files are not supported");
    const auto logical=fs::absolute(input).lexically_normal();
    const auto resolved=resolvedSource(logical);
    const auto before=objectIdentity(resolved);
    const auto format=detectModelContainer(logical);
    ModelTensorProbe result;
    if (format=="safetensors-index") {
        const auto index=inspectSafetensorsIndex(logical);
        for (const auto &tensor:index.tensors) missing.erase(tensor.name);
        if (!verifySafetensorsIndex(index)) fail("Model shards changed during tensor inspection");
    } else if (format=="safetensors") {
        Source source;source.logical=logical;source.actual=resolved;source.size=before.size;
        source.time=fs::last_write_time(resolved);readSafetensors(source,nullptr,true);
        result.embeddedPackage=source.metadata.contains("iild_package_schema");
        for (const auto &tensor:source.tensors) missing.erase(tensor.key);
    } else if (format=="gguf") {
        const auto catalog=detail::inspectGgufCatalog(resolved);
        for (const auto &name:catalog.tensorNames) missing.erase(name);
    } else fail("Unsupported model container for tensor inspection");
    if (resolvedSource(logical)!=resolved || objectIdentity(resolved)!=before)
        fail("Model changed during tensor inspection");
    std::set<std::string,std::less<>> found;
    for (const auto name:names) if (!missing.contains(name)) found.emplace(name);
    result.matches.assign(found.begin(),found.end());
    return result;
}

std::vector<std::string> findModelTensorNames(const fs::path &input,std::span<const std::string_view> names) {
    return probeModelTensorNames(input,names).matches;
}

bool modelContainsTensorNames(const fs::path &input,std::span<const std::string_view> names) {
    const auto found=findModelTensorNames(input,names);
    return std::all_of(names.begin(),names.end(),[&](const auto name) {
        return std::binary_search(found.begin(),found.end(),name);
    });
}

std::string detectModelContainer(const fs::path &path) noexcept {
    try {
        std::error_code error;
        if (!fs::is_regular_file(path, error) || error) return "unknown";
        const auto size = fs::file_size(path, error);
        if (error) return "unknown";
        auto result = modelContentFormat(path, size);
        if (!result.empty()) return result;
        if (detail::isPyTorchCandidate(path,size)) return "pytorch";
        const bool conventional=hasSafetensorsIndexSuffix(path.filename().string());
        if (conventional || looksLikeJsonObject(path,size)) {
            const auto shards=safetensorsIndexShardPaths(path);
            bool safe=true,torch=true,missing=false,sawTorch=false,invalid=false;
            for (const auto &shard:shards) {
                const auto bytes=fs::file_size(shard,error);
                if (error || !fs::is_regular_file(shard,error) || error) {
                    error.clear();safe=false;torch=false;missing=true;continue;
                }
                const auto format=modelContentFormat(shard,bytes);
                if (format=="safetensors") torch=false;
                else if (format.empty() && detail::isPyTorchCandidate(shard,bytes)) {safe=false;sawTorch=true;}
                else {safe=false;torch=false;invalid=true;}
            }
            if (safe) return "safetensors-index";
            if (torch) return "pytorch-index";
            // Preserve the conventional incomplete-Safetensors diagnostic,
            // without treating an incomplete or mixed PyTorch index as safe.
            if (conventional && missing && !sawTorch && !invalid) return "safetensors-index";
        }
        return "unknown";
    } catch (...) {
        return "unknown";
    }
}

std::string scanModelFolder(const fs::path &input,std::span<const std::string> excluded,
                           std::stop_token stop,const ModelPackagingObserver &observer) {
    return report(scan(input,excluded,stop,observer),"scan").dump();
}
std::string scanModelSelection(const fs::path &root,std::span<const fs::path> selected,
                               std::span<const std::string> excluded,std::stop_token stop,
                               const ModelPackagingObserver &observer) {
    if (selected.empty()) fail("Choose at least one model from the library");
    return report(scan(root,excluded,stop,observer,selected),"scan").dump();
}
std::string scanModelFolderWithAssignments(const fs::path &input,std::span<const ModelPackagingAssignment> assignments,
    std::span<const std::string> excluded,std::stop_token stop,const ModelPackagingObserver &observer) {
    return report(scan(input,excluded,stop,observer,{},assignments),"scan").dump();
}
std::string scanModelSelectionWithAssignments(const fs::path &root,std::span<const fs::path> selected,
    std::span<const ModelPackagingAssignment> assignments,std::span<const std::string> excluded,
    std::stop_token stop,const ModelPackagingObserver &observer) {
    if (selected.empty()) fail("Choose at least one model from the library");
    return report(scan(root,excluded,stop,observer,selected,assignments),"scan").dump();
}
static std::string writePackage(Plan plan,const fs::path &output,std::stop_token stop,
                                const ModelPackagingObserver &observer) {
    const auto target=fs::absolute(output).lexically_normal();
    if (target.extension() != ".safetensors") fail("Model package output must use the .safetensors extension");
    destination(target,0);
    if (!plan.errors.empty()) fail(plan.errors.front());
    auto header=outputHeader(plan);
    destination(target,add(add(8,header.size()),plan.payload));
    Temporary temporary{temporaryPath(target)};
    createExclusive(temporary.path);
    iiFileProvider::native::ReadWriteFile file(temporary.path,std::ios::binary|std::ios::in|std::ios::out);
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
std::string createModelPackage(const fs::path &input,const fs::path &output,std::span<const std::string> excluded,
                              std::stop_token stop,const ModelPackagingObserver &observer) {
    const auto target=fs::absolute(output).lexically_normal();
    if (target.extension() != ".safetensors") fail("Model package output must use the .safetensors extension");
    destination(target,0);
    return writePackage(scan(input,excluded,stop,observer),target,stop,observer);
}
std::string createModelPackageFromSelection(const fs::path &root,std::span<const fs::path> selected,
                                           const fs::path &output,std::span<const std::string> excluded,
                                           std::stop_token stop,const ModelPackagingObserver &observer) {
    if (selected.empty()) fail("Choose at least one model from the library");
    const auto target=fs::absolute(output).lexically_normal();
    if (target.extension() != ".safetensors") fail("Model package output must use the .safetensors extension");
    destination(target,0);
    return writePackage(scan(root,excluded,stop,observer,selected),target,stop,observer);
}
std::string createModelPackageWithAssignments(const fs::path &input,const fs::path &output,
    std::span<const ModelPackagingAssignment> assignments,std::span<const std::string> excluded,
    std::stop_token stop,const ModelPackagingObserver &observer) {
    return writePackage(scan(input,excluded,stop,observer,{},assignments),output,stop,observer);
}
std::string createModelPackageFromSelectionWithAssignments(const fs::path &root,std::span<const fs::path> selected,
    const fs::path &output,std::span<const ModelPackagingAssignment> assignments,std::span<const std::string> excluded,
    std::stop_token stop,const ModelPackagingObserver &observer) {
    if (selected.empty()) fail("Choose at least one model from the library");
    return writePackage(scan(root,excluded,stop,observer,selected,assignments),output,stop,observer);
}
std::string verifyModelPackage(const fs::path &input,std::stop_token stop,const ModelPackagingObserver &observer) {
    if (forbiddenModelExtension(input)) fail("ONNX model files are not supported");
    const auto verified=verify(input,stop,observer);
    auto result=J::object(); result.set("schema",reportSchema); result.set("operation","verify");
    result.set("verified",true); result.set("output",fs::absolute(input).string());
    result.set("size",verified.file.size); result.set("sha256",verified.digest); return result.dump();
}
std::string extractModelPackage(const fs::path &input,const fs::path &output,std::stop_token stop,
                               const ModelPackagingObserver &observer) {
    if (forbiddenModelExtension(input)) fail("ONNX model files are not supported");
    const auto target=fs::absolute(output).lexically_normal(); destination(target,0);
    const auto verified=verify(input,stop,observer); const auto files=verified.manifest.at("files");
    U64 total=0;
    for (std::size_t i=0;i<files.size();++i) {
        const auto file=files.at(i);
        const auto status=file.at("status").str();
        if (status=="included" || status=="duplicate") total=add(total,file.at("size").number());
    }
    destination(target,total); Temporary temporary{temporaryPath(target),true};
    iiFileProvider::native::createDirectory(temporary.path);
    std::map<std::string,const Tensor *> tensors;
    for (const auto &tensor : verified.file.tensors) tensors[tensor.key]=&tensor;
    std::ifstream source(input,std::ios::binary); std::vector<char> buffer(chunkSize); U64 done=0,countFiles=0;
    for (std::size_t i=0;i<files.size();++i) {
        const auto record=files.at(i);
        const auto status=record.at("status").str();
        if (status!="included" && status!="duplicate") continue;
        const auto path=temporary.path/fs::path(record.at("path").str()); iiFileProvider::native::createDirectories(path.parent_path());
        iiFileProvider::native::OutputFile file(path,std::ios::binary); Hash hash;
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
    if (!iiFileProvider::native::moveFile(temporary.path.c_str(),target.c_str(),MOVEFILE_WRITE_THROUGH)) fail("Could not save extracted directory");
#elif defined(__APPLE__)
    if (::renamex_np(temporary.path.c_str(),target.c_str(),RENAME_EXCL)!=0) fail("Could not save extracted directory");
#else
    // renameat2 is not available on every Unix platform; an empty reserved
    // directory prevents accidental replacement of existing user data.
    if (!iiFileProvider::native::createDirectory(target)) fail("Extraction destination already exists");
    try { iiFileProvider::native::renameFile(temporary.path,target); } catch (...) { fs::remove(target); throw; }
#endif
    auto result=J::object(); result.set("schema",reportSchema); result.set("operation","extract"); result.set("verified",true);
    result.set("output",target.string()); result.set("file_count",countFiles); return result.dump();
}

namespace {
struct WeightHealth {
    U64 checkedTensors=0, checkedValues=0, nonfinite=0, uncheckedTensors=0;
    std::set<std::string> uncheckedTypes;
};

WeightHealth weightHealth(const fs::path &input,bool repair,std::stop_token stop,
                          const ModelPackagingObserver &observer) {
    Source source;
    source.logical=fs::absolute(input); source.actual=resolvedSource(input);
    source.size=fs::file_size(source.actual); source.time=fs::last_write_time(source.actual);
    readSafetensors(source);
    auto mode=std::ios::binary|std::ios::in;
    if (repair) mode|=std::ios::out;
    std::fstream file(source.actual,mode);
    if (!file) fail("Could not open model weights");
    std::vector<char> buffer(chunkSize);
    WeightHealth result;
    for (const auto &tensor:source.tensors) {
        unsigned width=0; U64 exponentMask=0;
        if (tensor.dtype=="F16") { width=2; exponentMask=0x7c00; }
        else if (tensor.dtype=="BF16") { width=2; exponentMask=0x7f80; }
        else if (tensor.dtype=="F32") { width=4; exponentMask=0x7f800000; }
        else if (tensor.dtype=="F64") { width=8; exponentMask=0x7ff0000000000000ULL; }
        else { ++result.uncheckedTensors; result.uncheckedTypes.insert(tensor.dtype); continue; }
        ++result.checkedTensors;
        const auto bytes=tensor.end-tensor.begin;
        result.checkedValues=add(result.checkedValues,bytes/width);
        for (U64 pos=0;pos<bytes;) {
            progress(observer,stop,"weight-health",tensor.key,pos,bytes);
            const auto count=static_cast<std::size_t>(std::min<U64>(bytes-pos,buffer.size()));
            file.clear(); file.seekg(static_cast<std::streamoff>(source.base+tensor.begin+pos));
            if (!file.read(buffer.data(),static_cast<std::streamsize>(count))) fail("Could not read model weights");
            bool changed=false;
            for (std::size_t i=0;i<count;i+=width) {
                U64 bits=0;
                for (unsigned j=0;j<width;++j)
                    bits|=static_cast<U64>(static_cast<unsigned char>(buffer[i+j]))<<(8*j);
                if ((bits&exponentMask)==exponentMask) {
                    result.nonfinite=add(result.nonfinite,1);
                    if (repair) { std::fill_n(buffer.data()+i,width,0); changed=true; }
                }
            }
            if (changed) {
                file.clear(); file.seekp(static_cast<std::streamoff>(source.base+tensor.begin+pos));
                file.write(buffer.data(),static_cast<std::streamsize>(count));
                if (!file) fail("Could not repair temporary model weights");
            }
            pos+=count;
        }
    }
    if (repair) { file.flush(); if (!file) fail("Could not finalize temporary model weights"); }
    else stamp(source);
    check(stop);
    return result;
}
}

std::string inspectModelWeights(const fs::path &input,std::stop_token stop,const ModelPackagingObserver &observer) {
    const auto health=weightHealth(input,false,stop,observer);
    auto result=J::object(), types=J::array();
    for (const auto &type:health.uncheckedTypes) types.push(type);
    result.set("schema",reportSchema); result.set("operation","weight-health");
    result.set("checked_tensors",health.checkedTensors); result.set("checked_values",health.checkedValues);
    result.set("nonfinite_values",health.nonfinite); result.set("unchecked_tensors",health.uncheckedTensors);
    result.set("unchecked_dtypes",types); return result.dump();
}

std::string calibrateModelFile(const fs::path &input,const fs::path &output,std::stop_token stop,
                               const ModelPackagingObserver &observer) {
    const auto target=fs::absolute(output).lexically_normal();
    if (forbiddenModelExtension(input) || target.extension()!=".safetensors")
        fail("Choose a Safetensors model and a new .safetensors output");
    Source source;
    source.logical=fs::absolute(input); source.actual=resolvedSource(input);
    source.size=fs::file_size(source.actual); source.time=fs::last_write_time(source.actual);
    readSafetensors(source);
    if (source.metadata.contains("iild_package_schema")) fail("Use package calibration for packaged models");
    destination(target,source.size);
    Temporary temporary{temporaryPath(target)}; createExclusive(temporary.path);
    std::ifstream in(source.actual,std::ios::binary);
    iiFileProvider::native::OutputFile out(temporary.path,std::ios::binary);
    if (!in || !out) fail("Could not open calibration files");
    std::vector<char> buffer(chunkSize);
    for (U64 done=0;done<source.size;) {
        progress(observer,stop,"copy","Copying model weights",done,source.size);
        const auto count=static_cast<std::size_t>(std::min<U64>(source.size-done,buffer.size()));
        if (!in.read(buffer.data(),static_cast<std::streamsize>(count))) fail("Could not read source model");
        out.write(buffer.data(),static_cast<std::streamsize>(count));
        if (!out) fail("Could not write the corrected model");
        done+=count;
    }
    out.close(); stamp(source);
    const auto corrected=weightHealth(temporary.path,true,stop,observer);
    const auto verified=weightHealth(temporary.path,false,stop,observer);
    if (verified.nonfinite) fail("Numeric verification failed");
    const auto digest=hashRange(temporary.path,0,source.size,stop,observer,"Hashing corrected model");
    check(stop); syncFile(temporary.path); commit(temporary.path,target);
    auto result=J::object(); result.set("schema",reportSchema); result.set("operation","calibrate-file");
    result.set("verified",true); result.set("output",target.string()); result.set("sha256",digest);
    result.set("weights_modified",corrected.nonfinite!=0);
    result.set("nonfinite_values_replaced",corrected.nonfinite);
    result.set("unchecked_tensors",corrected.uncheckedTensors); return result.dump();
}

std::string calibrateModelPackage(const fs::path &input,const fs::path &output,std::stop_token stop,
                                  const ModelPackagingObserver &observer,ModelCalibrationOptions options) {
    const auto target = fs::absolute(output).lexically_normal();
    if (forbiddenModelExtension(input) || target.extension() != ".safetensors")
        fail("Choose a model package and a new .safetensors output");
    destination(target,0);
    const auto workspace = temporaryPath(target);
    if (!fs::create_directory(workspace)) fail("Could not create the calibration workspace");
    Temporary temporary{workspace,true};
    Source packaged;
    packaged.actual=resolvedSource(input); packaged.size=fs::file_size(packaged.actual);
    readSafetensors(packaged);
    if (!packaged.metadata.contains("iild_package_manifest")) fail("Model package manifest is missing");
    const auto previous=parse(packaged.metadata.at("iild_package_manifest").str());
    std::vector<ModelPackagingAssignment> assignments;
    const auto previousFiles=previous.at("files");
    for (std::size_t i=0;i<previousFiles.size();++i) {
        const auto file=previousFiles.at(i);
        const auto status=file.at("status").str();
        if (status!="included" && status!="duplicate") continue;
        ModelPackagingAssignment choice; choice.relativePath=file.at("path").str();
        if (file.contains("role_source") && file.at("role_source").str()=="user") choice.role=file.at("role").str();
        if (file.contains("position_source") && file.at("position_source").str()=="user") choice.position=file.at("position").str();
        if (!choice.role.empty() || !choice.position.empty()) assignments.push_back(std::move(choice));
    }
    const auto sources = workspace / "sources";
    (void)extractModelPackage(input,sources,stop,observer);
    check(stop);
    U64 replaced=0,unchecked=0;
    if (options.replaceNonfiniteWithZero) {
        for (const auto &entry:fs::recursive_directory_iterator(sources)) {
            check(stop);
            if (!entry.is_regular_file() || modelContentFormat(entry.path(), entry.file_size()) != "safetensors") continue;
            const auto health=weightHealth(entry.path(),true,stop,observer);
            replaced=add(replaced,health.nonfinite); unchecked=add(unchecked,health.uncheckedTensors);
        }
    }
    auto result = parse(createModelPackageWithAssignments(sources,target,assignments,{},stop,observer));
    result.set("operation","calibrate");
    result.set("source_package",fs::absolute(input).string());
    result.set("weights_modified",replaced!=0);
    result.set("nonfinite_values_replaced",replaced);
    result.set("unchecked_tensors",unchecked);
    return result.dump();
}

namespace {
using PackageFiles = std::map<std::string, J>;
struct IndexedShardGroup {
    std::string indexPath, indexDigest;
    std::map<std::string, std::string> weights;
    std::vector<std::string> paths;
};
struct ShardTensor {
    const Tensor *packed;
    std::string name, digest;
    U64 outputBegin;
};
struct ShardPart {
    std::string path, header, digest;
    std::vector<ShardTensor> tensors;
};
bool validDigest(const std::string &value) {
    return value.size()==64 && std::all_of(value.begin(),value.end(),[](char c) {
        return (c>='0' && c<='9') || (c>='a' && c<='f');
    });
}
bool embeddedShardIndex(const std::string &path,const J &file) {
    if (file.at("format").str()!="asset") return false;
    if (file.contains("shard_index")) return file.at("shard_index").boolean();
    return hasSafetensorsIndexSuffix(path);
}
fs::path pathFromUtf8(const std::string &value) {
    const std::u8string utf8(reinterpret_cast<const char8_t *>(value.data()),value.size());
    return fs::path(utf8);
}
std::string pathUtf8(const fs::path &value) {
    const auto utf8=value.generic_u8string(); return {utf8.begin(),utf8.end()};
}
using TensorLookup = std::map<std::string_view,const Tensor *,std::less<>>;
TensorLookup indexTensors(const Source &source) {
    TensorLookup result;
    for (const auto &tensor : source.tensors) result.emplace(tensor.key,&tensor);
    return result;
}
const Tensor &mappedTensor(const TensorLookup &tensors,const std::string &key) {
    const auto found=tensors.find(key);
    if (found==tensors.end()) fail("Embedded shard mapping refers to a missing tensor");
    return *found->second;
}

IndexedShardGroup readShardIndex(const Source &source,const TensorLookup &tensors,const J &index,const PackageFiles &files,
                                 std::stop_token stop,const ModelPackagingObserver &observer) {
    const auto path=index.at("path").str(),digest=index.at("sha256").str();
    if (!embeddedShardIndex(path,index) || index.at("status").str()!="included"
        || index.at("format").str()!="asset" || !validDigest(digest)) fail("Invalid embedded shard index");
    const auto bytes=index.at("size").number(); const auto records=index.at("tensors");
    if (!bytes || bytes>headerLimit || !records.is(json_type_array) || records.size()!=1) fail("Invalid embedded shard index size");
    const auto record=records.at(std::size_t{0});
    const auto key=record.at("output_key").str(); const auto &packed=mappedTensor(tensors,key);
    if (!key.starts_with("__iild_assets__/") || packed.dtype!="U8" || packed.end-packed.begin!=bytes
        || record.at("begin").number()!=0 || record.at("end").number()!=bytes || record.at("sha256").str()!=digest)
        fail("Invalid embedded shard index mapping");
    progress(observer,stop,"materialize","Checking shard index",0,bytes);
    std::ifstream input(source.actual,std::ios::binary);
    input.seekg(static_cast<std::streamoff>(source.base+packed.begin));
    std::string contents(static_cast<std::size_t>(bytes),'\0');
    if (!input.read(contents.data(),static_cast<std::streamsize>(bytes))) fail("Embedded shard index is truncated");
    Hash hash; hash.update(contents); if (hash.finish()!=digest) fail("Embedded shard index checksum mismatch");
    stamp(source); check(stop);
    const auto weightMap=parse(contents).at("weight_map");
    if (!weightMap.is(json_type_object) || weightMap.members().empty()) fail("Shard index has no weight map");
    IndexedShardGroup group{path,digest,{},{}}; std::set<std::string> paths;
    const auto parent=pathFromUtf8(path).parent_path();
    for (const auto &[name,value] : weightMap.members()) {
        const auto reference=pathFromUtf8(value.str());
        if (name.empty() || !safeRelative(reference)) fail("Unsafe shard path in embedded index");
        const auto joined=(parent/reference).lexically_normal();
        if (!safeRelative(joined)) fail("Unsafe shard path in embedded index");
        const auto relative=pathUtf8(joined); const auto found=files.find(relative);
        if (found==files.end() || found->second.at("format").str()!="safetensors"
            || (found->second.at("status").str()!="included" && found->second.at("status").str()!="duplicate"))
            fail("Embedded shard index references a missing Safetensors file");
        if (!group.weights.emplace(name,relative).second) fail("Duplicate tensor in embedded shard index");
        paths.insert(relative);
    }
    group.paths.assign(paths.begin(),paths.end());
    return group;
}

fs::path materializeShardGroup(const Source &source,const TensorLookup &tensors,const PackageFiles &files,const IndexedShardGroup &group,
                               const fs::path &cacheDirectory,std::stop_token stop,
                               const ModelPackagingObserver &observer) {
    auto header=J::object(),metadata=J::object(); std::map<std::string,std::string> metadataValues;
    std::vector<ShardPart> parts; std::set<std::string> seenNames; U64 outputBytes=0;
    Hash groupHash; groupHash.update("iild-merged-shards-v1\n"); groupHash.update(group.indexDigest);
    for (const auto &path : group.paths) {
        const auto &file=files.at(path); const auto rawHeader=file.at("header").str(),digest=file.at("sha256").str();
        if (!validDigest(digest) || rawHeader.empty() || rawHeader.size()>headerLimit) fail("Invalid embedded shard header or checksum");
        const auto original=parse(rawHeader),mapping=file.at("tensors");
        if (!original.is(json_type_object) || !mapping.is(json_type_array) || !mapping.size()) fail("Invalid embedded shard tensor mapping");
        if (original.contains("__metadata__")) for (const auto &[key,value] : original.at("__metadata__").members()) {
            const auto text=value.str(); const auto [found,inserted]=metadataValues.emplace(key,text);
            if (!inserted && found->second!=text) fail("Conflicting shard metadata");
        }
        auto originalCount=original.members().size(); if (original.contains("__metadata__")) --originalCount;
        if (mapping.size()!=originalCount) fail("Shard header and tensor mapping differ");
        ShardPart part{path,rawHeader,digest,{}}; U64 cursor=0;
        const auto pathSize=sizeBytes(path.size()); groupHash.update(pathSize.data(),pathSize.size());
        groupHash.update(path); groupHash.update(digest);
        for (std::size_t i=0;i<mapping.size();++i) {
            const auto record=mapping.at(i); const auto name=record.at("key").str(),out=record.at("output_key").str();
            const auto indexed=group.weights.find(name);
            if (name.empty() || !seenNames.insert(name).second || indexed==group.weights.end() || indexed->second!=path)
                fail("Shard index does not exactly cover its tensors");
            const auto descriptor=original.at(name),shape=descriptor.at("shape"),offsets=descriptor.at("data_offsets");
            if (!shape.is(json_type_array) || !offsets.is(json_type_array) || offsets.size()!=2) fail("Invalid shard tensor descriptor");
            const auto begin=record.at("begin").number(),end=record.at("end").number();
            const auto &packed=mappedTensor(tensors,out); const auto tensorDigest=record.at("sha256").str();
            if (end<begin || (end!=begin && begin!=cursor) || offsets.at(0).number()!=begin || offsets.at(1).number()!=end
                || packed.end-packed.begin!=end-begin || packed.dtype!=descriptor.at("dtype").str()
                || shape.size()!=packed.shape.size() || !validDigest(tensorDigest)) fail("Invalid embedded shard tensor mapping");
            for (std::size_t d=0;d<shape.size();++d) if (shape.at(d).number()!=packed.shape[d]) fail("Embedded shard tensor shape mismatch");
            cursor=std::max(cursor,end);
            // Preserve any format-specific descriptor fields; only the byte
            // offsets change when the original shards are combined.
            auto outputDescriptor=parse(descriptor.dump()),outputOffsets=J::array();
            outputOffsets.push(outputBytes); const auto start=outputBytes; outputBytes=add(outputBytes,end-begin); outputOffsets.push(outputBytes);
            outputDescriptor.set("data_offsets",outputOffsets);
            header.set(name,outputDescriptor); part.tensors.push_back({&packed,name,tensorDigest,start});
        }
        if (add(add(8,rawHeader.size()),cursor)!=file.at("size").number()) fail("Incomplete embedded shard mapping");
        parts.push_back(std::move(part));
    }
    if (seenNames.size()!=group.weights.size()) fail("Shard index does not exactly cover its tensors");
    for (const auto &[key,value] : metadataValues) metadata.set(key,value);
    if (!metadataValues.empty()) header.set("__metadata__",metadata);
    auto encoded=header.dump(); if (encoded.size()>headerLimit-8) fail("Merged shard header exceeds the Safetensors limit");
    encoded.append((8-encoded.size()%8)%8,' '); const auto total=add(add(8,encoded.size()),outputBytes);
    const auto cache=fs::absolute(cacheDirectory).lexically_normal(); iiFileProvider::native::createDirectories(cache);
    if (fs::canonical(cache)!=cache) fail("The model asset cache must not be redirected");
    const auto target=cache/("sharded-"+groupHash.finish()+".safetensors"); removeAbandonedAssets(target);
    const auto verifyData=[&](const fs::path &path,bool package,iiFileProvider::native::OutputFile *output) {
        std::ifstream input(path,std::ios::binary); if (!input) fail("Cannot read model shard data");
        std::vector<char> buffer(chunkSize); U64 done=0;
        for (const auto &part : parts) {
            Hash whole; const auto prefix=sizeBytes(part.header.size()); whole.update(prefix.data(),prefix.size()); whole.update(part.header);
            for (const auto &entry : part.tensors) {
                const auto length=entry.packed->end-entry.packed->begin;
                const auto offset=package?source.base+entry.packed->begin:8+encoded.size()+entry.outputBegin;
                input.clear(); input.seekg(static_cast<std::streamoff>(offset)); Hash tensorHash;
                for (U64 position=0;position<length;) {
                    progress(observer,stop,"materialize","Checking embedded shard tensors",done,outputBytes);
                    const auto count=static_cast<std::size_t>(std::min<U64>(length-position,buffer.size()));
                    if (!input.read(buffer.data(),static_cast<std::streamsize>(count))) fail("Embedded shard tensor is truncated");
                    if (output) { output->write(buffer.data(),static_cast<std::streamsize>(count)); if (!*output) fail("Cannot write merged shard cache"); }
                    tensorHash.update(buffer.data(),count); whole.update(buffer.data(),count); position+=count; done+=count;
                }
                if (tensorHash.finish()!=entry.digest) fail("Embedded shard tensor checksum mismatch");
            }
            if (whole.finish()!=part.digest) fail("Embedded shard checksum mismatch");
        }
        progress(observer,stop,"materialize","Checking embedded shard tensors",done,outputBytes);
    };
    const auto validateCached=[&] {
        if (fs::canonical(target)!=target || !fs::is_regular_file(target) || fs::file_size(target)!=total)
            fail("The cached merged shard component has changed");
        Source cached; cached.logical=target; cached.actual=target; cached.size=total; readSafetensors(cached);
        if (cached.header!=encoded) fail("The cached merged shard header has changed");
        verifyData(target,false,nullptr);
    };
    if (fs::exists(target)) { validateCached(); stamp(source); return target; }
    destination(target,total); Temporary temporary{temporaryPath(target)}; createExclusive(temporary.path);
    iiFileProvider::native::OutputFile output(temporary.path,std::ios::binary|std::ios::trunc);
    const auto prefix=sizeBytes(encoded.size()); output.write(prefix.data(),prefix.size()); output.write(encoded.data(),static_cast<std::streamsize>(encoded.size()));
    if (!output) fail("Cannot write merged shard header");
    verifyData(source.actual,true,&output); output.close(); if (!output) fail("Cannot finalize merged shard cache");
    Source merged; merged.logical=temporary.path; merged.actual=temporary.path; merged.size=fs::file_size(temporary.path); readSafetensors(merged);
    stamp(source); check(stop); syncFile(temporary.path);
    if (fs::exists(target)) validateCached();
    else {
        try { commit(temporary.path,target); }
        catch (const std::exception &) {
            if (!fs::exists(target)) throw;
            validateCached();
        }
    }
    return target;
}
}

std::vector<detail::PackageShardIndex> detail::inspectPackageShardIndexes(const fs::path &input) {
    if (forbiddenModelExtension(input)) fail("ONNX model files are not supported");
    Source source;
    source.logical=fs::absolute(input); source.actual=resolvedSource(input);
    source.size=fs::file_size(input); source.time=fs::last_write_time(input); readSafetensors(source);
    if (!source.metadata.contains("iild_package_schema") || source.metadata.at("iild_package_schema").str()!=schema)
        fail("This file has no supported embedded model package manifest");
    const auto manifest=parse(source.metadata.at("iild_package_manifest").str());
    if (manifest.at("schema").str()!=schema) fail("Unsupported model-package manifest");
    requireNoOnnxManifest(manifest);
    const auto records=manifest.at("files");
    if (!records.is(json_type_array) || records.size()>10'000) fail("Invalid package file list");
    PackageFiles files;
    for (std::size_t i=0;i<records.size();++i) {
        const auto file=records.at(i); const auto path=file.at("path").str();
        if (!safeRelative(pathFromUtf8(path)) || !files.emplace(path,file).second) fail("Unsafe or duplicate package source path");
    }
    std::vector<detail::PackageShardIndex> result;
    std::optional<TensorLookup> tensors;
    for (const auto &[path,file] : files) {
        if (!embeddedShardIndex(path,file) || file.at("format").str()!="asset" || file.at("status").str()!="included") continue;
        if (!tensors) tensors.emplace(indexTensors(source));
        const auto group=readShardIndex(source,*tensors,file,files,{},{});
        detail::PackageShardIndex index; index.path=group.indexPath;
        index.weights.assign(group.weights.begin(),group.weights.end()); result.push_back(std::move(index));
    }
    stamp(source); return result;
}

detail::PackageAssets detail::inspectIncludedPackageAssets(const fs::path &input) {
    if (forbiddenModelExtension(input)) fail("ONNX model files are not supported");
    Source source;
    source.logical=fs::absolute(input); source.actual=resolvedSource(input);
    const auto before=objectIdentity(source.actual);
    source.size=before.size; source.time=fs::last_write_time(source.actual);
    // A standalone model belongs to its native loader. Only package headers
    // should be subjected to the packager's supported dtype restrictions.
    std::ifstream probe(source.actual,std::ios::binary);
    const auto headerBytes=readLE(probe,8);
    if (headerBytes<2 || headerBytes>headerLimit || source.size<8 || headerBytes>source.size-8)
        fail("Invalid Safetensors header length");
    std::string header(static_cast<std::size_t>(headerBytes),'\0');
    if (!probe.read(header.data(),static_cast<std::streamsize>(headerBytes))) fail("Truncated Safetensors header");
    const auto root=parse(header);
    const bool hasMetadata=root.contains("__metadata__") && root.at("__metadata__").is(json_type_object);
    if (!hasMetadata || !root.at("__metadata__").contains("iild_package_schema")) {
        if (hasMetadata && root.at("__metadata__").contains("iild_package_manifest")) fail("Invalid model package metadata");
        if (objectIdentity(source.actual)!=before) fail("Model changed during package inspection");
        stamp(source); return {};
    }
    readSafetensors(source,&header);
    if (source.metadata.at("iild_package_schema").str()!=schema || !source.metadata.contains("iild_package_manifest"))
        fail("Unsupported model-package manifest");
    const auto manifest=parse(source.metadata.at("iild_package_manifest").str());
    if (manifest.at("schema").str()!=schema) fail("Unsupported model-package manifest");
    requireNoOnnxManifest(manifest);
    const auto files=manifest.at("files");
    if (!files.is(json_type_array) || files.size()>10'000) fail("Invalid package file list");
    std::set<std::string> paths;
    detail::PackageAssets result;
    if (manifest.contains("wan_moe_boundary")) {
        const auto value=manifest.at("wan_moe_boundary");
        if (!value.is(json_type_double) && !value.is(json_type_int)) fail("Invalid packaged Wan boundary");
        const auto boundary=json_object_get_double(value.get());
        if (!std::isfinite(boundary) || boundary<0 || boundary>1) fail("Invalid packaged Wan boundary");
        result.wanMoeBoundary=static_cast<float>(boundary);
    }
    for (std::size_t i=0;i<files.size();++i) {
        const auto file=files.at(i);const auto path=file.at("path").str();
        if (!safeRelative(pathFromUtf8(path)) || !paths.insert(path).second)
            fail("Unsafe or duplicate package source path");
        const auto format=file.at("format").str();
        if (file.at("status").str()=="included" && (format=="safetensors" || format=="gguf"))
            result.roles.insert(file.at("role").str());
    }
    stamp(source);
    if (objectIdentity(source.actual)!=before) fail("Model changed during package inspection");
    return result;
}

bool detail::hasIncludedPackageAsset(const fs::path &input, const std::string &role) {
    return inspectIncludedPackageAssets(input).roles.contains(role);
}

std::optional<detail::EmbeddedGgufMain> detail::inspectEmbeddedMainGguf(const fs::path &input) {
    if (forbiddenModelExtension(input)) fail("ONNX model files are not supported");
    Source source;
    source.logical=fs::absolute(input);source.actual=resolvedSource(input);
    const auto before=objectIdentity(source.actual);
    source.size=before.size;source.time=fs::last_write_time(source.actual);
    // Ordinary Safetensors need no package-specific validation; inspect their
    // bounded metadata first so this route does not restrict native dtypes.
    std::ifstream probe(source.actual,std::ios::binary);
    const auto headerBytes=readLE(probe,8);
    if (headerBytes<2 || headerBytes>headerLimit || source.size<8 || headerBytes>source.size-8)
        fail("Invalid Safetensors header length");
    std::string header(static_cast<std::size_t>(headerBytes),'\0');
    if (!probe.read(header.data(),static_cast<std::streamsize>(headerBytes))) fail("Truncated Safetensors header");
    const auto root=parse(header);
    const bool packaged=root.contains("__metadata__") && root.at("__metadata__").is(json_type_object) &&
        root.at("__metadata__").contains("iild_package_schema");
    if (!packaged) {
        stamp(source);if (objectIdentity(source.actual)!=before) fail("Model changed during package inspection");
        return std::nullopt;
    }
    readSafetensors(source,&header);
    if (source.metadata.at("iild_package_schema").str()!=schema || !source.metadata.contains("iild_package_manifest"))
        fail("Unsupported model-package manifest");
    const auto manifest=parse(source.metadata.at("iild_package_manifest").str());
    if (manifest.at("schema").str()!=schema) fail("Unsupported model-package manifest");
    requireNoOnnxManifest(manifest);
    const auto files=manifest.at("files");
    if (!files.is(json_type_array) || files.size()>10'000) fail("Invalid package file list");
    std::set<std::string> paths,mainGroups;
    std::optional<J> selected;
    for (std::size_t i=0;i<files.size();++i) {
        const auto file=files.at(i);const auto path=file.at("path").str();
        if (!safeRelative(pathFromUtf8(path)) || !paths.insert(path).second) fail("Unsafe or duplicate package source path");
        if (file.at("status").str()!="included" || file.at("role").str()!="model" ||
            !file.contains("position") || file.at("position").str()!="main") continue;
        const auto group=file.contains("group") ? file.at("group").str() : path;
        if (!safeRelative(pathFromUtf8(group))) fail("Invalid package model group");
        mainGroups.insert(group);
        if (file.at("format").str()=="gguf") {
            if (selected) fail("Multiple packaged primary GGUF models");
            selected=file;
        }
    }
    if (!selected) {stamp(source);return std::nullopt;}
    if (mainGroups.size()!=1) fail("Multiple packaged primary models");
    const auto path=selected->at("path").str(),digest=selected->at("sha256").str();
    const auto records=selected->at("tensors");const auto bytes=selected->at("size").number();
    if (!records.is(json_type_array) || records.size()!=1 || digest.size()!=64 ||
        !std::all_of(digest.begin(),digest.end(),[](char c){return (c>='0'&&c<='9') || (c>='a'&&c<='f');}))
        fail("Invalid embedded primary GGUF mapping");
    const auto record=records.at(std::size_t{0});
    const auto key=record.at("output_key").str();
    const auto tensor=std::find_if(source.tensors.begin(),source.tensors.end(),[&](const auto &item){return item.key==key;});
    if (key!="__iild_assets__/model/"+path || tensor==source.tensors.end() || tensor->dtype!="U8" ||
        tensor->shape!=std::vector<U64>{bytes} || tensor->end-tensor->begin!=bytes ||
        record.at("begin").number()!=0 || record.at("end").number()!=bytes || record.at("sha256").str()!=digest)
        fail("Invalid embedded primary GGUF size or mapping");
    Source embedded;embedded.actual=source.actual;embedded.size=bytes;
    detail::EmbeddedGgufMain result;result.relativePath=pathFromUtf8(path);
    readGGUF(embedded,&result.catalog,source.base+tensor->begin);
    result.catalog.metadataBytesRead=add(result.catalog.metadataBytesRead,source.base);
    stamp(source);
    if (objectIdentity(source.actual)!=before) fail("Model package changed during GGUF inspection");
    return result;
}

static fs::path materializeAsset(const fs::path &input, const fs::path &cacheDirectory,
                                 const std::string &role, const fs::path &relativeSource,
                                 std::stop_token stop, const ModelPackagingObserver &observer) {
    if (!relativeSource.empty() && !safeRelative(relativeSource)) fail("Invalid embedded asset path");
    if (forbiddenModelExtension(input)) fail("ONNX model files are not supported");
    Source source;
    source.logical = fs::absolute(input); source.actual = resolvedSource(input);
    source.size = fs::file_size(input); source.time = fs::last_write_time(input);
    readSafetensors(source);
    if (!source.metadata.contains("iild_package_schema") || source.metadata.at("iild_package_schema").str() != schema)
        fail("This file has no supported embedded model package manifest");
    const auto manifest = parse(source.metadata.at("iild_package_manifest").str());
    if (manifest.at("schema").str() != schema) fail("Unsupported model-package manifest");
    requireNoOnnxManifest(manifest);
    const auto files = manifest.at("files");
    if (!files.is(json_type_array) || files.size()>10'000) fail("Invalid package file list");
    std::optional<J> selected;
    std::optional<TensorLookup> tensors;
    const auto mappedTensors = [&]() -> const TensorLookup & {
        if (!tensors) tensors.emplace(indexTensors(source));
        return *tensors;
    };
    PackageFiles sourceFiles; std::vector<J> candidates; std::set<std::string> rolePaths;
    const auto utf8 = relativeSource.generic_u8string();
    const std::string requestedPath(utf8.begin(), utf8.end());
    for (std::size_t i = 0; i < files.size(); ++i) {
        const auto file = files.at(i);
        const auto manifestPath = file.at("path").str();
        const std::u8string pathUtf8(reinterpret_cast<const char8_t *>(manifestPath.data()), manifestPath.size());
        if (!safeRelative(fs::path(pathUtf8)) || !sourceFiles.emplace(manifestPath,file).second)
            fail("Unsafe or duplicate package source path");
        const auto status = file.at("status").str();
        const bool supported = file.at("format").str() == "safetensors" || file.at("format").str() == "gguf";
        const bool matches = relativeSource.empty() ? file.at("role").str() == role : manifestPath == requestedPath;
        if (supported && relativeSource.empty() && matches && (status == "included" || status == "duplicate"))
            rolePaths.insert(manifestPath);
        // Duplicate source mappings are safe to restore when the caller names
        // the exact file. Automatic single-file selection retains its prior
        // behavior and does not mistake embedded duplicates for new assets.
        if (supported && matches && (status == "included" || (!relativeSource.empty() && status == "duplicate"))) {
            candidates.push_back(file);
        }
    }
    const auto requestedIndex=sourceFiles.find(requestedPath);
    if (!relativeSource.empty() && requestedIndex!=sourceFiles.end() && embeddedShardIndex(requestedPath,requestedIndex->second)) {
        const auto index=requestedIndex;
        const auto group=readShardIndex(source,mappedTensors(),index->second,sourceFiles,stop,observer);
        return materializeShardGroup(source,mappedTensors(),sourceFiles,group,cacheDirectory,stop,observer);
    }
    if (!relativeSource.empty() && hasSafetensorsIndexSuffix(requestedPath) && requestedIndex==sourceFiles.end())
        fail("This model package has no embedded shard index at the selected path");
    if (!rolePaths.empty() && relativeSource.empty()) {
        std::set<std::string> candidatePaths;
        for (const auto &candidate : candidates) candidatePaths.insert(candidate.at("path").str());
        std::optional<IndexedShardGroup> selectedGroup; std::string indexError; bool referenced=false;
        for (const auto &[path,file] : sourceFiles) {
            if (!embeddedShardIndex(path,file) || file.at("format").str()!="asset" || file.at("status").str()!="included") continue;
            const auto parent=pathUtf8(pathFromUtf8(path).parent_path());
            if (!parent.empty() && std::none_of(rolePaths.begin(),rolePaths.end(),[&](const auto &candidate) {
                    return candidate.starts_with(parent+"/");
                })) continue;
            std::optional<IndexedShardGroup> group;
            try { group=readShardIndex(source,mappedTensors(),file,sourceFiles,stop,observer); }
            catch (const std::exception &error) { if (stop.stop_requested()) throw; indexError=error.what(); continue; }
            if (std::none_of(group->paths.begin(),group->paths.end(),[&](const auto &candidate) {
                    return rolePaths.contains(candidate);
                })) continue;
            if (referenced) fail("Multiple shard indexes reference the selected component; choose an explicit source");
            referenced=true;
            std::set<std::string> includedPaths; bool sameRole=true;
            for (const auto &member : group->paths) {
                const auto &sourceFile=sourceFiles.at(member);
                sameRole &= sourceFile.at("role").str()==role;
                if (sourceFile.at("status").str()=="included") includedPaths.insert(member);
            }
            if (!sameRole || includedPaths!=candidatePaths) continue;
            if (selectedGroup) fail("Multiple matching shard indexes; choose an explicit source");
            selectedGroup=std::move(group);
        }
        if (selectedGroup) return materializeShardGroup(source,mappedTensors(),sourceFiles,*selectedGroup,cacheDirectory,stop,observer);
        if (referenced || candidates.size()>1 || !indexError.empty())
            fail("Multiple matching or inconsistent embedded assets; choose an unambiguous shard index"+(indexError.empty()?std::string{}:": "+indexError));
    }
    if (candidates.size()>1) fail("Multiple matching embedded assets; choose an unambiguous source");
    if (!candidates.empty()) selected=candidates.front();
    if (!selected) fail("This model package has no embedded supported " + role);
    const auto digest = selected->at("sha256").str();
    if (digest.size() != 64 || !std::all_of(digest.begin(), digest.end(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }))
        fail("Invalid embedded asset checksum");
    if (selected->at("format").str() == "gguf") {
        const auto mapping = selected->at("tensors");
        if (!mapping.is(json_type_array) || mapping.size() != 1) fail("Invalid embedded asset mapping");
        const auto record = mapping.at(std::size_t{0});
        const auto key = record.at("output_key").str();
        const auto tensor = std::find_if(source.tensors.begin(), source.tensors.end(), [&](const auto &item) { return item.key == key; });
        const auto bytes = selected->at("size").number();
        if (!key.starts_with("__iild_assets__/") || tensor == source.tensors.end() || tensor->dtype != "U8"
            || tensor->end - tensor->begin != bytes || record.at("begin").number() != 0
            || record.at("end").number() != bytes || record.at("sha256").str() != digest)
            fail("Invalid embedded asset size or checksum mapping");
        const auto cache = fs::absolute(cacheDirectory).lexically_normal();
        iiFileProvider::native::createDirectories(cache);
        if (fs::canonical(cache) != cache) fail("The model asset cache must not be redirected");
        const auto target = cache / (digest + ".gguf");
        removeAbandonedAssets(target);
        const auto validateGGUF = [](const fs::path &path) {
            Source candidate; candidate.logical = path; candidate.actual = path; candidate.size = fs::file_size(path);
            readGGUF(candidate);
        };
        if (fs::exists(target)) {
            if (fs::canonical(target) != target || !fs::is_regular_file(target) || fs::file_size(target) != bytes
                || hashRange(target, 0, bytes, stop, observer, "Checking cached " + role) != digest)
                fail("The cached model component has changed; remove the invalid cache asset and retry");
            validateGGUF(target);
            stamp(source); return target;
        }
        destination(target, bytes);
        Temporary temporary{temporaryPath(target)}; createExclusive(temporary.path);
        std::vector<char> inputBuffer(1024 * 1024), outputBuffer(1024 * 1024);
        std::ifstream file;
        file.rdbuf()->pubsetbuf(inputBuffer.data(), static_cast<std::streamsize>(inputBuffer.size()));
        file.open(source.actual, std::ios::binary);
        file.seekg(static_cast<std::streamoff>(source.base + tensor->begin));
        iiFileProvider::native::OutputFile output;
        output.rdbuf()->pubsetbuf(outputBuffer.data(), static_cast<std::streamsize>(outputBuffer.size()));
        output.open(temporary.path, std::ios::binary | std::ios::trunc);
        std::vector<char> buffer(chunkSize); U64 done = 0; Hash hash;
        while (done < bytes) {
            progress(observer, stop, "materialize", "Preparing embedded " + role, done, bytes);
            const auto count = static_cast<std::size_t>(std::min<U64>(bytes - done, buffer.size()));
            if (!file.read(buffer.data(), static_cast<std::streamsize>(count))) fail("Embedded asset is truncated");
            output.write(buffer.data(), static_cast<std::streamsize>(count));
            if (!output) fail("Cannot write the embedded model component cache");
            hash.update(buffer.data(), count); done += count;
        }
        progress(observer, stop, "materialize", "Preparing embedded " + role, done, bytes);
        output.close();
        if (hash.finish() != digest) fail("Embedded model component checksum mismatch");
        validateGGUF(temporary.path);
        stamp(source); check(stop); syncFile(temporary.path);
        if (fs::exists(target)) {
            if (fs::canonical(target) != target || fs::file_size(target) != bytes
                || hashRange(target, 0, bytes, stop, observer, "Checking cached " + role) != digest)
                fail("Conflicting model asset cache publication");
            validateGGUF(target);
        } else commit(temporary.path, target);
        return target;
    }
    const auto mapping = selected->at("tensors");
    if (!mapping.is(json_type_array) || mapping.size() == 0 || mapping.size() > source.tensors.size())
        fail("Invalid embedded asset mapping");
    const auto header = selected->at("header").str();
    const auto bytes = selected->at("size").number();
    if (header.size() > headerLimit || bytes < add(8, header.size())) fail("Invalid embedded safetensors header");
    std::vector<const Tensor *> mapped;
    std::set<std::string> usedKeys;
    U64 cursor = 0;
    for (std::size_t i = 0; i < mapping.size(); ++i) {
        const auto record = mapping.at(i);
        const auto key = record.at("output_key").str();
        const auto tensor = std::find_if(source.tensors.begin(), source.tensors.end(), [&](const auto &item) { return item.key == key; });
        const auto begin = record.at("begin").number(), end = record.at("end").number();
        if (!usedKeys.insert(key).second || tensor == source.tensors.end() || end < begin || begin != cursor ||
            end - begin != tensor->end - tensor->begin || record.at("sha256").str().size() != 64)
            fail("Invalid embedded safetensors size or checksum mapping");
        cursor = end;
        mapped.push_back(&*tensor);
    }
    if (add(add(8, header.size()), cursor) != bytes) fail("Incomplete embedded safetensors mapping");
    const auto cache = fs::absolute(cacheDirectory).lexically_normal();
    iiFileProvider::native::createDirectories(cache);
    if (fs::canonical(cache) != cache) fail("The model asset cache must not be redirected");
    const auto target = cache / (digest + ".safetensors");
    removeAbandonedAssets(target);
    const auto validateCached = [&] {
        Source cached;
        cached.logical = target; cached.actual = target;
        cached.size = fs::file_size(target);
        readSafetensors(cached);
    };
    if (fs::exists(target)) {
        if (fs::canonical(target) != target || !fs::is_regular_file(target) || fs::file_size(target) != bytes
            || hashRange(target, 0, bytes, stop, observer, "Checking cached " + role) != digest)
            fail("The cached model component has changed; remove the invalid cache asset and retry");
        validateCached();
        stamp(source); return target;
    }
    destination(target, bytes);
    Temporary temporary{temporaryPath(target)}; createExclusive(temporary.path);
    std::vector<char> inputBuffer(1024 * 1024), outputBuffer(1024 * 1024);
    std::ifstream file;
    file.rdbuf()->pubsetbuf(inputBuffer.data(), static_cast<std::streamsize>(inputBuffer.size()));
    file.open(source.actual, std::ios::binary);
    iiFileProvider::native::OutputFile output;
    output.rdbuf()->pubsetbuf(outputBuffer.data(), static_cast<std::streamsize>(outputBuffer.size()));
    output.open(temporary.path, std::ios::binary | std::ios::trunc);
    std::vector<char> buffer(chunkSize); U64 done = add(8, header.size()); Hash hash;
    const auto headerBytes = sizeBytes(header.size());
    output.write(headerBytes.data(), headerBytes.size());
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    hash.update(headerBytes.data(), headerBytes.size()); hash.update(header);
    for (std::size_t i = 0; i < mapped.size(); ++i) {
        const auto *tensor = mapped[i];
        file.clear(); file.seekg(static_cast<std::streamoff>(source.base + tensor->begin));
        Hash tensorHash; U64 position = 0;
        while (position < tensor->end - tensor->begin) {
            progress(observer, stop, "materialize", "Preparing embedded " + role, done, bytes);
            const auto count = static_cast<std::size_t>(std::min<U64>(tensor->end - tensor->begin - position, buffer.size()));
            if (!file.read(buffer.data(), static_cast<std::streamsize>(count))) fail("Embedded asset is truncated");
            output.write(buffer.data(), static_cast<std::streamsize>(count));
            if (!output) fail("Cannot write the embedded model component cache");
            tensorHash.update(buffer.data(), count); hash.update(buffer.data(), count);
            position += count; done += count;
        }
        if (tensorHash.finish() != mapping.at(i).at("sha256").str()) fail("Embedded tensor checksum mismatch");
    }
    progress(observer, stop, "materialize", "Preparing embedded " + role, done, bytes);
    output.close();
    if (hash.finish() != digest) fail("Embedded model component checksum mismatch");
    Source reconstructed;
    reconstructed.logical = temporary.path; reconstructed.actual = temporary.path;
    reconstructed.size = fs::file_size(temporary.path);
    readSafetensors(reconstructed);
    stamp(source); check(stop); syncFile(temporary.path);
    if (fs::exists(target)) {
        if (fs::canonical(target) != target || fs::file_size(target) != bytes
            || hashRange(target, 0, bytes, stop, observer, "Checking cached " + role) != digest)
            fail("Conflicting model asset cache publication");
        validateCached();
    } else commit(temporary.path, target);
    return target;
}

fs::path materializeModelPackageAsset(const fs::path &input, const fs::path &cacheDirectory,
                                     const std::string &role, std::stop_token stop,
                                     const ModelPackagingObserver &observer) {
    return materializeAsset(input, cacheDirectory, role, {}, stop, observer);
}

fs::path materializeModelPackageAssetByPath(const fs::path &input, const fs::path &cacheDirectory,
                                           const fs::path &relativeSource, std::stop_token stop,
                                           const ModelPackagingObserver &observer) {
    if (relativeSource.empty()) fail("Choose an embedded asset path");
    const auto utf8 = relativeSource.generic_u8string();
    return materializeAsset(input, cacheDirectory, std::string(utf8.begin(), utf8.end()), relativeSource, stop, observer);
}
}
