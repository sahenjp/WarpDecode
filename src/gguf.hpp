#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace wd {

constexpr uint32_t GGUF_MAGIC = 0x46554747u;

enum class GgufValueType : uint32_t {
    U8 = 0, I8 = 1, U16 = 2, I16 = 3, U32 = 4, I32 = 5,
    F32 = 6, Bool = 7, String = 8, Array = 9, U64 = 10, I64 = 11, F64 = 12,
};

enum class GgmlType : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
};

class MappedFile {
public:
    explicit MappedFile(const std::string& path) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("GGUFを開けません: " + path);
        struct stat st{};
        if (::fstat(fd_, &st) != 0 || st.st_size <= 0) {
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("GGUFのサイズ取得に失敗しました");
        }
        size_ = static_cast<size_t>(st.st_size);
        data_ = static_cast<const uint8_t*>(::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0));
        if (data_ == MAP_FAILED) {
            data_ = nullptr;
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("GGUFのmmapに失敗しました");
        }
    }

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    MappedFile(MappedFile&& other) noexcept { swap(other); }
    MappedFile& operator=(MappedFile&& other) noexcept {
        if (this != &other) {
            cleanup();
            swap(other);
        }
        return *this;
    }

    ~MappedFile() { cleanup(); }

    const uint8_t* data() const { return data_; }
    size_t size() const { return size_; }

private:
    void cleanup() {
        if (data_) ::munmap(const_cast<uint8_t*>(data_), size_);
        if (fd_ >= 0) ::close(fd_);
        data_ = nullptr;
        size_ = 0;
        fd_ = -1;
    }
    void swap(MappedFile& o) noexcept {
        std::swap(fd_, o.fd_);
        std::swap(data_, o.data_);
        std::swap(size_, o.size_);
    }

    int fd_ = -1;
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
};

class Cursor {
public:
    Cursor(const uint8_t* data, size_t size) : data_(data), size_(size) {}

    template <typename T>
    T read() {
        require(sizeof(T));
        T v{};
        std::memcpy(&v, data_ + off_, sizeof(T));
        off_ += sizeof(T);
        return v;
    }

    std::string read_string() {
        const uint64_t n = read<uint64_t>();
        if (n > (1ull << 31)) throw std::runtime_error("GGUF文字列が異常に大きいです");
        require(static_cast<size_t>(n));
        std::string s(reinterpret_cast<const char*>(data_ + off_), static_cast<size_t>(n));
        off_ += static_cast<size_t>(n);
        return s;
    }

    void skip(size_t n) {
        require(n);
        off_ += n;
    }

    size_t offset() const { return off_; }

private:
    void require(size_t n) const {
        if (n > size_ - off_) throw std::runtime_error("GGUFが途中で切れています");
    }
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
    size_t off_ = 0;
};

struct MetaValue {
    enum class Kind { None, U64, I64, F64, Bool, String } kind = Kind::None;
    uint64_t u64 = 0;
    int64_t i64 = 0;
    double f64 = 0.0;
    bool boolean = false;
    std::string str;
};

inline MetaValue read_scalar(Cursor& c, GgufValueType t);

inline void skip_value(Cursor& c, GgufValueType t) {
    switch (t) {
        case GgufValueType::U8: case GgufValueType::I8: case GgufValueType::Bool: c.skip(1); break;
        case GgufValueType::U16: case GgufValueType::I16: c.skip(2); break;
        case GgufValueType::U32: case GgufValueType::I32: case GgufValueType::F32: c.skip(4); break;
        case GgufValueType::U64: case GgufValueType::I64: case GgufValueType::F64: c.skip(8); break;
        case GgufValueType::String: (void)c.read_string(); break;
        case GgufValueType::Array: {
            const auto elem = static_cast<GgufValueType>(c.read<uint32_t>());
            const uint64_t n = c.read<uint64_t>();
            if (n > (1ull << 34)) throw std::runtime_error("GGUF配列が異常に大きいです");
            for (uint64_t i = 0; i < n; ++i) skip_value(c, elem);
            break;
        }
        default: throw std::runtime_error("未対応のGGUFメタデータ型です");
    }
}

inline MetaValue read_scalar(Cursor& c, GgufValueType t) {
    MetaValue out;
    switch (t) {
        case GgufValueType::U8: out.kind = MetaValue::Kind::U64; out.u64 = c.read<uint8_t>(); break;
        case GgufValueType::I8: out.kind = MetaValue::Kind::I64; out.i64 = c.read<int8_t>(); break;
        case GgufValueType::U16: out.kind = MetaValue::Kind::U64; out.u64 = c.read<uint16_t>(); break;
        case GgufValueType::I16: out.kind = MetaValue::Kind::I64; out.i64 = c.read<int16_t>(); break;
        case GgufValueType::U32: out.kind = MetaValue::Kind::U64; out.u64 = c.read<uint32_t>(); break;
        case GgufValueType::I32: out.kind = MetaValue::Kind::I64; out.i64 = c.read<int32_t>(); break;
        case GgufValueType::U64: out.kind = MetaValue::Kind::U64; out.u64 = c.read<uint64_t>(); break;
        case GgufValueType::I64: out.kind = MetaValue::Kind::I64; out.i64 = c.read<int64_t>(); break;
        case GgufValueType::F32: out.kind = MetaValue::Kind::F64; out.f64 = c.read<float>(); break;
        case GgufValueType::F64: out.kind = MetaValue::Kind::F64; out.f64 = c.read<double>(); break;
        case GgufValueType::Bool: out.kind = MetaValue::Kind::Bool; out.boolean = c.read<uint8_t>() != 0; break;
        case GgufValueType::String: out.kind = MetaValue::Kind::String; out.str = c.read_string(); break;
        case GgufValueType::Array: skip_value(c, t); break;
        default: throw std::runtime_error("未対応のGGUFメタデータ型です");
    }
    return out;
}

struct TensorInfo {
    std::string name;
    std::vector<uint64_t> dims;
    GgmlType type = GgmlType::F32;
    uint64_t relative_offset = 0;
    const uint8_t* data = nullptr;

    uint64_t elements() const {
        uint64_t n = 1;
        for (uint64_t d : dims) n *= d;
        return n;
    }

    size_t byte_size() const {
        const uint64_t n = elements();
        switch (type) {
            case GgmlType::F32: return static_cast<size_t>(n * 4);
            case GgmlType::F16: return static_cast<size_t>(n * 2);
            case GgmlType::Q4_0:
                if (n % 32 != 0) throw std::runtime_error("Q4_0テンソル要素数が32の倍数ではありません: " + name);
                return static_cast<size_t>((n / 32) * 18);
            default: throw std::runtime_error("未対応テンソル型: " + name);
        }
    }
};

class GgufFile {
public:
    explicit GgufFile(const std::string& path) : mapped_(path) { parse(); }

    const TensorInfo& tensor(const std::string& name) const {
        auto it = tensor_index_.find(name);
        if (it == tensor_index_.end()) throw std::runtime_error("必須テンソルがありません: " + name);
        return tensors_[it->second];
    }

    const TensorInfo* optional_tensor(const std::string& name) const {
        auto it = tensor_index_.find(name);
        return it == tensor_index_.end() ? nullptr : &tensors_[it->second];
    }

    std::string string_meta(const std::string& key) const {
        const auto& v = meta(key);
        if (v.kind != MetaValue::Kind::String) throw std::runtime_error("文字列メタデータではありません: " + key);
        return v.str;
    }

    uint64_t uint_meta(const std::string& key) const {
        const auto& v = meta(key);
        if (v.kind == MetaValue::Kind::U64) return v.u64;
        if (v.kind == MetaValue::Kind::I64 && v.i64 >= 0) return static_cast<uint64_t>(v.i64);
        throw std::runtime_error("整数メタデータではありません: " + key);
    }

    double number_meta(const std::string& key, double fallback) const {
        auto it = metadata_.find(key);
        if (it == metadata_.end()) return fallback;
        const auto& v = it->second;
        if (v.kind == MetaValue::Kind::F64) return v.f64;
        if (v.kind == MetaValue::Kind::U64) return static_cast<double>(v.u64);
        if (v.kind == MetaValue::Kind::I64) return static_cast<double>(v.i64);
        return fallback;
    }

    std::string string_meta_or(const std::string& key, const std::string& fallback) const {
        auto it = metadata_.find(key);
        if (it == metadata_.end()) return fallback;
        if (it->second.kind != MetaValue::Kind::String) throw std::runtime_error("文字列メタデータではありません: " + key);
        return it->second.str;
    }

    bool has_meta(const std::string& key) const { return metadata_.count(key) != 0; }
    const std::vector<TensorInfo>& tensors() const { return tensors_; }
    uint32_t version() const { return version_; }
    size_t data_offset() const { return data_offset_; }

private:
    const MetaValue& meta(const std::string& key) const {
        auto it = metadata_.find(key);
        if (it == metadata_.end()) throw std::runtime_error("必須メタデータがありません: " + key);
        return it->second;
    }

    static size_t align_up(size_t x, size_t a) {
        if (a == 0 || (a & (a - 1)) != 0) throw std::runtime_error("GGUF alignmentが不正です");
        return (x + a - 1) & ~(a - 1);
    }

    void parse() {
        Cursor c(mapped_.data(), mapped_.size());
        if (c.read<uint32_t>() != GGUF_MAGIC) throw std::runtime_error("GGUF magicが一致しません");
        version_ = c.read<uint32_t>();
        if (version_ < 2 || version_ > 3) throw std::runtime_error("対応外GGUF versionです");
        const uint64_t tensor_count = c.read<uint64_t>();
        const uint64_t kv_count = c.read<uint64_t>();
        if (tensor_count > 1'000'000 || kv_count > 1'000'000) throw std::runtime_error("GGUFヘッダ件数が異常です");

        for (uint64_t i = 0; i < kv_count; ++i) {
            const std::string key = c.read_string();
            const auto type = static_cast<GgufValueType>(c.read<uint32_t>());
            if (type == GgufValueType::Array) {
                skip_value(c, type);
            } else {
                metadata_.emplace(key, read_scalar(c, type));
            }
        }

        tensors_.reserve(static_cast<size_t>(tensor_count));
        for (uint64_t i = 0; i < tensor_count; ++i) {
            TensorInfo t;
            t.name = c.read_string();
            const uint32_t nd = c.read<uint32_t>();
            if (nd == 0 || nd > 4) throw std::runtime_error("テンソル次元数が不正です: " + t.name);
            t.dims.resize(nd);
            for (uint32_t d = 0; d < nd; ++d) t.dims[d] = c.read<uint64_t>();
            t.type = static_cast<GgmlType>(c.read<uint32_t>());
            t.relative_offset = c.read<uint64_t>();
            tensor_index_.emplace(t.name, tensors_.size());
            tensors_.push_back(std::move(t));
        }

        size_t alignment = 32;
        auto ait = metadata_.find("general.alignment");
        if (ait != metadata_.end()) {
            if (ait->second.kind == MetaValue::Kind::U64) alignment = static_cast<size_t>(ait->second.u64);
            else if (ait->second.kind == MetaValue::Kind::I64 && ait->second.i64 > 0) alignment = static_cast<size_t>(ait->second.i64);
        }
        data_offset_ = align_up(c.offset(), alignment);
        if (data_offset_ > mapped_.size()) throw std::runtime_error("GGUFデータ開始位置がファイル外です");

        for (auto& t : tensors_) {
            const size_t abs = data_offset_ + static_cast<size_t>(t.relative_offset);
            const size_t bytes = t.byte_size();
            if (abs > mapped_.size() || bytes > mapped_.size() - abs) {
                throw std::runtime_error("テンソルがファイル外を参照しています: " + t.name);
            }
            t.data = mapped_.data() + abs;
        }
    }

    MappedFile mapped_;
    uint32_t version_ = 0;
    size_t data_offset_ = 0;
    std::unordered_map<std::string, MetaValue> metadata_;
    std::vector<TensorInfo> tensors_;
    std::unordered_map<std::string, size_t> tensor_index_;
};

} // namespace wd
