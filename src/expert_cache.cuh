#pragma once

#include "gguf.hpp"
#include "lowbit.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace wd {

struct ExpertHostSlice {
    const uint8_t * data = nullptr;
    size_t bytes = 0;
    int rows = 0;
    int cols = 0;
    GgmlType type = GgmlType::F32;
};

inline ExpertHostSlice expert_slice(const TensorInfo & t, int expert) {
    if (t.dims.size() != 3) {
        throw std::runtime_error("expert tensor must be 3D: " + t.name);
    }
    const int n_expert = static_cast<int>(t.dims[2]);
    if (expert < 0 || expert >= n_expert) {
        throw std::runtime_error("expert index out of range: " + t.name);
    }
    if (!lowbit_gemv_supported(t.type)) {
        throw std::runtime_error("expert tensor has unsupported low-bit type: " + t.name);
    }

    TensorInfo one;
    one.name = t.name;
    one.dims = {t.dims[0], t.dims[1]};
    one.type = t.type;
    const size_t slice_bytes = one.byte_size();

    const size_t total_expected = slice_bytes * static_cast<size_t>(n_expert);
    if (total_expected != t.byte_size()) {
        throw std::runtime_error("expert tensor is not slice-contiguous: " + t.name);
    }

    ExpertHostSlice out;
    out.data = t.data + slice_bytes * static_cast<size_t>(expert);
    out.bytes = slice_bytes;
    out.cols = static_cast<int>(t.dims[0]);
    out.rows = static_cast<int>(t.dims[1]);
    out.type = t.type;
    return out;
}

struct ExpertDeviceWeights {
    LowBitMatrix gate;
    LowBitMatrix up;
    LowBitMatrix down;
    bool cache_hit = false;
    size_t slot = 0;
};

class GpuExpertCache {
public:
    GpuExpertCache() = default;
    GpuExpertCache(const GpuExpertCache &) = delete;
    GpuExpertCache & operator=(const GpuExpertCache &) = delete;

    GpuExpertCache(GpuExpertCache && o) noexcept { move_from(std::move(o)); }
    GpuExpertCache & operator=(GpuExpertCache && o) noexcept {
        if (this != &o) {
            release();
            move_from(std::move(o));
        }
        return *this;
    }

    ~GpuExpertCache() { release(); }

    void initialize(
            size_t capacity_mib,
            const TensorInfo & sample_gate,
            const TensorInfo & sample_up,
            const TensorInfo & sample_down) {
        if (pool_) throw std::runtime_error("expert cache already initialized");
        if (capacity_mib == 0) throw std::runtime_error("expert cache capacity is zero");

        const auto g = expert_slice(sample_gate, 0);
        const auto u = expert_slice(sample_up, 0);
        const auto d = expert_slice(sample_down, 0);

        gate_bytes_ = g.bytes;
        up_bytes_ = u.bytes;
        down_bytes_ = d.bytes;
        slot_bytes_ = gate_bytes_ + up_bytes_ + down_bytes_;
        if (slot_bytes_ == 0) throw std::runtime_error("expert cache slot is empty");

        gate_rows_ = g.rows; gate_cols_ = g.cols; gate_type_ = g.type;
        up_rows_   = u.rows; up_cols_   = u.cols; up_type_   = u.type;
        down_rows_ = d.rows; down_cols_ = d.cols; down_type_ = d.type;

        const size_t capacity_bytes = capacity_mib * 1024ull * 1024ull;
        n_slots_ = capacity_bytes / slot_bytes_;
        if (n_slots_ == 0) {
            throw std::runtime_error("expert cache capacity is smaller than one expert slot");
        }

        const cudaError_t e = cudaMalloc(reinterpret_cast<void **>(&pool_), n_slots_ * slot_bytes_);
        if (e != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMalloc expert cache failed: ") + cudaGetErrorString(e));
        }
        slots_.resize(n_slots_);
    }

    size_t slot_count() const { return n_slots_; }
    size_t slot_bytes() const { return slot_bytes_; }
    size_t resident_count() const { return index_.size(); }

    ExpertDeviceWeights acquire(
            int layer,
            int expert,
            const TensorInfo & gate,
            const TensorInfo & up,
            const TensorInfo & down,
            cudaStream_t stream) {
        if (!pool_) throw std::runtime_error("expert cache is not initialized");

        const uint64_t key = make_key(layer, expert);
        ++clock_;

        auto it = index_.find(key);
        if (it != index_.end()) {
            Slot & s = slots_[it->second];
            s.last_use = clock_;
            return views(it->second, true);
        }

        const auto gh = expert_slice(gate, expert);
        const auto uh = expert_slice(up, expert);
        const auto dh = expert_slice(down, expert);
        validate_geometry(gh, uh, dh);

        const size_t slot = choose_slot();
        Slot & meta = slots_[slot];
        if (meta.valid) index_.erase(meta.key);

        uint8_t * base = pool_ + slot * slot_bytes_;
        copy_async(base, gh.data, gate_bytes_, stream, gate.name);
        copy_async(base + gate_bytes_, uh.data, up_bytes_, stream, up.name);
        copy_async(base + gate_bytes_ + up_bytes_, dh.data, down_bytes_, stream, down.name);

        meta.key = key;
        meta.last_use = clock_;
        meta.valid = true;
        index_[key] = slot;
        return views(slot, false);
    }

private:
    void release() {
        if (pool_) cudaFree(pool_);
        pool_ = nullptr;
        n_slots_ = 0;
        slot_bytes_ = gate_bytes_ = up_bytes_ = down_bytes_ = 0;
        slots_.clear();
        index_.clear();
    }

    void move_from(GpuExpertCache && o) noexcept {
        pool_ = o.pool_;
        n_slots_ = o.n_slots_;
        slot_bytes_ = o.slot_bytes_;
        gate_bytes_ = o.gate_bytes_;
        up_bytes_ = o.up_bytes_;
        down_bytes_ = o.down_bytes_;
        gate_rows_ = o.gate_rows_; gate_cols_ = o.gate_cols_;
        up_rows_ = o.up_rows_; up_cols_ = o.up_cols_;
        down_rows_ = o.down_rows_; down_cols_ = o.down_cols_;
        gate_type_ = o.gate_type_;
        up_type_ = o.up_type_;
        down_type_ = o.down_type_;
        clock_ = o.clock_;
        slots_ = std::move(o.slots_);
        index_ = std::move(o.index_);

        o.pool_ = nullptr;
        o.n_slots_ = 0;
        o.slot_bytes_ = o.gate_bytes_ = o.up_bytes_ = o.down_bytes_ = 0;
        o.clock_ = 0;
    }

    struct Slot {
        uint64_t key = 0;
        uint64_t last_use = 0;
        bool valid = false;
    };

    static uint64_t make_key(int layer, int expert) {
        return (static_cast<uint64_t>(static_cast<uint32_t>(layer)) << 32)
             | static_cast<uint32_t>(expert);
    }

    static void copy_async(
            void * dst,
            const void * src,
            size_t bytes,
            cudaStream_t stream,
            const std::string & name) {
        const cudaError_t e = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
        if (e != cudaSuccess) {
            throw std::runtime_error("expert stage failed for " + name + ": " + cudaGetErrorString(e));
        }
    }

    void validate_geometry(
            const ExpertHostSlice & g,
            const ExpertHostSlice & u,
            const ExpertHostSlice & d) const {
        if (g.bytes != gate_bytes_ || g.rows != gate_rows_ || g.cols != gate_cols_ || g.type != gate_type_ ||
            u.bytes != up_bytes_   || u.rows != up_rows_     || u.cols != up_cols_     || u.type != up_type_ ||
            d.bytes != down_bytes_ || d.rows != down_rows_   || d.cols != down_cols_   || d.type != down_type_) {
            throw std::runtime_error("expert tensor geometry changed between layers");
        }
    }

    size_t choose_slot() const {
        for (size_t i = 0; i < slots_.size(); ++i) {
            if (!slots_[i].valid) return i;
        }
        size_t best = 0;
        uint64_t oldest = std::numeric_limits<uint64_t>::max();
        for (size_t i = 0; i < slots_.size(); ++i) {
            if (slots_[i].last_use < oldest) {
                oldest = slots_[i].last_use;
                best = i;
            }
        }
        return best;
    }

    ExpertDeviceWeights views(size_t slot, bool hit) const {
        uint8_t * base = pool_ + slot * slot_bytes_;
        ExpertDeviceWeights out;
        out.gate = borrow_lowbit_matrix(base, gate_rows_, gate_cols_, gate_type_, gate_bytes_);
        out.up = borrow_lowbit_matrix(base + gate_bytes_, up_rows_, up_cols_, up_type_, up_bytes_);
        out.down = borrow_lowbit_matrix(
            base + gate_bytes_ + up_bytes_, down_rows_, down_cols_, down_type_, down_bytes_);
        out.cache_hit = hit;
        out.slot = slot;
        return out;
    }

    uint8_t * pool_ = nullptr;
    size_t n_slots_ = 0;
    size_t slot_bytes_ = 0;
    size_t gate_bytes_ = 0;
    size_t up_bytes_ = 0;
    size_t down_bytes_ = 0;

    int gate_rows_ = 0, gate_cols_ = 0;
    int up_rows_ = 0, up_cols_ = 0;
    int down_rows_ = 0, down_cols_ = 0;
    GgmlType gate_type_ = GgmlType::F32;
    GgmlType up_type_ = GgmlType::F32;
    GgmlType down_type_ = GgmlType::F32;

    uint64_t clock_ = 0;
    std::vector<Slot> slots_;
    std::unordered_map<uint64_t, size_t> index_;
};

struct ExpertDeviceFusedWeights {
    LowBitMatrix gate_up;
    LowBitMatrix down;
    bool cache_hit = false;
    size_t slot = 0;
};

class GpuFusedExpertCache {
public:
    GpuFusedExpertCache() = default;
    GpuFusedExpertCache(const GpuFusedExpertCache &) = delete;
    GpuFusedExpertCache & operator=(const GpuFusedExpertCache &) = delete;

    GpuFusedExpertCache(GpuFusedExpertCache && o) noexcept { move_from(std::move(o)); }
    GpuFusedExpertCache & operator=(GpuFusedExpertCache && o) noexcept {
        if (this != &o) {
            release();
            move_from(std::move(o));
        }
        return *this;
    }

    ~GpuFusedExpertCache() { release(); }

    void initialize(
            size_t capacity_mib,
            const TensorInfo & sample_gate_up,
            const TensorInfo & sample_down) {
        if (pool_) throw std::runtime_error("fused expert cache already initialized");
        if (capacity_mib == 0) throw std::runtime_error("fused expert cache capacity is zero");

        const auto gu = expert_slice(sample_gate_up, 0);
        const auto d  = expert_slice(sample_down, 0);

        gate_up_bytes_ = gu.bytes;
        down_bytes_ = d.bytes;
        slot_bytes_ = gate_up_bytes_ + down_bytes_;
        if (slot_bytes_ == 0) throw std::runtime_error("fused expert cache slot is empty");

        gate_up_rows_ = gu.rows;
        gate_up_cols_ = gu.cols;
        gate_up_type_ = gu.type;
        down_rows_ = d.rows;
        down_cols_ = d.cols;
        down_type_ = d.type;

        const size_t capacity_bytes = capacity_mib * 1024ull * 1024ull;
        n_slots_ = capacity_bytes / slot_bytes_;
        if (n_slots_ == 0) {
            throw std::runtime_error("fused expert cache capacity is smaller than one expert slot");
        }

        const cudaError_t e = cudaMalloc(reinterpret_cast<void **>(&pool_), n_slots_ * slot_bytes_);
        if (e != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMalloc fused expert cache failed: ") + cudaGetErrorString(e));
        }
        slots_.resize(n_slots_);
    }

    size_t slot_count() const { return n_slots_; }
    size_t resident_count() const { return index_.size(); }

    ExpertDeviceFusedWeights acquire(
            int layer,
            int expert,
            const TensorInfo & gate_up,
            const TensorInfo & down,
            cudaStream_t stream) {
        if (!pool_) throw std::runtime_error("fused expert cache is not initialized");

        const uint64_t key = make_key(layer, expert);
        ++clock_;

        auto it = index_.find(key);
        if (it != index_.end()) {
            slots_[it->second].last_use = clock_;
            return views(it->second, true);
        }

        const auto guh = expert_slice(gate_up, expert);
        const auto dh = expert_slice(down, expert);
        if (guh.bytes != gate_up_bytes_ || guh.rows != gate_up_rows_ || guh.cols != gate_up_cols_ || guh.type != gate_up_type_ ||
            dh.bytes != down_bytes_ || dh.rows != down_rows_ || dh.cols != down_cols_ || dh.type != down_type_) {
            throw std::runtime_error("fused expert tensor geometry changed between layers");
        }

        const size_t slot = choose_slot();
        Slot & meta = slots_[slot];
        if (meta.valid) index_.erase(meta.key);

        uint8_t * base = pool_ + slot * slot_bytes_;
        copy_async(base, guh.data, gate_up_bytes_, stream, gate_up.name);
        copy_async(base + gate_up_bytes_, dh.data, down_bytes_, stream, down.name);

        meta.key = key;
        meta.last_use = clock_;
        meta.valid = true;
        index_[key] = slot;
        return views(slot, false);
    }

private:
    void release() {
        if (pool_) cudaFree(pool_);
        pool_ = nullptr;
        n_slots_ = 0;
        slot_bytes_ = gate_up_bytes_ = down_bytes_ = 0;
        slots_.clear();
        index_.clear();
    }

    void move_from(GpuFusedExpertCache && o) noexcept {
        pool_ = o.pool_;
        n_slots_ = o.n_slots_;
        slot_bytes_ = o.slot_bytes_;
        gate_up_bytes_ = o.gate_up_bytes_;
        down_bytes_ = o.down_bytes_;
        gate_up_rows_ = o.gate_up_rows_;
        gate_up_cols_ = o.gate_up_cols_;
        down_rows_ = o.down_rows_;
        down_cols_ = o.down_cols_;
        gate_up_type_ = o.gate_up_type_;
        down_type_ = o.down_type_;
        clock_ = o.clock_;
        slots_ = std::move(o.slots_);
        index_ = std::move(o.index_);

        o.pool_ = nullptr;
        o.n_slots_ = 0;
        o.slot_bytes_ = o.gate_up_bytes_ = o.down_bytes_ = 0;
        o.clock_ = 0;
    }

    struct Slot {
        uint64_t key = 0;
        uint64_t last_use = 0;
        bool valid = false;
    };

    static uint64_t make_key(int layer, int expert) {
        return (static_cast<uint64_t>(static_cast<uint32_t>(layer)) << 32)
             | static_cast<uint32_t>(expert);
    }

    static void copy_async(
            void * dst,
            const void * src,
            size_t bytes,
            cudaStream_t stream,
            const std::string & name) {
        const cudaError_t e = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
        if (e != cudaSuccess) {
            throw std::runtime_error("fused expert stage failed for " + name + ": " + cudaGetErrorString(e));
        }
    }

    size_t choose_slot() const {
        for (size_t i = 0; i < slots_.size(); ++i) if (!slots_[i].valid) return i;
        size_t best = 0;
        uint64_t oldest = std::numeric_limits<uint64_t>::max();
        for (size_t i = 0; i < slots_.size(); ++i) {
            if (slots_[i].last_use < oldest) {
                oldest = slots_[i].last_use;
                best = i;
            }
        }
        return best;
    }

    ExpertDeviceFusedWeights views(size_t slot, bool hit) const {
        uint8_t * base = pool_ + slot * slot_bytes_;
        ExpertDeviceFusedWeights out;
        out.gate_up = borrow_lowbit_matrix(
            base, gate_up_rows_, gate_up_cols_, gate_up_type_, gate_up_bytes_);
        out.down = borrow_lowbit_matrix(
            base + gate_up_bytes_, down_rows_, down_cols_, down_type_, down_bytes_);
        out.cache_hit = hit;
        out.slot = slot;
        return out;
    }

    uint8_t * pool_ = nullptr;
    size_t n_slots_ = 0;
    size_t slot_bytes_ = 0;
    size_t gate_up_bytes_ = 0;
    size_t down_bytes_ = 0;

    int gate_up_rows_ = 0, gate_up_cols_ = 0;
    int down_rows_ = 0, down_cols_ = 0;
    GgmlType gate_up_type_ = GgmlType::F32;
    GgmlType down_type_ = GgmlType::F32;

    uint64_t clock_ = 0;
    std::vector<Slot> slots_;
    std::unordered_map<uint64_t, size_t> index_;
};

} // namespace wd
