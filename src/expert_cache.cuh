#pragma once

#include "gguf.hpp"
#include "lowbit.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace wd {

// ---------------------------------------------------------------------------
// Task C: expert cache / host->device transfer switches.
//
// Every switch below only changes *how* an expert slot is filled or evicted,
// never the bytes that end up in the slot, so the greedy decode output is
// unchanged (checked with the 64/64 parity runner):
//
//   WARODECODE_EXPERT_STATS=1        counters + one JSON line on stderr at exit
//   WARODECODE_EXPERT_SHADOW=a,b,..  shadow-LRU capacities (slots) to simulate
//   WARODECODE_EXPERT_COPY_STREAM=0  DMA on the decode stream (default 1: a
//                                    private non-blocking stream, so the host
//                                    side of the staging copy and the DMA run
//                                    beside the kernels already queued)
//   WARODECODE_EXPERT_EVICT=layer    per-layer LRU partition instead of the
//                                    global LRU (default global)
//
// Task H adds the host read path of the same slot fill:
//
//   WARODECODE_EXPERT_IO_PREAD=0     read the slices with pread() into a pinned
//                                    ring and DMA from there (default 1; 0 gives
//                                    the old pageable cudaMemcpyAsync from the
//                                    MAP_PRIVATE mmap)
//   WARODECODE_EXPERT_IO_THREADS=n   pread worker threads (default 4, 1 = none)
//   WARODECODE_EXPERT_IO_RING=n      pinned ring entries (default 4)
//   WARODECODE_EXPERT_IO_FADVISE=willneed|sequential|random|dontneed
//                                    posix_fadvise() hint on the read range
//                                    (measured: no benefit, so it stays off)
//
// WARODECODE_EXPERT_STATS=1 also reports the read path of every fill
// (io_pread_ms / io_ring_wait_ms / io_served / io_fallbacks / io_enabled /
// io_ring / io_threads / io_fadvise).
//
// The layer count of the partition policy comes from the runtime trunk size.
//
// A pinned-ring staging candidate (WARODECODE_EXPERT_PIN: cudaMallocHost ring,
// one CPU memcpy per slice, DMA from pinned memory) was measured but
// withdrawn: at the 1536 MiB cache it produced wrong tokens (64/64 -> 35/64 and
// 10/64), so it must not be re-enabled before that is understood. See
// docs/ISSUE1_FINDINGS_20261012.md, 追記 26. 追記 27 (Task H) identifies the
// likely cause as ring reuse without a DMA-completion guard: the withdrawn ring
// overwrote a entry the DMA engine was still reading. The pread ring below puts
// one cudaEvent_t on every entry and calls cudaEventSynchronize() before the
// host writes into it again; the read path is otherwise a drop-in replacement
// for the pageable one (same bytes, same slot layout).
// ---------------------------------------------------------------------------
struct ExpertCacheOptions {
    bool stats = false;
    bool copy_stream = true;
    bool layer_evict = false;
    int layer_count = 0;
    std::vector<size_t> shadow_slots;

    // Task H: pread + pinned ring staging. ON by default because it was
    // measured faster in every configuration (A100 +41% tok/s at the 1536 MiB
    // cache, all 64/64); WARODECODE_EXPERT_IO_PREAD=0 restores the pageable
    // cudaMemcpyAsync path, and an unusable mapping falls back to it anyway.
    bool io_pread = true;
    int io_ring = 4;
    int io_threads = 4;
    bool io_stats = false;
};

// The switch values are parsed from the environment by qwen38_tuning.hpp
// (qwen38_tuning_for_gpu) and handed to the caches by the runtime, so this
// header stays free of process-wide state. The Task H switches below are read
// here instead, because they only steer the read path of this header and no
// other translation unit has to know about them.
inline bool expert_cache_env_flag(const char * name, bool fallback) {
    const char * v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return fallback;
    return v[0] != '0';
}

inline int expert_cache_env_int(const char * name, int fallback) {
    const char * v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return fallback;
    const long n = std::strtol(v, nullptr, 10);
    return n > 0 ? static_cast<int>(n) : fallback;
}

// posix_fadvise() advice for the range about to be read. Returns 0 for "none".
inline int expert_cache_env_fadvise() {
    const char * v = std::getenv("WARODECODE_EXPERT_IO_FADVISE");
    if (v == nullptr || v[0] == '\0') return 0;
    const std::string s(v);
    if (s == "willneed") return POSIX_FADV_WILLNEED;
    if (s == "sequential") return POSIX_FADV_SEQUENTIAL;
    if (s == "random") return POSIX_FADV_RANDOM;
    if (s == "dontneed") return POSIX_FADV_DONTNEED;
    return 0;
}


// ---------------------------------------------------------------------------
// Host file mapping discovery.
//
// The GGUF is mapped by MappedFile::MappedFile() with
// mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0) and the cache only ever
// receives raw pointers into that mapping. To read the same bytes with pread()
// the read path needs (a) an fd for the file and (b) the file offset of a
// pointer. Both are recovered from /proc/self/maps, which keeps the change
// inside this header instead of widening the runtime interface:
//
//     file_offset(p) = region.file_off + (p - region.start)
//
// A read-only file mapping is never written, so the bytes pread() returns are
// identical to the bytes the mapping would fault in; the only difference is
// that pread() copies the page cache page in one syscall instead of taking a
// fault per 64 KiB (measured 0.31 ms vs 0.56 ms per 2 MiB on the A100 Colab VM,
// tests/expert_io_bandwidth.cu).
// ---------------------------------------------------------------------------
struct ExpertMappingRegion {
    const uint8_t * start = nullptr;
    size_t length = 0;
    uint64_t file_off = 0;
    std::string path;
};

class ExpertFileMap {
public:
    ExpertFileMap() = default;
    ExpertFileMap(const ExpertFileMap &) = delete;
    ExpertFileMap & operator=(const ExpertFileMap &) = delete;

    ExpertFileMap(ExpertFileMap && o) noexcept { move_from(std::move(o)); }
    ExpertFileMap & operator=(ExpertFileMap && o) noexcept {
        if (this != &o) {
            release();
            move_from(std::move(o));
        }
        return *this;
    }

    ~ExpertFileMap() { release(); }

    // Resolve one pointer into the mmap to its file offset. Returns false when
    // the pointer is not backed by a readable file mapping (or the mapping
    // table changed under us), in which case the caller must use the pageable
    // path for that expert.
    bool resolve(const uint8_t * ptr) {
        if (ptr == nullptr) return false;
        for (int attempt = 0; attempt < 2; ++attempt) {
            const ExpertMappingRegion * region = find(ptr);
            if (region == nullptr) {
                if (attempt > 0) return false;
                if (!load(ptr)) return false;
                continue;
            }
            if (fd_ < 0 || region->path != path_) {
                if (fd_ >= 0) ::close(fd_);
                fd_ = ::open(region->path.c_str(), O_RDONLY);
                path_ = region->path;
                if (fd_ < 0) return false;
            }
            return true;
        }
        return false;
    }

    bool valid() const { return fd_ >= 0; }
    int fd() const { return fd_; }
    const std::vector<ExpertMappingRegion> & regions() const { return regions_; }

    uint64_t offset_of(const uint8_t * ptr) const {
        const ExpertMappingRegion * region = find(ptr);
        if (region == nullptr) return 0;
        return region->file_off + static_cast<uint64_t>(ptr - region->start);
    }

    void release() {
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
        regions_.clear();
        last_region_ = static_cast<size_t>(-1);
        path_.clear();
    }

private:
    void move_from(ExpertFileMap && o) noexcept {
        regions_ = std::move(o.regions_);
        fd_ = o.fd_;
        path_ = std::move(o.path_);
        o.fd_ = -1;
        o.regions_.clear();
        o.path_.clear();
    }

    const ExpertMappingRegion * find(const uint8_t * ptr) const {
        // Every expert of a layer lives in the same VMA, so remembering the last
        // hit turns the scan per miss into one comparison.
        if (last_region_ < regions_.size()) {
            const ExpertMappingRegion & r = regions_[last_region_];
            if (ptr >= r.start && ptr < r.start + r.length) return &r;
        }
        for (size_t i = 0; i < regions_.size(); ++i) {
            const ExpertMappingRegion & r = regions_[i];
            if (ptr >= r.start && ptr < r.start + r.length) {
                last_region_ = i;
                return &r;
            }
        }
        return nullptr;
    }

    // Parse /proc/self/maps and keep the readable, file backed regions. The
    // region containing ptr must be one of them for the read to be legal.
    bool load(const uint8_t * ptr) {
        regions_.clear();
        last_region_ = static_cast<size_t>(-1);
        std::FILE * f = std::fopen("/proc/self/maps", "r");
        if (f == nullptr) return false;
        char line[4096];
        bool found = false;
        while (std::fgets(line, sizeof(line), f) != nullptr) {
            unsigned long long lo = 0;
            unsigned long long hi = 0;
            char perms[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            unsigned long long pgoff = 0;
            unsigned int dev_major = 0;
            unsigned int dev_minor = 0;
            unsigned long long inode = 0;
            int consumed = 0;
            if (std::sscanf(line, "%llx-%llx %7s %llx %x:%x %llu %n",
                            &lo, &hi, perms, &pgoff, &dev_major, &dev_minor, &inode,
                            &consumed) != 7) {
                continue;
            }
            if (perms[0] != 'r') continue;
            const char * name = line + consumed;
            while (*name == ' ') ++name;
            const size_t len = std::strlen(name);
            if (len == 0) continue;
            // [heap], [stack], [vvar], anonymous mappings and deleted files are
            // not usable read sources.
            if (name[0] != '/') continue;
            std::string path(name);
            while (!path.empty() && (path.back() == '\n' || path.back() == ' ')) path.pop_back();
            if (path.empty() || path[0] != '/') continue;
            if (path.find(" (deleted)") != std::string::npos) continue;

            ExpertMappingRegion region;
            region.start = reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(lo));
            region.length = static_cast<size_t>(hi - lo);
            region.file_off = static_cast<uint64_t>(pgoff) * 4096ull;
            region.path = path;
            if (ptr >= region.start && ptr < region.start + region.length) found = true;
            regions_.push_back(std::move(region));
        }
        std::fclose(f);
        return found;
    }

    std::vector<ExpertMappingRegion> regions_;
    mutable size_t last_region_ = static_cast<size_t>(-1);
    int fd_ = -1;
    std::string path_;
};


// ---------------------------------------------------------------------------
// Task H: fork-join worker pool for the pread() side of the staging path.
//
// One thread pread()ing the ~1.85 MiB an expert needs reaches ~6 GB/s on the
// Colab VMs (tests/expert_io_bandwidth.cu); the same bytes split over 2 / 4
// threads reach 12.3 / 17.8 GB/s, because a single core cannot saturate the
// memory the page cache lives in. acquire() publishes one job per expert,
// works on its own share and returns only after every other share landed in the
// ring entry, so the DMA that follows always sees complete slices.
//
// threads == 1 keeps the whole fill inline (no threads are created).
// ---------------------------------------------------------------------------
class ExpertPreadPool {
public:
    ExpertPreadPool() = default;
    ExpertPreadPool(const ExpertPreadPool &) = delete;
    ExpertPreadPool & operator=(const ExpertPreadPool &) = delete;
    ~ExpertPreadPool() { stop(); }

    void start(int threads) {
        stop();
        threads_ = threads < 1 ? 1 : threads;
        if (threads_ == 1) return;
        running_.store(true, std::memory_order_release);
        generation_.store(0, std::memory_order_release);
        workers_.reserve(static_cast<size_t>(threads_) - 1);
        for (int i = 1; i < threads_; ++i) {
            workers_.emplace_back([this, i] { worker_loop(static_cast<size_t>(i)); });
        }
    }

    void stop() {
        if (workers_.empty()) {
            threads_ = 1;
            return;
        }
        running_.store(false, std::memory_order_release);
        for (std::thread & w : workers_) w.join();
        workers_.clear();
        threads_ = 1;
    }

    int threads() const { return threads_; }

    // Reads bytes[i] of slice i from file_off[i] into base + ring_off[i].
    // Returns false when a slice could not be fully read.
    bool run(int fd, uint8_t * base, const size_t * ring_off, const size_t * bytes,
             const uint64_t * file_off, int count) {
        job_.fd = fd;
        job_.base = base;
        job_.ring_off = ring_off;
        job_.bytes = bytes;
        job_.file_off = file_off;
        job_.count = count;
        // The ring space is the *slot* layout, and the three slices of a
        // separate gate/up/down cache do not tile it: the slot uses the largest
        // slice of any layer, so there are gaps. The shared space therefore ends
        // at the furthest slice end, not at the sum of the slice lengths (using
        // the sum silently truncated the last slice, which corrupted experts).
        total_ = 0;
        for (int i = 0; i < count; ++i) {
            const size_t end = ring_off[i] + bytes[i];
            if (end > total_) total_ = end;
        }
        failed_.store(false, std::memory_order_relaxed);
        if (total_ == 0) return true;

        if (threads_ == 1) {
            // No worker threads: read each slice whole, in file order.
            for (int i = 0; i < count; ++i) {
                if (!read_slice(i, ring_off[i], bytes[i])) return false;
            }
            return true;
        }
        pending_.store(threads_ - 1, std::memory_order_release);
        generation_.fetch_add(1, std::memory_order_release);
        {
            const std::lock_guard<std::mutex> lock(idle_mutex_);
            idle_cv_.notify_all();
        }
        const bool ok = share(0, static_cast<size_t>(threads_));
        while (pending_.load(std::memory_order_acquire) != 0) {
            std::this_thread::yield();
        }
        return ok && !failed_.load(std::memory_order_relaxed);
    }

private:
    struct Job {
        int fd = -1;
        uint8_t * base = nullptr;
        const size_t * ring_off = nullptr;
        const size_t * bytes = nullptr;
        const uint64_t * file_off = nullptr;
        int count = 0;
    };

    void worker_loop(size_t index) {
        uint64_t last = 0;
        int idle_rounds = 0;
        while (running_.load(std::memory_order_acquire)) {
            const uint64_t gen = generation_.load(std::memory_order_acquire);
            if (gen != last) {
                last = gen;
                idle_rounds = 0;
                if (!share(index, static_cast<size_t>(threads_))) {
                    failed_.store(true, std::memory_order_relaxed);
                }
                pending_.fetch_sub(1, std::memory_order_release);
                continue;
            }
            // acquires come back to back while a token runs, so spin first; the
            // decode also blocks on the GPU for ~200 ms per token, and burning a
            // core for that long would steal it from the launch thread.
            if (++idle_rounds < kSpinRounds) {
                std::this_thread::yield();
            } else {
                std::unique_lock<std::mutex> lock(idle_mutex_);
                idle_cv_.wait_for(lock, std::chrono::microseconds(500));
            }
        }
    }

    // Byte range [total*index/threads, total*(index+1)/threads) of the flattened
    // slice space, mapped back onto the slices.
    bool read_slice(int i, size_t from, size_t want) {
        const uint64_t file_off = job_.file_off[i] + (from - job_.ring_off[i]);
        size_t done = 0;
        while (done < want) {
            const ssize_t got = ::pread(job_.fd, job_.base + from + done, want - done,
                                        static_cast<off_t>(file_off + done));
            if (got <= 0) return false;
            done += static_cast<size_t>(got);
        }
        return true;
    }

    bool share(size_t index, size_t threads) {
        const size_t begin = total_ * index / threads;
        const size_t end = total_ * (index + 1) / threads;
        for (int i = 0; i < job_.count; ++i) {
            const size_t seg_begin = job_.ring_off[i];
            const size_t seg_end = seg_begin + job_.bytes[i];
            if (seg_end <= begin || seg_begin >= end) continue;
            const size_t from = seg_begin > begin ? seg_begin : begin;
            const size_t to = seg_end < end ? seg_end : end;
            if (!read_slice(i, from, to - from)) return false;
        }
        return true;
    }

    static constexpr int kSpinRounds = 4096;

    std::vector<std::thread> workers_;
    std::mutex idle_mutex_;
    std::condition_variable idle_cv_;
    Job job_{};
    size_t total_ = 0;
    int threads_ = 1;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> generation_{0};
    std::atomic<int> pending_{0};
    std::atomic<bool> failed_{false};
};


// ---------------------------------------------------------------------------
// Host->device path of one expert slot.
//
// Without a copy stream the slices are staged by the driver on the decode
// stream, so the host side of cudaMemcpyAsync() cannot start before every
// kernel queued so far has drained. With a copy stream the DMA is queued on a
// private non-blocking stream, and the only ordering it needs is the retire
// event of the slot it is about to overwrite. That event is recorded on the
// decode stream at the entry of the acquire() that follows the hand-out, i.e.
// after every kernel that reads the old content was queued, so the DMA may run
// beside the kernels of the previous expert.
//
// Task H adds a second fill path (WARODECODE_EXPERT_IO_PREAD=1) that replaces
// the pageable cudaMemcpyAsync by:
//
//   1. one pread() per slice into a cudaMallocHost ring entry, laid out exactly
//      like the destination slot, and
//   2. one cudaMemcpyAsync() per contiguous run from that pinned entry.
//
// The DMA then no longer stages through the driver's bounce buffer, so the host
// leaves the critical path (measured: 0.002 ms vs 0.31 ms of host time per
// 2 MiB) and the page faults of the mapping disappear (pread copies the page
// cache page in bulk). Every ring entry carries the event of the DMA that last
// read it and the host waits on it with cudaEventSynchronize() before writing
// again; the withdrawn WARODECODE_EXPERT_PIN ring did not do that, which is the
// most likely reason it corrupted experts under high churn (64/64 -> 35/64 at
// the 1536 MiB cache).
//
// Any failure on the pread path (unresolvable pointer, open() failure, short
// read) falls back to the pageable copy for that one expert, so the bytes that
// reach the slot are always the model bytes.
// ---------------------------------------------------------------------------
class ExpertStagePath {
public:
    ExpertStagePath() = default;
    ExpertStagePath(const ExpertStagePath &) = delete;
    ExpertStagePath & operator=(const ExpertStagePath &) = delete;

    ExpertStagePath(ExpertStagePath && o) noexcept { move_from(std::move(o)); }
    ExpertStagePath & operator=(ExpertStagePath && o) noexcept {
        if (this != &o) {
            release();
            move_from(std::move(o));
        }
        return *this;
    }

    ~ExpertStagePath() { release(); }

    void configure(const ExpertCacheOptions & opt, size_t slot_bytes, const uint8_t * sample) {
        release();
        stage_enabled_ = opt.copy_stream;
        ring_slot_bytes_ = slot_bytes;
        io_enabled_ = expert_cache_env_flag("WARODECODE_EXPERT_IO_PREAD", opt.io_pread);
        io_ring_entries_ = static_cast<size_t>(expert_cache_env_int("WARODECODE_EXPERT_IO_RING", opt.io_ring));
        if (io_ring_entries_ == 0) io_ring_entries_ = 1;
        io_threads_ = expert_cache_env_int("WARODECODE_EXPERT_IO_THREADS", opt.io_threads);
        if (io_threads_ < 1) io_threads_ = 1;
        io_fadvise_ = expert_cache_env_fadvise();
        if (stage_enabled_ && opt.copy_stream) {
            cudaError_t e = cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking);
            if (e != cudaSuccess) {
                copy_stream_ = nullptr;
                throw std::runtime_error(std::string("expert copy stream create failed: ") + cudaGetErrorString(e));
            }
            e = cudaEventCreateWithFlags(&done_, cudaEventDisableTiming);
            if (e != cudaSuccess) {
                throw std::runtime_error(std::string("expert done event create failed: ") + cudaGetErrorString(e));
            }
        }
        if (io_enabled_) {
            // Fail soft: a cache that cannot pread must still be able to stage.
            // verify_mapping() is the parity guard of the whole path: it checks
            // that pread(offset_of(p)) really returns the bytes at p for one
            // whole slot, so a bad /proc/self/maps parse can only ever fall back
            // to the pageable path, never fill a slot with the wrong expert.
            if (!map_.resolve(sample) || !map_.valid() || !verify_mapping(sample, slot_bytes)) {
                std::fprintf(stderr, "expert-io: pread path disabled (mapping verify failed)\n");
                io_enabled_ = false;
            } else {
                ring_.resize(io_ring_entries_);
                for (RingEntry & entry : ring_) {
                    entry.buf = nullptr;
                    entry.done = nullptr;
                    entry.pending = false;
                    if (cudaMallocHost(reinterpret_cast<void **>(&entry.buf), ring_slot_bytes_) != cudaSuccess) {
                        entry.buf = nullptr;
                        io_enabled_ = false;
                        break;
                    }
                    if (cudaEventCreateWithFlags(&entry.done, cudaEventDisableTiming) != cudaSuccess) {
                        entry.done = nullptr;
                        io_enabled_ = false;
                        break;
                    }
                }
                if (io_enabled_) {
                    pool_ = std::make_unique<ExpertPreadPool>();
                    pool_->start(io_threads_);
                } else {
                    for (RingEntry & entry : ring_) {
                        if (entry.done) cudaEventDestroy(entry.done);
                        if (entry.buf) cudaFreeHost(entry.buf);
                        entry.done = nullptr;
                        entry.buf = nullptr;
                    }
                    ring_.clear();
                }
            }
        }
    }

    void release() {
        if (pool_) {
            pool_->stop();
            pool_.reset();
        }
        for (RingEntry & entry : ring_) {
            if (entry.done) cudaEventDestroy(entry.done);
            if (entry.buf) cudaFreeHost(entry.buf);
            entry.done = nullptr;
            entry.buf = nullptr;
            entry.pending = false;
        }
        ring_.clear();
        map_.release();
        if (done_) cudaEventDestroy(done_);
        done_ = nullptr;
        if (copy_stream_) cudaStreamDestroy(copy_stream_);
        copy_stream_ = nullptr;
        stage_enabled_ = false;
        io_enabled_ = false;
    }

    bool enabled() const { return stage_enabled_; }
    bool io_enabled() const { return io_enabled_; }
    size_t ring_entries() const { return ring_.size(); }
    int fadvise_mode() const { return io_fadvise_; }
    int io_threads() const { return io_threads_; }
    bool io_busy() const { return pool_ != nullptr && pool_->threads() > 1; }
    int fd() const { return map_.fd(); }
    size_t slot_bytes() const { return ring_slot_bytes_; }
    // Retire events are only needed when the DMA is decoupled from the decode
    // stream; the plain path keeps the stream order.
    bool needs_retire() const { return copy_stream_ != nullptr; }

    // Queue one expert: count slices land at dst + offset[i]. depend orders the
    // DMA after the last consumer of the slot that is being overwritten.
    // Returns true when the pinned pread path served the fill.
    bool submit(
            const uint8_t * const * src,
            const size_t * offset,
            const size_t * bytes,
            int count,
            uint8_t * dst,
            cudaStream_t consumer,
            cudaEvent_t depend) {
        // The pinned path works with and without the copy stream: without it the
        // DMA and its reuse event are ordered on the decode stream instead.
        cudaStream_t s = copy_stream_ ? copy_stream_ : consumer;
        if (io_enabled_ && fill_pinned(src, offset, bytes, count)) {
            wait_depend(s, depend);
            RingEntry & entry = ring_[ring_next_];
            ring_next_ = (ring_next_ + 1) % ring_.size();
            // Contiguous runs of the slot are one DMA, which keeps the transfer
            // per expert at two calls (gate_up + down) instead of four.
            int i = 0;
            while (i < count) {
                int j = i + 1;
                while (j < count && offset[j] == offset[j - 1] + bytes[j - 1]) ++j;
                copy(dst + offset[i], entry.buf + offset[i], offset[j - 1] + bytes[j - 1] - offset[i], s, "expert");
                i = j;
            }
            // The host must not write this entry again before the DMA above has
            // read it: record the event the reuse guard waits on.
            copy_record(entry.done, s, "expert ring reuse event");
            entry.pending = true;
            return true;
        }
        if (!stage_enabled_) {
            for (int i = 0; i < count; ++i) {
                copy(dst + offset[i], src[i], bytes[i], consumer, "expert");
            }
            return false;
        }
        wait_depend(s, depend);
        for (int i = 0; i < count; ++i) {
            copy(dst + offset[i], src[i], bytes[i], s, "expert");
        }
        return false;
    }

    // Make the decode stream wait for everything submit() queued.
    void finish(cudaStream_t consumer) {
        if (copy_stream_ == nullptr) return;
        copy_record(done_, copy_stream_, "expert done event");
        const cudaError_t e = cudaStreamWaitEvent(consumer, done_, 0);
        if (e != cudaSuccess) throw std::runtime_error(std::string("expert done wait failed: ") + cudaGetErrorString(e));
    }

    double last_pread_ms() const { return last_pread_ms_; }
    double last_ring_wait_ms() const { return last_ring_wait_ms_; }

private:
    struct RingEntry {
        uint8_t * buf = nullptr;
        cudaEvent_t done = nullptr;   // DMA that last read buf
        bool pending = false;
    };

    void move_from(ExpertStagePath && o) noexcept {
        copy_stream_ = o.copy_stream_;
        done_ = o.done_;
        stage_enabled_ = o.stage_enabled_;
        io_enabled_ = o.io_enabled_;
        io_fadvise_ = o.io_fadvise_;
        io_ring_entries_ = o.io_ring_entries_;
        ring_ = std::move(o.ring_);
        pool_ = std::move(o.pool_);
        io_threads_ = o.io_threads_;
        ring_next_ = o.ring_next_;
        ring_slot_bytes_ = o.ring_slot_bytes_;
        map_ = std::move(o.map_);
        o.copy_stream_ = nullptr;
        o.done_ = nullptr;
        o.stage_enabled_ = false;
        o.io_enabled_ = false;
        o.ring_next_ = 0;
        o.ring_slot_bytes_ = 0;
        o.ring_.clear();
        o.pool_.reset();
        o.io_threads_ = 1;
    }

    void wait_depend(cudaStream_t s, cudaEvent_t depend) {
        if (copy_stream_ && depend != nullptr) {
            const cudaError_t e = cudaStreamWaitEvent(copy_stream_, depend, 0);
            if (e != cudaSuccess) throw std::runtime_error(std::string("expert retire wait failed: ") + cudaGetErrorString(e));
        }
        (void)s;
    }

    // One-shot consistency check between the mapping and the file: the bytes
    // pread() returns for the sample pointer must equal the mapped bytes.
    bool verify_mapping(const uint8_t * ptr, size_t bytes) {
        if (ptr == nullptr || bytes == 0) return false;
        // Never read past the end of the mapping.
        const ExpertMappingRegion * region = nullptr;
        for (const ExpertMappingRegion & r : map_.regions()) {
            if (ptr >= r.start && ptr < r.start + r.length) {
                region = &r;
                break;
            }
        }
        if (region == nullptr) return false;
        bytes = std::min(bytes, static_cast<size_t>(region->start + region->length - ptr));
        const uint64_t file_off = region->file_off + static_cast<uint64_t>(ptr - region->start);
        const size_t window = 64 * 1024;
        std::vector<uint8_t> buf(window);
        for (size_t pos = 0; pos < bytes; pos += window) {
            const size_t n = std::min(window, bytes - pos);
            const ssize_t got = ::pread(map_.fd(), buf.data(), n, static_cast<off_t>(file_off + pos));
            if (got != static_cast<ssize_t>(n)) return false;
            if (std::memcmp(buf.data(), ptr + pos, n) != 0) return false;
        }
        return true;
    }

    // Fill a pinned ring entry from the file. Returns false (and leaves the ring
    // untouched) when any slice cannot be resolved or read.
    bool fill_pinned(const uint8_t * const * src, const size_t * offset, const size_t * bytes, int count) {
        size_t total = 0;
        for (int i = 0; i < count; ++i) total = std::max(total, offset[i] + bytes[i]);
        if (total > ring_slot_bytes_ || ring_.empty()) return false;
        RingEntry & entry = ring_[ring_next_];
        if (entry.buf == nullptr) return false;

        const auto wait_start = clock::now();
        if (entry.pending) {
            // Reuse guard: the previous DMA that read this entry must be done.
            const cudaError_t e = cudaEventSynchronize(entry.done);
            if (e != cudaSuccess && e != cudaErrorNotReady) {
                (void)cudaGetLastError();
                return false;
            }
            if (e == cudaErrorNotReady) return false;
            entry.pending = false;
        }
        const double wait_ms = ms_since(wait_start);
        last_ring_wait_ms_ = wait_ms;

        const auto read_start = clock::now();
        uint64_t file_offs[8];
        if (count > 8) return false;
        for (int i = 0; i < count; ++i) {
            if (!map_.resolve(src[i])) return false;
            file_offs[i] = map_.offset_of(src[i]);
            if (io_fadvise_ != 0) {
                (void)posix_fadvise(map_.fd(), static_cast<off_t>(file_offs[i]),
                                    static_cast<off_t>(bytes[i]), io_fadvise_);
            }
        }
        bool ok = true;
        if (pool_) {
            ok = pool_->run(map_.fd(), entry.buf, offset, bytes, file_offs, count);
        } else {
            for (int i = 0; i < count && ok; ++i) {
                size_t done = 0;
                while (done < bytes[i]) {
                    const ssize_t n = ::pread(map_.fd(), entry.buf + offset[i] + done,
                                              bytes[i] - done,
                                              static_cast<off_t>(file_offs[i] + done));
                    if (n <= 0) { ok = false; break; }
                    done += static_cast<size_t>(n);
                }
            }
        }
        last_pread_ms_ = ms_since(read_start);
        return ok;
    }

    static void copy(void * dst, const void * src, size_t bytes, cudaStream_t stream, const char * what) {
        const cudaError_t e = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
        if (e != cudaSuccess) {
            throw std::runtime_error(std::string(what) + " stage failed: " + cudaGetErrorString(e));
        }
    }

    static void copy_record(cudaEvent_t ev, cudaStream_t stream, const char * what) {
        const cudaError_t e = cudaEventRecord(ev, stream);
        if (e != cudaSuccess) {
            throw std::runtime_error(std::string(what) + " record failed: " + cudaGetErrorString(e));
        }
    }

    using clock = std::chrono::steady_clock;

    static double ms_since(const clock::time_point & t) {
        return std::chrono::duration<double, std::milli>(clock::now() - t).count();
    }

    cudaStream_t copy_stream_ = nullptr;
    cudaEvent_t done_ = nullptr;
    bool stage_enabled_ = false;

    bool io_enabled_ = false;
    int io_fadvise_ = 0;
    size_t io_ring_entries_ = 0;
    size_t ring_next_ = 0;
    size_t ring_slot_bytes_ = 0;
    int io_threads_ = 1;
    std::vector<RingEntry> ring_;
    std::unique_ptr<ExpertPreadPool> pool_;
    ExpertFileMap map_;

    double last_pread_ms_ = 0.0;
    double last_ring_wait_ms_ = 0.0;
};


// ---------------------------------------------------------------------------
// Statistics. Host side bookkeeping of the acquire() stream only; the copied
// bytes are never touched. A "token" is detected as a decreasing layer index
// in the acquire order (the decode loop walks layer 0..N-1 once per token), so
// the forward pass does not have to report anything.
// ---------------------------------------------------------------------------
class ShadowLru {
public:
    explicit ShadowLru(size_t capacity) : capacity_(capacity ? capacity : 1) {}

    void access(uint64_t key) {
        auto it = index_.find(key);
        if (it != index_.end()) {
            ++hits_;
            order_.splice(order_.begin(), order_, it->second);
            return;
        }
        order_.push_front(key);
        index_[key] = order_.begin();
        if (order_.size() > capacity_) {
            index_.erase(order_.back());
            order_.pop_back();
        }
    }

    uint64_t hits() const { return hits_; }
    size_t capacity() const { return capacity_; }

private:
    size_t capacity_ = 1;
    uint64_t hits_ = 0;
    std::list<uint64_t> order_;
    std::unordered_map<uint64_t, std::list<uint64_t>::iterator> index_;
};

class ExpertCacheMonitor {
public:
    struct ShadowSet {
        size_t total_slots = 0;
        std::vector<ShadowLru> layers;
        uint64_t hits() const {
            uint64_t h = 0;
            for (const ShadowLru & l : layers) h += l.hits();
            return h;
        }
    };

    void configure(
            const ExpertCacheOptions & opt,
            const char * mode,
            size_t capacity_mib,
            size_t slots,
            size_t slot_bytes,
            int layer_count) {
        enabled_ = opt.stats;
        mode_ = mode ? mode : "?";
        capacity_mib_ = capacity_mib;
        slots_ = slots;
        slot_bytes_ = slot_bytes;
        layer_count_ = layer_count > 0 ? layer_count : 1;
        layer_access_.assign(static_cast<size_t>(layer_count_), 0);
        layer_hits_.assign(static_cast<size_t>(layer_count_), 0);
        layer_misses_.assign(static_cast<size_t>(layer_count_), 0);
        shadow_global_.clear();
        shadow_layer_.clear();
        for (size_t s : opt.shadow_slots) {
            shadow_global_.push_back(ShadowLru(s));
            ShadowSet set;
            set.total_slots = s;
            const size_t per = s / static_cast<size_t>(layer_count_);
            set.layers.reserve(static_cast<size_t>(layer_count_));
            for (int i = 0; i < layer_count_; ++i) set.layers.push_back(ShadowLru(per ? per : 1));
            shadow_layer_.push_back(std::move(set));
        }
        if (!enabled_) return;
        std::fprintf(stderr,
            "expert-cache mode=%s capacity_mib=%zu slots=%zu slot_bytes=%zu copy_stream=%d "
            "evict=%s layers=%d shadow=%zu io=%s\n",
            mode_.c_str(), capacity_mib_, slots_, slot_bytes_,
            opt.copy_stream ? 1 : 0,
            opt.layer_evict ? "layer" : "global", layer_count_, shadow_global_.size(),
            opt.io_pread ? "pread" : "pageable");
        std::fflush(stderr);
    }

    // A moved-from cache must not print an empty record.
    void discard() { enabled_ = false; }

    bool enabled() const { return enabled_; }

    void note_access(int layer, uint64_t key) {
        if (!enabled_) return;
        if (last_layer_ >= 0 && layer < last_layer_) close_token();
        last_layer_ = layer;
        const size_t li = static_cast<size_t>(layer >= 0 ? layer : 0);
        ensure_layer(li);
        if (cur_.access == 0) token_start_ = clock::now();
        ++accesses_;
        ++layer_access_[li];
        ++cur_.access;
        cur_.set.insert(key);
        if (seen_.insert(key).second) ++distinct_;
        if (!prev_set_.empty()) {
            ++prev_ready_;
            if (prev_set_.count(key)) ++prev_hit_;
        }
        for (ShadowLru & s : shadow_global_) s.access(key);
        for (ShadowSet & set : shadow_layer_) {
            if (li < set.layers.size()) set.layers[li].access(key);
        }
    }

    void note_hit(int layer) {
        if (!enabled_) return;
        ++hits_;
        ++cur_.hits;
        const size_t li = static_cast<size_t>(layer >= 0 ? layer : 0);
        ensure_layer(li);
        ++layer_hits_[li];
    }

    void note_fill(int layer, size_t bytes, bool evicted) {
        if (!enabled_) return;
        ++misses_;
        if (evicted) ++evictions_;
        bytes_copied_ += bytes;
        cur_.bytes += bytes;
        ++cur_.misses;
        const size_t li = static_cast<size_t>(layer >= 0 ? layer : 0);
        ensure_layer(li);
        ++layer_misses_[li];
    }

    void note_stage_ms(double ms) {
        if (!enabled_) return;
        stage_ms_ += ms;
        cur_.stage_ms += ms;
    }

    void note_io_configured(bool enabled, size_t ring, int fadvise, int threads) {
        io_enabled_ = enabled;
        io_ring_ = ring;
        io_fadvise_ = fadvise;
        io_threads_ = threads;
    }

    // Task H: pread / pinned ring read path, split out of stage_ms.
    void note_io_ms(double pread_ms, double ring_wait_ms, bool served) {
        if (!enabled_) return;
        pread_ms_ += pread_ms;
        ring_wait_ms_ += ring_wait_ms;
        if (served) {
            ++io_served_;
        } else {
            ++io_fallbacks_;
        }
    }

    void note_acquire_ms(double ms) {
        if (!enabled_) return;
        acquire_ms_ += ms;
    }

    void dump() {
        if (!enabled_) return;
        if (cur_.access > 0) close_token();
        const double token_count = static_cast<double>(token_access_.empty() ? 1 : token_access_.size());
        std::string s = "{";
        s += "\"mode\":\"" + mode_ + "\"";
        s += ",\"capacity_mib\":" + std::to_string(capacity_mib_);
        s += ",\"slots\":" + std::to_string(slots_);
        s += ",\"slot_bytes\":" + std::to_string(slot_bytes_);
        s += ",\"accesses\":" + std::to_string(accesses_);
        s += ",\"hits\":" + std::to_string(hits_);
        s += ",\"misses\":" + std::to_string(misses_);
        s += ",\"hit_rate\":" + fixed(accesses_ ? static_cast<double>(hits_) / static_cast<double>(accesses_) : 0.0, 4);
        s += ",\"evictions\":" + std::to_string(evictions_);
        s += ",\"bytes_copied\":" + std::to_string(bytes_copied_);
        s += ",\"gib_copied\":" + fixed(static_cast<double>(bytes_copied_) / (1024.0 * 1024.0 * 1024.0), 3);
        s += ",\"stage_ms\":" + fixed(stage_ms_, 1);
        s += ",\"acquire_ms\":" + fixed(acquire_ms_, 1);
        s += ",\"stage_share\":" + fixed(acquire_ms_ > 0.0 ? stage_ms_ / acquire_ms_ : 0.0, 4);
        s += ",\"io_pread_ms\":" + fixed(pread_ms_, 1);
        s += ",\"io_ring_wait_ms\":" + fixed(ring_wait_ms_, 1);
        s += ",\"io_served\":" + std::to_string(io_served_);
        s += ",\"io_fallbacks\":" + std::to_string(io_fallbacks_);
        s += ",\"io_enabled\":" + std::string(io_enabled_ ? "true" : "false");
        s += ",\"io_ring\":" + std::to_string(io_ring_);
        s += ",\"io_fadvise\":" + std::to_string(io_fadvise_);
        s += ",\"io_threads\":" + std::to_string(io_threads_);
        s += ",\"tokens\":" + std::to_string(token_access_.size());
        s += ",\"distinct_pairs\":" + std::to_string(distinct_);
        s += ",\"prev_token_ready\":" + std::to_string(prev_ready_);
        s += ",\"prev_token_reuse\":" + fixed(prev_ready_ ? static_cast<double>(prev_hit_) / static_cast<double>(prev_ready_) : 0.0, 4);
        s += ",\"host_ms_per_token\":" + fixed(host_ms_total() / token_count, 2);
        s += ",\"layer_access\":" + array_u64(layer_access_);
        s += ",\"layer_misses\":" + array_u64(layer_misses_);
        s += ",\"token_miss\":" + array_u64(token_miss_);
        s += ",\"token_stage_ms\":" + array_f64(token_stage_ms_);
        s += ",\"token_host_ms\":" + array_f64(token_host_ms_);
        s += ",\"shadow\":[";
        for (size_t i = 0; i < shadow_global_.size(); ++i) {
            if (i) s += ",";
            const uint64_t gh = shadow_global_[i].hits();
            const uint64_t lh = shadow_layer_[i].hits();
            s += "{\"slots\":" + std::to_string(shadow_global_[i].capacity());
            s += ",\"global_hits\":" + std::to_string(gh);
            s += ",\"global_hit_rate\":" + fixed(accesses_ ? static_cast<double>(gh) / static_cast<double>(accesses_) : 0.0, 4);
            s += ",\"layer_hits\":" + std::to_string(lh);
            s += ",\"layer_hit_rate\":" + fixed(accesses_ ? static_cast<double>(lh) / static_cast<double>(accesses_) : 0.0, 4);
            s += "}";
        }
        s += "]}";
        std::fprintf(stderr, "{\"expert_cache_stats\":%s}\n", s.c_str());
        std::fflush(stderr);
    }

private:
    using clock = std::chrono::steady_clock;

    struct TokenCounters {
        uint64_t access = 0;
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t bytes = 0;
        double stage_ms = 0.0;
        std::unordered_set<uint64_t> set;
    };

    static double ms_since(const clock::time_point & t) {
        return std::chrono::duration<double, std::milli>(clock::now() - t).count();
    }

    static std::string fixed(double v, int digits) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.*f", digits, v);
        return std::string(buf);
    }

    static std::string array_u64(const std::vector<uint64_t> & values) {
        std::string s = "[";
        for (size_t i = 0; i < values.size(); ++i) {
            if (i) s += ",";
            s += std::to_string(values[i]);
        }
        s += "]";
        return s;
    }

    static std::string array_f64(const std::vector<double> & values) {
        std::string s = "[";
        for (size_t i = 0; i < values.size(); ++i) {
            if (i) s += ",";
            s += fixed(values[i], 2);
        }
        s += "]";
        return s;
    }

    void ensure_layer(size_t li) {
        if (li < layer_access_.size()) return;
        const size_t n = li + 1;
        layer_access_.resize(n, 0);
        layer_hits_.resize(n, 0);
        layer_misses_.resize(n, 0);
    }

    void close_token() {
        token_access_.push_back(cur_.access);
        token_hits_.push_back(cur_.hits);
        token_miss_.push_back(cur_.misses);
        token_bytes_.push_back(cur_.bytes);
        token_stage_ms_.push_back(cur_.stage_ms);
        token_host_ms_.push_back(ms_since(token_start_));
        prev_set_.swap(cur_.set);
        cur_ = TokenCounters{};
        last_layer_ = -1;
    }

    double host_ms_total() const {
        double total = 0.0;
        for (double v : token_host_ms_) total += v;
        return total;
    }

    bool enabled_ = false;
    std::string mode_ = "?";
    size_t capacity_mib_ = 0;
    size_t slots_ = 0;
    size_t slot_bytes_ = 0;
    int layer_count_ = 1;

    uint64_t accesses_ = 0;
    uint64_t hits_ = 0;
    uint64_t misses_ = 0;
    uint64_t evictions_ = 0;
    uint64_t bytes_copied_ = 0;
    uint64_t distinct_ = 0;
    uint64_t prev_ready_ = 0;
    uint64_t prev_hit_ = 0;
    double stage_ms_ = 0.0;
    double acquire_ms_ = 0.0;
    double pread_ms_ = 0.0;
    double ring_wait_ms_ = 0.0;
    uint64_t io_served_ = 0;
    uint64_t io_fallbacks_ = 0;
    bool io_enabled_ = false;
    size_t io_ring_ = 0;
    int io_fadvise_ = 0;
    int io_threads_ = 1;

    int last_layer_ = -1;
    clock::time_point token_start_{};
    TokenCounters cur_;
    std::unordered_set<uint64_t> prev_set_;
    std::unordered_set<uint64_t> seen_;

    std::vector<uint64_t> layer_access_;
    std::vector<uint64_t> layer_hits_;
    std::vector<uint64_t> layer_misses_;
    std::vector<uint64_t> token_access_;
    std::vector<uint64_t> token_hits_;
    std::vector<uint64_t> token_miss_;
    std::vector<uint64_t> token_bytes_;
    std::vector<double> token_stage_ms_;
    std::vector<double> token_host_ms_;

    std::vector<ShadowLru> shadow_global_;
    std::vector<ShadowSet> shadow_layer_;
};


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

    ~GpuExpertCache() {
        release();
        monitor_.dump();
    }

    void initialize(
            size_t capacity_mib,
            const TensorInfo & sample_gate,
            const TensorInfo & sample_up,
            const TensorInfo & sample_down,
            size_t gate_capacity_bytes = 0,
            size_t up_capacity_bytes = 0,
            size_t down_capacity_bytes = 0,
            const ExpertCacheOptions & options = ExpertCacheOptions{}) {
        if (pool_) throw std::runtime_error("expert cache already initialized");
        if (capacity_mib == 0) throw std::runtime_error("expert cache capacity is zero");

        const auto g = expert_slice(sample_gate, 0);
        const auto u = expert_slice(sample_up, 0);
        const auto d = expert_slice(sample_down, 0);

        gate_bytes_ = gate_capacity_bytes ? gate_capacity_bytes : g.bytes;
        up_bytes_ = up_capacity_bytes ? up_capacity_bytes : u.bytes;
        down_bytes_ = down_capacity_bytes ? down_capacity_bytes : d.bytes;
        if (gate_bytes_ < g.bytes || up_bytes_ < u.bytes || down_bytes_ < d.bytes) {
            throw std::runtime_error("expert cache slot capacity is smaller than sample tensors");
        }
        slot_bytes_ = gate_bytes_ + up_bytes_ + down_bytes_;
        if (slot_bytes_ == 0) throw std::runtime_error("expert cache slot is empty");

        gate_rows_ = g.rows; gate_cols_ = g.cols;
        up_rows_   = u.rows; up_cols_   = u.cols;
        down_rows_ = d.rows; down_cols_ = d.cols;

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

        options_ = options;
        layer_evict_ = options.layer_evict;
        layer_count_ = options.layer_count;
        stage_.configure(options_, slot_bytes_, sample_gate.data);
        monitor_.configure(options_, "separate", capacity_mib, n_slots_, slot_bytes_, options.layer_count);
        monitor_.note_io_configured(stage_.io_enabled(), stage_.ring_entries(), stage_.fadvise_mode(), stage_.io_threads());
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
        const auto acquire_start = clock::now();

        // Every kernel that reads the slot handed out by the previous acquire()
        // has been queued by now, so this is the point after which that slot may
        // be overwritten by an asynchronous DMA.
        if (last_slot_valid_ && stage_.needs_retire()) retire_slot(last_slot_, stream);

        const uint64_t key = make_key(layer, expert);
        ++clock_;
        monitor_.note_access(layer, key);

        auto it = index_.find(key);
        if (it != index_.end()) {
            Slot & s = slots_[it->second];
            s.last_use = clock_;
            monitor_.note_hit(layer);
            monitor_.note_acquire_ms(ms_since(acquire_start));
            return views(it->second, true);
        }

        const auto gh = expert_slice(gate, expert);
        const auto uh = expert_slice(up, expert);
        const auto dh = expert_slice(down, expert);
        validate_geometry(gh, uh, dh);

        const size_t slot = choose_slot(layer);
        Slot & meta = slots_[slot];
        const bool evicted = meta.valid;
        if (evicted) index_.erase(meta.key);

        uint8_t * base = pool_ + slot * slot_bytes_;
        const uint8_t * src[3] = {gh.data, uh.data, dh.data};
        const size_t offset[3] = {0, gate_bytes_, gate_bytes_ + up_bytes_};
        const size_t length[3] = {gh.bytes, uh.bytes, dh.bytes};
        const auto stage_start = clock::now();
        const bool io_served = stage_.submit(src, offset, length, 3, base, stream, meta.retire);
        stage_.finish(stream);
        monitor_.note_stage_ms(ms_since(stage_start));
        monitor_.note_io_ms(stage_.last_pread_ms(), stage_.last_ring_wait_ms(), io_served);

        meta.key = key;
        meta.last_use = clock_;
        meta.valid = true;
        meta.gate_type = gh.type;
        meta.up_type = uh.type;
        meta.down_type = dh.type;
        meta.gate_actual_bytes = gh.bytes;
        meta.up_actual_bytes = uh.bytes;
        meta.down_actual_bytes = dh.bytes;
        index_[key] = slot;
        monitor_.note_fill(layer, gh.bytes + uh.bytes + dh.bytes, evicted);
        monitor_.note_acquire_ms(ms_since(acquire_start));
        return views(slot, false);
    }

private:
    void release() {
        for (Slot & s : slots_) {
            if (s.retire) cudaEventDestroy(s.retire);
            s.retire = nullptr;
        }
        stage_.release();
        if (pool_) cudaFree(pool_);
        pool_ = nullptr;
        n_slots_ = 0;
        slot_bytes_ = gate_bytes_ = up_bytes_ = down_bytes_ = 0;
        slots_.clear();
        index_.clear();
        last_slot_ = 0;
        last_slot_valid_ = false;
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
        clock_ = o.clock_;
        slots_ = std::move(o.slots_);
        index_ = std::move(o.index_);
        stage_ = std::move(o.stage_);
        monitor_ = std::move(o.monitor_);
        o.monitor_.discard();
        options_ = o.options_;
        layer_evict_ = o.layer_evict_;
        layer_count_ = o.layer_count_;
        last_slot_ = o.last_slot_;
        last_slot_valid_ = o.last_slot_valid_;

        o.pool_ = nullptr;
        o.n_slots_ = 0;
        o.slot_bytes_ = o.gate_bytes_ = o.up_bytes_ = o.down_bytes_ = 0;
        o.clock_ = 0;
        o.last_slot_ = 0;
        o.last_slot_valid_ = false;
    }

    struct Slot {
        uint64_t key = 0;
        uint64_t last_use = 0;
        bool valid = false;
        GgmlType gate_type = GgmlType::F32;
        GgmlType up_type = GgmlType::F32;
        GgmlType down_type = GgmlType::F32;
        size_t gate_actual_bytes = 0;
        size_t up_actual_bytes = 0;
        size_t down_actual_bytes = 0;
        cudaEvent_t retire = nullptr;
    };

    using clock = std::chrono::steady_clock;

    static double ms_since(const clock::time_point & t) {
        return std::chrono::duration<double, std::milli>(clock::now() - t).count();
    }

    static uint64_t make_key(int layer, int expert) {
        return (static_cast<uint64_t>(static_cast<uint32_t>(layer)) << 32)
             | static_cast<uint32_t>(expert);
    }

    void retire_slot(size_t slot, cudaStream_t stream) {
        Slot & s = slots_[slot];
        if (s.retire == nullptr) {
            if (cudaEventCreateWithFlags(&s.retire, cudaEventDisableTiming) != cudaSuccess) {
                s.retire = nullptr;
                return;
            }
        }
        cudaEventRecord(s.retire, stream);
    }

    void validate_geometry(
            const ExpertHostSlice & g,
            const ExpertHostSlice & u,
            const ExpertHostSlice & d) const {
        if (g.rows != gate_rows_ || g.cols != gate_cols_ ||
            u.rows != up_rows_ || u.cols != up_cols_ ||
            d.rows != down_rows_ || d.cols != down_cols_) {
            throw std::runtime_error("expert tensor geometry changed between layers");
        }
        if (g.bytes > gate_bytes_ || u.bytes > up_bytes_ || d.bytes > down_bytes_) {
            throw std::runtime_error("expert tensor exceeds cache slot capacity");
        }
    }

    size_t choose_slot(int layer) const {
        const size_t n = slots_.size();
        size_t begin = 0;
        size_t end = n;
        if (layer_evict_ && layer_count_ > 0 && static_cast<size_t>(layer_count_) <= n && layer >= 0) {
            const size_t li = static_cast<size_t>(layer);
            if (li < static_cast<size_t>(layer_count_)) {
                const size_t per = n / static_cast<size_t>(layer_count_);
                const size_t rem = n % static_cast<size_t>(layer_count_);
                begin = li * per + std::min(li, rem);
                end = begin + per + (li < rem ? 1u : 0u);
            }
        }
        if (end <= begin) { begin = 0; end = n; }
        for (size_t i = begin; i < end; ++i) {
            if (!slots_[i].valid) return i;
        }
        size_t best = begin;
        uint64_t oldest = std::numeric_limits<uint64_t>::max();
        for (size_t i = begin; i < end; ++i) {
            if (slots_[i].last_use < oldest) {
                oldest = slots_[i].last_use;
                best = i;
            }
        }
        return best;
    }

    ExpertDeviceWeights views(size_t slot, bool hit) {
        uint8_t * base = pool_ + slot * slot_bytes_;
        const Slot & meta = slots_[slot];
        ExpertDeviceWeights out;
        out.gate = borrow_lowbit_matrix(
            base, gate_rows_, gate_cols_, meta.gate_type, meta.gate_actual_bytes);
        out.up = borrow_lowbit_matrix(
            base + gate_bytes_, up_rows_, up_cols_, meta.up_type, meta.up_actual_bytes);
        out.down = borrow_lowbit_matrix(
            base + gate_bytes_ + up_bytes_, down_rows_, down_cols_,
            meta.down_type, meta.down_actual_bytes);
        out.cache_hit = hit;
        out.slot = slot;
        last_slot_ = slot;
        last_slot_valid_ = true;
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
    uint64_t clock_ = 0;
    std::vector<Slot> slots_;
    std::unordered_map<uint64_t, size_t> index_;

    ExpertCacheOptions options_;
    ExpertStagePath stage_;
    ExpertCacheMonitor monitor_;
    bool layer_evict_ = false;
    int layer_count_ = 0;
    size_t last_slot_ = 0;
    bool last_slot_valid_ = false;
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

    ~GpuFusedExpertCache() {
        release();
        monitor_.dump();
    }

    void initialize(
            size_t capacity_mib,
            const TensorInfo & sample_gate_up,
            const TensorInfo & sample_down,
            const ExpertCacheOptions & options = ExpertCacheOptions{}) {
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

        options_ = options;
        layer_evict_ = options.layer_evict;
        layer_count_ = options.layer_count;
        stage_.configure(options_, slot_bytes_, sample_gate_up.data);
        monitor_.configure(options_, "fused", capacity_mib, n_slots_, slot_bytes_, options.layer_count);
        monitor_.note_io_configured(stage_.io_enabled(), stage_.ring_entries(), stage_.fadvise_mode(), stage_.io_threads());
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
        const auto acquire_start = clock::now();

        if (last_slot_valid_ && stage_.needs_retire()) retire_slot(last_slot_, stream);

        const uint64_t key = make_key(layer, expert);
        ++clock_;
        monitor_.note_access(layer, key);

        auto it = index_.find(key);
        if (it != index_.end()) {
            slots_[it->second].last_use = clock_;
            monitor_.note_hit(layer);
            monitor_.note_acquire_ms(ms_since(acquire_start));
            return views(it->second, true);
        }

        const auto guh = expert_slice(gate_up, expert);
        const auto dh = expert_slice(down, expert);
        if (guh.bytes != gate_up_bytes_ || guh.rows != gate_up_rows_ || guh.cols != gate_up_cols_ || guh.type != gate_up_type_ ||
            dh.bytes != down_bytes_ || dh.rows != down_rows_ || dh.cols != down_cols_ || dh.type != down_type_) {
            throw std::runtime_error("fused expert tensor geometry changed between layers");
        }

        const size_t slot = choose_slot(layer);
        Slot & meta = slots_[slot];
        const bool evicted = meta.valid;
        if (evicted) index_.erase(meta.key);

        uint8_t * base = pool_ + slot * slot_bytes_;
        const uint8_t * src[2] = {guh.data, dh.data};
        const size_t offset[2] = {0, gate_up_bytes_};
        const size_t length[2] = {guh.bytes, dh.bytes};
        const auto stage_start = clock::now();
        const bool io_served = stage_.submit(src, offset, length, 2, base, stream, meta.retire);
        stage_.finish(stream);
        monitor_.note_stage_ms(ms_since(stage_start));
        monitor_.note_io_ms(stage_.last_pread_ms(), stage_.last_ring_wait_ms(), io_served);

        meta.key = key;
        meta.last_use = clock_;
        meta.valid = true;
        index_[key] = slot;
        monitor_.note_fill(layer, guh.bytes + dh.bytes, evicted);
        monitor_.note_acquire_ms(ms_since(acquire_start));
        return views(slot, false);
    }

private:
    void release() {
        for (Slot & s : slots_) {
            if (s.retire) cudaEventDestroy(s.retire);
            s.retire = nullptr;
        }
        stage_.release();
        if (pool_) cudaFree(pool_);
        pool_ = nullptr;
        n_slots_ = 0;
        slot_bytes_ = gate_up_bytes_ = down_bytes_ = 0;
        slots_.clear();
        index_.clear();
        last_slot_ = 0;
        last_slot_valid_ = false;
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
        stage_ = std::move(o.stage_);
        monitor_ = std::move(o.monitor_);
        o.monitor_.discard();
        options_ = o.options_;
        layer_evict_ = o.layer_evict_;
        layer_count_ = o.layer_count_;
        last_slot_ = o.last_slot_;
        last_slot_valid_ = o.last_slot_valid_;

        o.pool_ = nullptr;
        o.n_slots_ = 0;
        o.slot_bytes_ = o.gate_up_bytes_ = o.down_bytes_ = 0;
        o.clock_ = 0;
        o.last_slot_ = 0;
        o.last_slot_valid_ = false;
    }

    struct Slot {
        uint64_t key = 0;
        uint64_t last_use = 0;
        bool valid = false;
        cudaEvent_t retire = nullptr;
    };

    using clock = std::chrono::steady_clock;

    static double ms_since(const clock::time_point & t) {
        return std::chrono::duration<double, std::milli>(clock::now() - t).count();
    }

    static uint64_t make_key(int layer, int expert) {
        return (static_cast<uint64_t>(static_cast<uint32_t>(layer)) << 32)
             | static_cast<uint32_t>(expert);
    }

    void retire_slot(size_t slot, cudaStream_t stream) {
        Slot & s = slots_[slot];
        if (s.retire == nullptr) {
            if (cudaEventCreateWithFlags(&s.retire, cudaEventDisableTiming) != cudaSuccess) {
                s.retire = nullptr;
                return;
            }
        }
        cudaEventRecord(s.retire, stream);
    }

    size_t choose_slot(int layer) const {
        const size_t n = slots_.size();
        size_t begin = 0;
        size_t end = n;
        if (layer_evict_ && layer_count_ > 0 && static_cast<size_t>(layer_count_) <= n && layer >= 0) {
            const size_t li = static_cast<size_t>(layer);
            if (li < static_cast<size_t>(layer_count_)) {
                const size_t per = n / static_cast<size_t>(layer_count_);
                const size_t rem = n % static_cast<size_t>(layer_count_);
                begin = li * per + std::min(li, rem);
                end = begin + per + (li < rem ? 1u : 0u);
            }
        }
        if (end <= begin) { begin = 0; end = n; }
        for (size_t i = begin; i < end; ++i) {
            if (!slots_[i].valid) return i;
        }
        size_t best = begin;
        uint64_t oldest = std::numeric_limits<uint64_t>::max();
        for (size_t i = begin; i < end; ++i) {
            if (slots_[i].last_use < oldest) {
                oldest = slots_[i].last_use;
                best = i;
            }
        }
        return best;
    }

    ExpertDeviceFusedWeights views(size_t slot, bool hit) {
        uint8_t * base = pool_ + slot * slot_bytes_;
        ExpertDeviceFusedWeights out;
        out.gate_up = borrow_lowbit_matrix(
            base, gate_up_rows_, gate_up_cols_, gate_up_type_, gate_up_bytes_);
        out.down = borrow_lowbit_matrix(
            base + gate_up_bytes_, down_rows_, down_cols_, down_type_, down_bytes_);
        out.cache_hit = hit;
        out.slot = slot;
        last_slot_ = slot;
        last_slot_valid_ = true;
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

    ExpertCacheOptions options_;
    ExpertStagePath stage_;
    ExpertCacheMonitor monitor_;
    bool layer_evict_ = false;
    int layer_count_ = 0;
    size_t last_slot_ = 0;
    bool last_slot_valid_ = false;
};

} // namespace wd
