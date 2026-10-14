// Host test for the Task K expert read-ahead ring (ExpertReadAhead and the
// ExpertStagePath::submit() read-ahead branch of src/expert_cache.cuh).
//
// The read-ahead ring is the part of the expert staging path that can silently
// put the wrong bytes in a slot: a prediction is a *guess* about which expert a
// later acquire() will want, and the guess is turned into a DMA source buffer
// before the acquire arrives. This test pins the contract down:
//
//   * a completed prediction serves the fill and the acquiring thread never
//     calls pread() for it (last_pread_ms() == 0, last_served_prefetch() true),
//   * the bytes a prefetched fill writes into the slot are bit identical to the
//     bytes the on-demand fill writes, which is what keeps the 64/64 parity of
//     the greedy decode independent of the prediction,
//   * a key that was never prefetched takes the unchanged on-demand path,
//   * the same key is never queued twice,
//   * a read that fails (a file offset past EOF) never serves a fill and does
//     not wedge the ring,
//   * the prediction is the previous token's per-layer list, it is evaluated
//     once per layer, it never contains the expert being acquired, and a depth
//     of 1 never reaches into the next layer.
//
// It includes expert_cache.cuh, so it needs the CUDA stubs that
// tests/qwen38_cuda_syntax_check.sh installs in $Q38_CUDA_SYNTAX_WORK/stub.
// tests/expert_prefetch_host_check.sh additionally patches the *generated* stub
// so cudaMallocHost() really allocates and cudaMemcpyAsync() really copies:
// without that the read-ahead buffer is a null pointer and no byte can move.
//
// Usage: expert_prefetch_host_test
#include "expert_cache.cuh"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

// The syntax-check stubs declare the CUDA builtins as extern const globals and
// the syntax check never links, so this linked test supplies the definitions.
const q38_stub_dim3 threadIdx = q38_stub_dim3();
const q38_stub_dim3 blockIdx = q38_stub_dim3();
const q38_stub_dim3 blockDim = q38_stub_dim3(1024, 1, 1);
const q38_stub_dim3 gridDim = q38_stub_dim3(1, 1, 1);

namespace {

int failures = 0;

void check(bool ok, const std::string & what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    } else {
        std::printf("ok: %s\n", what.c_str());
    }
}

constexpr size_t kFileSize = 4u << 20;
constexpr size_t kSlice0FileOff = 1u << 20;
constexpr size_t kSlice1FileOff = 3u << 20;
constexpr size_t kSlice0Bytes = 8192;
constexpr size_t kSlice1Bytes = 8192;
constexpr size_t kSlice1RingOff = 12288;   // a gap, like a real gate/up/down slot
constexpr size_t kSlotBytes = kSlice1RingOff + kSlice1Bytes;

uint64_t key_of(int layer, int expert) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(layer)) << 32) |
           static_cast<uint32_t>(expert);
}

bool contains(const std::vector<uint64_t> & v, uint64_t key) {
    for (uint64_t k : v) {
        if (k == key) return true;
    }
    return false;
}

// Waits for a background counter to reach want, so the test never assumes the
// reader thread was scheduled.
bool wait_for(const std::function<uint64_t()> & get, uint64_t want, int timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (get() >= want) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return get() >= want;
}

std::string temp_path() {
    char tmpl[] = "/tmp/warodecode-prefetch-XXXXXX";
    const int fd = ::mkstemp(tmpl);
    if (fd < 0) {
        std::perror("mkstemp");
        std::exit(1);
    }
    return std::string(tmpl);
}

}  // namespace

int main() {
    std::vector<uint8_t> disk(kFileSize);
    for (size_t i = 0; i < kFileSize; ++i) {
        disk[i] = static_cast<uint8_t>((i * 131 + (i >> 8) + 7) & 0xff);
    }

    const std::string path = temp_path();
    {
        const int wfd = ::open(path.c_str(), O_WRONLY | O_TRUNC);
        if (wfd < 0) { std::perror("open write"); return 1; }
        if (::write(wfd, disk.data(), disk.size()) != static_cast<ssize_t>(disk.size())) {
            std::perror("write");
            return 1;
        }
        ::close(wfd);
    }
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { std::perror("open read"); return 1; }
    void * raw = ::mmap(nullptr, kFileSize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (raw == MAP_FAILED) { std::perror("mmap"); return 1; }
    const uint8_t * map = static_cast<const uint8_t *>(raw);

    const uint8_t * src[2] = {map + kSlice0FileOff, map + kSlice1FileOff};
    const size_t offset[2] = {0, kSlice1RingOff};
    const size_t length[2] = {kSlice0Bytes, kSlice1Bytes};

    // -----------------------------------------------------------------------
    // ExpertReadAhead on its own: content, duplicate rejection, failure.
    // -----------------------------------------------------------------------
    {
        wd::ExpertReadAhead ring;
        check(ring.configure(2, 1, kSlotBytes), "the read-ahead ring allocates pinned entries");
        check(ring.enabled() && ring.slots() == 2, "the ring reports its size");

        const uint64_t file_off[1] = {kSlice0FileOff};
        const size_t one_off[1] = {0};
        const size_t one_len[1] = {kSlice0Bytes};
        check(ring.post(11, fd, file_off, one_off, one_len, 1), "a job is published");
        check(!ring.post(11, fd, file_off, one_off, one_len, 1), "the same key is not queued twice");
        check(wait_for([&] { return ring.delivered(); }, 1, 10000), "the reader completes the job");

        cudaEvent_t ev = nullptr;
        uint8_t * buf = ring.borrow(11, &ev);
        check(buf != nullptr, "a completed entry is handed out");
        check(buf != nullptr &&
              std::memcmp(buf, disk.data() + kSlice0FileOff, kSlice0Bytes) == 0,
              "the read-ahead buffer holds the file bytes of its slice");
        check(ring.borrow(12, &ev) == nullptr, "a key that was never queued is not handed out");

        // A prediction that cannot be read: the offset lies past EOF.
        const uint64_t bad_off[1] = {static_cast<uint64_t>(kFileSize) + 4096};
        check(ring.post(13, fd, bad_off, one_off, one_len, 1), "a job past EOF is published");
        check(wait_for([&] { return ring.failed(); }, 1, 10000), "the reader reports the failure");
        check(ring.borrow(13, &ev) == nullptr, "a failed read never serves a fill");
        check(ring.post(14, fd, file_off, one_off, one_len, 1),
              "the ring still accepts jobs after a failed read");
        ring.release();
        check(!ring.enabled(), "release() drops the ring");
    }

    // -----------------------------------------------------------------------
    // ExpertStagePath: prediction history, prefetched fill, byte parity.
    // -----------------------------------------------------------------------
    wd::ExpertCacheOptions opt;
    opt.io_pread = true;
    opt.io_ring = 2;
    opt.io_threads = 2;
    opt.copy_stream = true;
    opt.prefetch_depth = 1;
    opt.prefetch_threads = 2;
    opt.prefetch_slots = 4;

    wd::ExpertStagePath stage;
    stage.configure(opt, kSlotBytes, map + kSlice0FileOff);
    check(stage.io_enabled(), "the pinned pread path is enabled by the mapping check");
    check(stage.prefetch_enabled(), "the read-ahead ring is enabled");
    check(stage.prefetch_slots() == 4, "the ring is sized from the option");

    // Token 1: layer 0 selects three experts, layer 1 selects one.
    stage.note_access(0, key_of(0, 1));
    stage.note_access(0, key_of(0, 2));
    stage.note_access(0, key_of(0, 3));
    stage.note_access(1, key_of(1, 5));
    // Token 2: the layer index decreases, so the list above is now the prediction.
    const std::vector<uint64_t> & candidates = stage.note_access(0, key_of(0, 9));
    check(candidates.size() == 3, "the prediction is the previous token's list of this layer");
    check(contains(candidates, key_of(0, 1)) && contains(candidates, key_of(0, 2)) &&
          contains(candidates, key_of(0, 3)), "every expert of that list is a candidate");
    check(!contains(candidates, key_of(0, 9)), "the expert being acquired is never a candidate");
    check(!contains(candidates, key_of(1, 5)), "a depth of 1 does not read into the next layer");
    const std::vector<uint64_t> & second = stage.note_access(0, key_of(0, 4));
    check(second.empty(), "the prediction is evaluated once per layer");

    const uint64_t prefetched_key = key_of(2, 7);
    const uint64_t on_demand_key = key_of(2, 8);
    check(stage.post_prefetch(prefetched_key, src, offset, length, 2),
          "stage publishes a read-ahead job for a registered key");
    check(wait_for([&] { return stage.prefetch_delivered(); }, 1, 10000),
          "the stage reader completes the job");

    std::vector<uint8_t> dst_prefetch(kSlotBytes, 0);
    const bool prefetch_served = stage.submit(
        src, offset, length, 2, dst_prefetch.data(), nullptr, nullptr, prefetched_key);
    check(prefetch_served, "a completed prediction serves the fill");
    check(stage.last_served_prefetch(), "the fill reports the read-ahead ring as its source");
    check(stage.last_pread_ms() == 0.0, "the acquiring thread did not pread for a prediction");
    check(stage.prefetch_used() == 1, "the used counter follows the hand-out");

    std::vector<uint8_t> dst_on_demand(kSlotBytes, 0);
    const bool on_demand_served = stage.submit(
        src, offset, length, 2, dst_on_demand.data(), nullptr, nullptr, on_demand_key);
    check(on_demand_served, "a key that was never prefetched still takes the on-demand path");
    check(!stage.last_served_prefetch(), "the on-demand fill does not claim a prediction");
    check(stage.last_pread_ms() > 0.0, "the on-demand fill reads with pread()");

    check(std::memcmp(dst_prefetch.data(), dst_on_demand.data(), kSlotBytes) == 0,
          "the prefetched fill and the on-demand fill are byte identical");
    check(std::memcmp(dst_prefetch.data(), disk.data() + kSlice0FileOff, kSlice0Bytes) == 0,
          "the prefetched fill carries slice 0 from its own file offset");
    check(std::memcmp(dst_prefetch.data() + kSlice1RingOff,
                      disk.data() + kSlice1FileOff, kSlice1Bytes) == 0,
          "the prefetched fill carries slice 1 from its own file offset");
    check(dst_prefetch[kSlice0Bytes] == 0 && dst_prefetch[kSlice1RingOff - 1] == 0,
          "the gap between the slices is left untouched");

    // A prediction may also be wrong on purpose: the ring must simply not serve
    // it and the on-demand path must still produce the right bytes.
    const uint64_t wrong_key = key_of(3, 11);
    check(stage.post_prefetch(wrong_key, src, offset, length, 2), "a wrong prediction is queued");
    check(wait_for([&] { return stage.prefetch_delivered(); }, 2, 10000),
          "the reader completes the wrong prediction too");
    std::vector<uint8_t> dst_wrong(kSlotBytes, 0);
    const bool wrong_served = stage.submit(
        src, offset, length, 2, dst_wrong.data(), nullptr, nullptr, key_of(3, 12));
    check(wrong_served && !stage.last_served_prefetch(),
          "an acquire whose key was not predicted takes the on-demand path");
    check(std::memcmp(dst_wrong.data(), dst_on_demand.data(), kSlotBytes) == 0,
          "the fallback fill is still byte identical");

    stage.release();
    check(!stage.prefetch_enabled(), "release() stops the read-ahead");

    // -----------------------------------------------------------------------
    // GpuExpertCache end to end: registration -> prediction -> post -> serve.
    //
    // Two fake layers of four Q8_0 experts live in the mapped file. The slot
    // holds one expert, so with a 1 MiB cache only two slots exist and the third
    // acquire of the first token evicts the first one: that is what makes the
    // second token's prediction point at bytes that are *not* resident, which is
    // the only situation in which a read-ahead is worth posting.
    // -----------------------------------------------------------------------
    {
        const int hidden = 512;
        const int ffn = 256;
        const int experts = 4;
        wd::TensorInfo layer_tensor[2][3];
        const size_t per_expert = static_cast<size_t>(hidden) * ffn / 32 * 34;
        const size_t per_tensor = per_expert * static_cast<size_t>(experts);
        const char * names[3] = {"gate", "up", "down"};
        for (int l = 0; l < 2; ++l) {
            for (int s = 0; s < 3; ++s) {
                wd::TensorInfo & t = layer_tensor[l][s];
                t.name = std::string(names[s]) + std::to_string(l);
                t.dims = {static_cast<uint64_t>(hidden), static_cast<uint64_t>(ffn),
                          static_cast<uint64_t>(experts)};
                t.type = wd::GgmlType::Q8_0;
                t.data = map + (static_cast<size_t>(l) * 3 + static_cast<size_t>(s)) * per_tensor;
            }
        }

        wd::ExpertCacheOptions copts;
        copts.io_pread = true;
        copts.copy_stream = false;
        copts.layer_count = 2;
        copts.prefetch_depth = 1;
        copts.prefetch_threads = 2;
        copts.prefetch_slots = 4;

        wd::GpuExpertCache cache;
        cache.initialize(1, layer_tensor[0][0], layer_tensor[0][1], layer_tensor[0][2],
                         0, 0, 0, copts);
        check(cache.prefetch_depth() == 1,
              "the cache forwards the read-ahead depth to the staging path");
        cache.register_layer_experts(0, layer_tensor[0][0], layer_tensor[0][1], layer_tensor[0][2]);
        cache.register_layer_experts(1, layer_tensor[1][0], layer_tensor[1][1], layer_tensor[1][2]);

        // Token 1: layer 0 uses experts 0 and 1, layer 1 uses expert 0.
        cache.acquire(0, 0, layer_tensor[0][0], layer_tensor[0][1], layer_tensor[0][2], nullptr);
        cache.acquire(0, 1, layer_tensor[0][0], layer_tensor[0][1], layer_tensor[0][2], nullptr);
        cache.acquire(1, 0, layer_tensor[1][0], layer_tensor[1][1], layer_tensor[1][2], nullptr);
        check(cache.prefetch_posted() == 0, "the first token has no history to predict from");

        // Token 2: layer 0 is entered again, so the previous token's layer 0 list
        // is now the prediction, and expert 0 was evicted by the layer 1 acquire.
        cache.acquire(0, 1, layer_tensor[0][0], layer_tensor[0][1], layer_tensor[0][2], nullptr);
        check(cache.prefetch_posted() >= 1, "the second token posts a read-ahead job");
        check(wait_for([&] { return cache.prefetch_delivered(); }, 1, 10000),
              "the read-ahead reader completes it");
        cache.acquire(0, 0, layer_tensor[0][0], layer_tensor[0][1], layer_tensor[0][2], nullptr);
        check(cache.prefetch_used() >= 1, "the predicted expert is served from the read-ahead ring");
        check(cache.prefetch_failed() == 0, "no read-ahead read failed");
    }

    ::munmap(raw, kFileSize);
    ::close(fd);
    ::unlink(path.c_str());

    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
