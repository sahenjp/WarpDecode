// Host test for the Task H pread pool (ExpertPreadPool in src/expert_cache.cuh).
//
// The pool reads the ring layout of an expert slot, which for the separate
// gate/up/down cache contains *gaps*: the slot uses the largest slice of any
// layer, so the three slices do not tile it. A first version distributed the
// work over "sum of slice lengths" bytes and silently truncated the last slice,
// which corrupted every expert and broke the 64/64 parity (matched_prefix=0).
// This test pins the corrected contract down:
//
//   * with 1, 2, 3 and 4 threads the full slice contents land in the ring,
//   * slice contents are taken from their own file offsets (not from a running
//     position in the file),
//   * a gap between slices is never written and never read past,
//   * a slice that points at a bad file offset fails the whole job.
//
// It includes expert_cache.cuh, so it needs the CUDA stubs that
// tests/qwen38_cuda_syntax_check.sh installs in $Q38_CUDA_SYNTAX_WORK/stub.
//
// Usage: expert_io_pool_host_test
#include "expert_cache.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
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

std::string temp_path() {
    char tmpl[] = "/tmp/warodecode-io-pool-XXXXXX";
    const int fd = ::mkstemp(tmpl);
    if (fd < 0) {
        std::perror("mkstemp");
        std::exit(1);
    }
    ::close(fd);
    return std::string(tmpl);
}

}  // namespace

int main() {
    const size_t file_size = 1u << 20;
    std::vector<uint8_t> disk(file_size);
    for (size_t i = 0; i < file_size; ++i) disk[i] = static_cast<uint8_t>((i * 131 + (i >> 8)) & 0xff);

    const std::string path = temp_path();
    {
        const int fd = ::open(path.c_str(), O_WRONLY | O_TRUNC);
        if (fd < 0) { std::perror("open"); return 1; }
        if (::write(fd, disk.data(), disk.size()) != static_cast<ssize_t>(disk.size())) {
            std::perror("write");
            return 1;
        }
        ::close(fd);
    }

    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { std::perror("open read"); return 1; }

    // Ring layout with a gap: [0,4096) then a 1024 byte hole, then [5120, 11264),
    // then another hole, then [12288, 18432) -- the shape a real slot has when
    // the cache reserves the largest slice of any layer.
    const size_t n = 3;
    const size_t ring_off[n] = {0, 5120, 12288};
    const size_t bytes[n] = {4096, 6144, 6144};
    const uint64_t file_off[n] = {100000, 300000, 700000};
    const size_t span = ring_off[n - 1] + bytes[n - 1];

    for (int threads = 1; threads <= 4; ++threads) {
        wd::ExpertPreadPool pool;
        pool.start(threads);
        std::vector<uint8_t> ring(span, 0xcd);
        const bool ok = pool.run(fd, ring.data(), ring_off, bytes, file_off, static_cast<int>(n));
        check(ok, "pool.run succeeds with " + std::to_string(threads) + " thread(s)");
        bool content_ok = true;
        for (size_t i = 0; i < n; ++i) {
            if (std::memcmp(ring.data() + ring_off[i], disk.data() + file_off[i], bytes[i]) != 0) {
                content_ok = false;
            }
        }
        check(content_ok, "every slice matches its own file offset (" + std::to_string(threads) + " thread(s))");
        check(ring[4096] == 0xcd && ring[12288 - 1] == 0xcd && ring[18432 - 1] != 0xcd,
              "the gaps between slices are left untouched (" + std::to_string(threads) + " thread(s))");
        pool.stop();
    }

    // A slice whose file offset lies past EOF must fail, not return stale bytes.
    {
        wd::ExpertPreadPool pool;
        pool.start(2);
        const uint64_t bad[n] = {100000, static_cast<uint64_t>(file_size) + 4096, 700000};
        std::vector<uint8_t> ring(span, 0);
        const bool ok = pool.run(fd, ring.data(), ring_off, bytes, bad, static_cast<int>(n));
        check(!ok, "a slice past EOF fails the job");
        pool.stop();
    }

    ::close(fd);
    ::unlink(path.c_str());
    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
