// Task H: isolate which part of the expert staging path costs the 3.7 GB/s.
//
// The engine copies one expert slice (~2 MiB) per cache miss with
// cudaMemcpyAsync() from a MAP_PRIVATE mmap of the GGUF file. This probe runs
// the same byte volume through each candidate read path so the winning one can
// be chosen from numbers instead of guesses:
//
//   mmap                     : memcpy from the mapping (page faults included)
//   mmap+willneed            : madvise(MADV_WILLNEED) first
//   mmap+populate            : madvise(MADV_POPULATE_READ) first
//   mmap+h2d                 : what the engine does today
//   pageable                 : malloc'd buffer + memcpy  (control)
//   pageable+h2d             : malloc'd buffer + cudaMemcpyAsync H2D (control)
//   pread                    : pread() into a malloc'd buffer
//   pread+h2d                : pread() into a pageable buffer, then H2D
//   pinpread                 : pread() into cudaMallocHost (pinned)
//   pinpread+h2d             : pread() into pinned, then H2D from pinned
//   pin+h2d                  : H2D from a already-filled pinned buffer
//   odirect                  : O_DIRECT pread into an aligned buffer
//   readahead                : readahead(2) then memcpy from the mapping
//
// Every case is timed over the same random chunk list; the reported GB/s is
// bytes / wall time of the *host side* (what the engine measures as stage_ms).
// H2D cases additionally report the time including the device sync.
//
// Usage: expert_io_bandwidth <model.gguf> [chunks] [chunk_mib] [cases...]
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

double now_s() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

long page_faults_minor() {
    struct rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_minflt;
}

long page_faults_major() {
    struct rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_majflt;
}

void die(const char * what) {
    std::fprintf(stderr, "error: %s (%s)\n", what, std::strerror(errno));
    std::exit(1);
}

struct Case {
    std::string name;
    double gb_s = 0.0;
    double synced_gb_s = 0.0;
    double ms_per_chunk = 0.0;
    long minflt = 0;
    long majflt = 0;
};

struct Sample {
    size_t off = 0;
    bool resident = false;
};

size_t chunk_resident_bytes(const uint8_t * map, size_t file_size, size_t off, size_t bytes, long page) {
    const size_t first = off & ~static_cast<size_t>(page - 1);
    const size_t last = (off + bytes + page - 1) & ~static_cast<size_t>(page - 1);
    const size_t n = (last - first) / static_cast<size_t>(page);
    std::vector<unsigned char> vec(n ? n : 1);
    const size_t aligned_off = first;
    if (mincore(const_cast<uint8_t *>(map) + aligned_off, last - first, vec.data()) != 0) {
        return bytes;
    }
    size_t resident = 0;
    for (size_t i = 0; i < n; ++i) {
        if (vec[i] & 1) resident += static_cast<size_t>(page);
    }
    return resident;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf> [chunks] [chunk_mib] [case...]\n", argv[0]);
        return 1;
    }
    const char * path = argv[1];
    const int n_chunks = argc > 2 ? std::atoi(argv[2]) : 128;
    const double chunk_mib = argc > 3 ? std::atof(argv[3]) : 2.0;
    std::vector<std::string> want;
    for (int i = 4; i < argc; ++i) want.push_back(argv[i]);

    const long page = sysconf(_SC_PAGESIZE);
    const size_t chunk = static_cast<size_t>(chunk_mib * 1048576.0 + 0.5);

    const int fd = ::open(path, O_RDONLY);
    if (fd < 0) die("open");
    struct stat st{};
    if (fstat(fd, &st) != 0) die("fstat");
    const size_t file_size = static_cast<size_t>(st.st_size);
    uint8_t * map = static_cast<uint8_t *>(mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (map == MAP_FAILED) die("mmap");

    // Deterministic chunk list across the weight region (skip the metadata head).
    std::mt19937_64 rng(0x5157454eu);
    const size_t lo = 64ull * 1024 * 1024;
    const size_t hi = file_size > chunk ? file_size - chunk : lo + chunk;
    std::vector<Sample> samples;
    samples.reserve(static_cast<size_t>(n_chunks));
    for (int i = 0; i < n_chunks; ++i) {
        std::uniform_int_distribution<size_t> dist(lo, hi - 1);
        Sample s;
        s.off = (dist(rng) / page) * page;
        s.resident = chunk_resident_bytes(map, file_size, s.off, chunk, page) >= chunk;
        samples.push_back(s);
    }
    size_t resident_chunks = 0;
    for (const Sample & s : samples) resident_chunks += s.resident ? 1 : 0;
    std::printf("{\"file_size\":%zu,\"chunks\":%d,\"chunk_bytes\":%zu,\"chunks_resident\":%zu,"
                "\"resident_share\":%.4f}\n",
                file_size, n_chunks, chunk, resident_chunks,
                static_cast<double>(resident_chunks) / static_cast<double>(n_chunks));

    uint8_t * host_buf = static_cast<uint8_t *>(std::malloc(chunk));
    uint8_t * host_raw = nullptr;
    if (posix_memalign(reinterpret_cast<void **>(&host_raw), 4096, chunk) != 0) die("memalign");
    uint8_t * pinned = nullptr;
    if (cudaMallocHost(reinterpret_cast<void **>(&pinned), chunk) != cudaSuccess) die("cudaMallocHost");
    uint8_t * dev = nullptr;
    if (cudaMalloc(reinterpret_cast<void **>(&dev), chunk) != cudaSuccess) die("cudaMalloc");
    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) die("stream");

    const size_t bytes_total = chunk * static_cast<size_t>(n_chunks);
    auto selected = [&](const char * name) {
        if (want.empty()) return true;
        for (const std::string & w : want) if (w == name) return true;
        return false;
    };

    std::vector<Case> results;

    auto run = [&](const char * name, auto && body) {
        if (!selected(name)) return;
        cudaStreamSynchronize(stream);
        const long f0 = page_faults_minor();
        const long m0 = page_faults_major();
        const double t0 = now_s();
        body();
        const double t1 = now_s();
        const long f1 = page_faults_minor();
        const long m1 = page_faults_major();
        cudaError_t sync = cudaStreamSynchronize(stream);
        const double t2 = now_s();
        if (sync != cudaSuccess) {
            std::fprintf(stderr, "case %s: sync failed: %s\n", name, cudaGetErrorString(sync));
        }
        Case c;
        c.name = name;
        c.gb_s = static_cast<double>(bytes_total) / (t1 - t0) / 1e9;
        c.synced_gb_s = static_cast<double>(bytes_total) / (t2 - t0) / 1e9;
        c.ms_per_chunk = (t1 - t0) * 1000.0 / static_cast<double>(n_chunks);
        c.minflt = f1 - f0;
        c.majflt = m1 - m0;
        results.push_back(c);
    };

    // -- pure host read paths (no GPU) ------------------------------------
    run("mmap", [&] {
        for (const Sample & s : samples) std::memcpy(host_buf, map + s.off, chunk);
    });
    run("mmap+willneed", [&] {
        for (const Sample & s : samples) {
            madvise(map + s.off, chunk, MADV_WILLNEED);
            std::memcpy(host_buf, map + s.off, chunk);
        }
    });
    run("mmap+populate", [&] {
        for (const Sample & s : samples) {
            madvise(map + s.off, chunk, MADV_POPULATE_READ);
            std::memcpy(host_buf, map + s.off, chunk);
        }
    });
    run("mmap+sequential", [&] {
        madvise(map, file_size, MADV_SEQUENTIAL);
        for (const Sample & s : samples) std::memcpy(host_buf, map + s.off, chunk);
        madvise(map, file_size, MADV_NORMAL);  // do not leak the policy into later cases
    });
    run("readahead", [&] {
        for (const Sample & s : samples) {
            readahead(fd, static_cast<off_t>(s.off), chunk);
            std::memcpy(host_buf, map + s.off, chunk);
        }
    });
    run("pread", [&] {
        for (const Sample & s : samples) {
            const ssize_t got = pread(fd, host_buf, chunk, static_cast<off_t>(s.off));
            if (got != static_cast<ssize_t>(chunk)) die("pread");
        }
    });
    run("pread+willneed", [&] {
        for (const Sample & s : samples) {
            posix_fadvise(fd, static_cast<off_t>(s.off), chunk, POSIX_FADV_WILLNEED);
            const ssize_t got = pread(fd, host_buf, chunk, static_cast<off_t>(s.off));
            if (got != static_cast<ssize_t>(chunk)) die("pread");
        }
    });
    run("pinpread", [&] {
        for (const Sample & s : samples) {
            const ssize_t got = pread(fd, pinned, chunk, static_cast<off_t>(s.off));
            if (got != static_cast<ssize_t>(chunk)) die("pread");
        }
    });
    run("pinpread+willneed", [&] {
        for (const Sample & s : samples) {
            posix_fadvise(fd, static_cast<off_t>(s.off), chunk, POSIX_FADV_WILLNEED);
            const ssize_t got = pread(fd, pinned, chunk, static_cast<off_t>(s.off));
            if (got != static_cast<ssize_t>(chunk)) die("pread");
        }
    });
    run("pinfill", [&] {
        std::memset(pinned, 0xa5, chunk);
        for (int i = 0; i < n_chunks; ++i) std::memcpy(pinned, host_buf, chunk);
    });
    run("odirect", [&] {
        const int ofd = ::open(path, O_RDONLY | O_DIRECT);
        if (ofd < 0) die("open O_DIRECT");
        for (const Sample & s : samples) {
            const ssize_t got = pread(ofd, host_raw, chunk, static_cast<off_t>(s.off));
            if (got != static_cast<ssize_t>(chunk)) die("pread O_DIRECT");
        }
        ::close(ofd);
    });

    // Several pread() streams in parallel: the engine reads 3 slices of ~0.6 MiB
    // per expert, so the single-thread copy may not be the ceiling.
    auto multi_pread = [&](int nthreads) {
        std::vector<uint8_t *> bufs(static_cast<size_t>(nthreads), nullptr);
        for (int i = 0; i < nthreads; ++i) {
            if (cudaMallocHost(reinterpret_cast<void **>(&bufs[static_cast<size_t>(i)]), chunk) != cudaSuccess) die("mt host");
        }
        std::vector<std::thread> workers;
        for (int ti = 0; ti < nthreads; ++ti) {
            workers.emplace_back([&, ti] {
                const size_t stride = static_cast<size_t>(nthreads);
                for (size_t i = static_cast<size_t>(ti); i < samples.size(); i += stride) {
                    const ssize_t got = pread(fd, bufs[static_cast<size_t>(ti)], chunk,
                                              static_cast<off_t>(samples[i].off));
                    if (got != static_cast<ssize_t>(chunk)) die("mt pread");
                }
            });
        }
        for (std::thread & w : workers) w.join();
        for (int i = 0; i < nthreads; ++i) cudaFreeHost(bufs[static_cast<size_t>(i)]);
    };
    run("pinpread_mt2", [&] { multi_pread(2); });
    run("pinpread_mt4", [&] { multi_pread(4); });
    run("pread_mt4", [&] {
        std::vector<uint8_t *> bufs(4, nullptr);
        for (int i = 0; i < 4; ++i) {
            bufs[static_cast<size_t>(i)] = static_cast<uint8_t *>(std::malloc(chunk));
        }
        std::vector<std::thread> workers;
        for (int ti = 0; ti < 4; ++ti) {
            workers.emplace_back([&, ti] {
                for (size_t i = static_cast<size_t>(ti); i < samples.size(); i += 4) {
                    const ssize_t got = pread(fd, bufs[static_cast<size_t>(ti)], chunk,
                                              static_cast<off_t>(samples[i].off));
                    if (got != static_cast<ssize_t>(chunk)) die("mt pread pageable");
                }
            });
        }
        for (std::thread & w : workers) w.join();
        for (int i = 0; i < 4; ++i) std::free(bufs[static_cast<size_t>(i)]);
    });

    // -- host -> device paths (what the engine measures) -------------------
    run("pageable+h2d", [&] {
        for (int i = 0; i < n_chunks; ++i) {
            if (cudaMemcpyAsync(dev, host_raw, chunk, cudaMemcpyHostToDevice, stream) != cudaSuccess) die("h2d");
        }
    });
    run("mmap+h2d", [&] {
        for (const Sample & s : samples) {
            if (cudaMemcpyAsync(dev, map + s.off, chunk, cudaMemcpyHostToDevice, stream) != cudaSuccess) die("h2d");
        }
    });
    run("mmap+populate+h2d", [&] {
        for (const Sample & s : samples) {
            madvise(map + s.off, chunk, MADV_POPULATE_READ);
            if (cudaMemcpyAsync(dev, map + s.off, chunk, cudaMemcpyHostToDevice, stream) != cudaSuccess) die("h2d");
        }
    });
    run("mmap+willneed+h2d", [&] {
        for (const Sample & s : samples) {
            madvise(map + s.off, chunk, MADV_WILLNEED);
            if (cudaMemcpyAsync(dev, map + s.off, chunk, cudaMemcpyHostToDevice, stream) != cudaSuccess) die("h2d");
        }
    });
    run("pread+h2d", [&] {
        for (const Sample & s : samples) {
            const ssize_t got = pread(fd, host_buf, chunk, static_cast<off_t>(s.off));
            if (got != static_cast<ssize_t>(chunk)) die("pread");
            if (cudaMemcpyAsync(dev, host_buf, chunk, cudaMemcpyHostToDevice, stream) != cudaSuccess) die("h2d");
        }
    });
    run("pinpread+h2d", [&] {
        for (const Sample & s : samples) {
            const ssize_t got = pread(fd, pinned, chunk, static_cast<off_t>(s.off));
            if (got != static_cast<ssize_t>(chunk)) die("pread");
            if (cudaMemcpyAsync(dev, pinned, chunk, cudaMemcpyHostToDevice, stream) != cudaSuccess) die("h2d");
        }
    });
    run("pin+h2d", [&] {
        for (int i = 0; i < n_chunks; ++i) {
            if (cudaMemcpyAsync(dev, pinned, chunk, cudaMemcpyHostToDevice, stream) != cudaSuccess) die("h2d");
        }
    });

    std::printf("{\"cases\":[");
    for (size_t i = 0; i < results.size(); ++i) {
        const Case & c = results[i];
        std::printf("%s{\"case\":\"%s\",\"gb_s\":%.3f,\"gb_s_synced\":%.3f,\"ms_per_chunk\":%.3f,"
                    "\"minflt_per_chunk\":%.1f,\"majflt_per_chunk\":%.2f}",
                    i ? "," : "", c.name.c_str(), c.gb_s, c.synced_gb_s, c.ms_per_chunk,
                    static_cast<double>(c.minflt) / n_chunks, static_cast<double>(c.majflt) / n_chunks);
    }
    std::printf("]}\n");

    cudaStreamDestroy(stream);
    cudaFree(dev);
    cudaFreeHost(pinned);
    std::free(host_buf);
    std::free(host_raw);
    munmap(map, file_size);
    ::close(fd);
    return 0;
}
