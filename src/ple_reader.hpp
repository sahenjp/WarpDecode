#pragma once

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unistd.h>
#include <utility>
#include <vector>

namespace wd {

// File-backed sparse row reader for very large PLE/n-gram tables.
// It avoids demand-faulting a full VM page for each tiny row: requested rows are
// deduplicated, sorted, adjacent rows are coalesced into one pread(), then copied
// back to the caller's original order. The caller may layer async prefetch/cache
// on top without changing the table format.
class PleDirectReader {
public:
    PleDirectReader(const std::string& path, uint64_t base_offset, uint32_t row_bytes, uint64_t row_count)
        : base_offset_(base_offset), row_bytes_(row_bytes), row_count_(row_count) {
        if (row_bytes_ == 0 || row_count_ == 0) throw std::runtime_error("PLE row geometry is invalid");
        fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) throw std::runtime_error("PLE open failed: " + path + ": " + std::strerror(errno));
    }

    PleDirectReader(const PleDirectReader&) = delete;
    PleDirectReader& operator=(const PleDirectReader&) = delete;
    ~PleDirectReader() { if (fd_ >= 0) ::close(fd_); }

    uint32_t row_bytes() const { return row_bytes_; }

    // dst is indices.size() * row_bytes().
    void read_rows(const std::vector<uint64_t>& indices, void* dst) const {
        if (indices.empty()) return;
        auto* out = static_cast<uint8_t*>(dst);

        std::vector<uint64_t> unique = indices;
        for (uint64_t r : unique) if (r >= row_count_) throw std::runtime_error("PLE row out of range");
        std::sort(unique.begin(), unique.end());
        unique.erase(std::unique(unique.begin(), unique.end()), unique.end());

        std::unordered_map<uint64_t, std::vector<uint8_t>> rows;
        rows.reserve(unique.size() * 2);

        size_t i = 0;
        while (i < unique.size()) {
            size_t j = i + 1;
            while (j < unique.size() && unique[j] == unique[j - 1] + 1) ++j;

            const uint64_t first = unique[i];
            const size_t nrows = j - i;
            const size_t bytes = nrows * static_cast<size_t>(row_bytes_);
            std::vector<uint8_t> run(bytes);
            pread_exact(run.data(), bytes, base_offset_ + first * static_cast<uint64_t>(row_bytes_));

            for (size_t k = 0; k < nrows; ++k) {
                rows.emplace(first + k, std::vector<uint8_t>(
                    run.begin() + static_cast<ptrdiff_t>(k * row_bytes_),
                    run.begin() + static_cast<ptrdiff_t>((k + 1) * row_bytes_)));
            }
            i = j;
        }

        for (size_t k = 0; k < indices.size(); ++k) {
            const auto it = rows.find(indices[k]);
            if (it == rows.end()) throw std::runtime_error("internal PLE row lookup failure");
            std::memcpy(out + k * row_bytes_, it->second.data(), row_bytes_);
        }
    }

private:
    void pread_exact(void* dst, size_t n, uint64_t off) const {
        auto* p = static_cast<uint8_t*>(dst);
        size_t done = 0;
        while (done < n) {
            const ssize_t got = ::pread(fd_, p + done, n - done, static_cast<off_t>(off + done));
            if (got < 0 && errno == EINTR) continue;
            if (got <= 0) throw std::runtime_error("PLE pread failed");
            done += static_cast<size_t>(got);
        }
    }

    int fd_ = -1;
    uint64_t base_offset_ = 0;
    uint32_t row_bytes_ = 0;
    uint64_t row_count_ = 0;
};

} // namespace wd
