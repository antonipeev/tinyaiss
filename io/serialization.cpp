#include "serialization.h"
#include <fstream>
#include <cstring>
#include <limits>

namespace tinyaiss {

// header structure (64 bytes)
struct FileHeader {
    uint32_t magic;           // 0x00: magic number
    uint32_t version;         // 0x04: format version
    uint32_t index_type;      // 0x08: 0=flat, 1=ivf
    uint32_t metric_type;     // 0x0c: 0=l2, 1=cosine
    uint32_t dim;             // 0x10: original dimension
    uint32_t dim_stride;      // 0x14: padded dimension
    uint64_t ntotal;          // 0x18: total vectors
    uint32_t nlist;           // 0x20: number of lists (ivf only)
    uint32_t nprobe;          // 0x24: number of probes (ivf only)
    uint8_t reserved[24];     // 0x28: reserved for future use
};

static_assert(sizeof(FileHeader) == 64, "FileHeader must be 64 bytes");

// safety limits for deserialization to reject crafted/corrupted files
static constexpr uint32_t MAX_DIMENSION       = 65536;   // 64K dimensions
static constexpr uint64_t MAX_TOTAL_VECTORS   = 100000000; // 100M vectors
static constexpr uint32_t MAX_NLIST           = 1000000;  // 1M clusters
static constexpr uint32_t MAX_NPROBE          = 65536;

// validate header fields are within sane bounds
static bool validate_header_bounds(const FileHeader& header) {
    if (header.dim == 0 || header.dim > MAX_DIMENSION) return false;
    if (header.ntotal > MAX_TOTAL_VECTORS) return false;
    if (header.dim_stride == 0 || header.dim_stride > MAX_DIMENSION + 16) return false;
    if (header.metric_type > 1) return false;

    if (header.index_type == static_cast<uint32_t>(IndexType::IVF)) {
        if (header.nlist == 0 || header.nlist > MAX_NLIST) return false;
        if (header.nprobe == 0 || header.nprobe > MAX_NPROBE) return false;
    }

    // verify ntotal fits in uint32_t (internal storage type)
    if (header.ntotal > std::numeric_limits<uint32_t>::max()) return false;

    return true;
}

// validate file has enough remaining bytes for a read of expected_bytes
static bool validate_file_remaining(std::ifstream& file, size_t expected_bytes) {
    auto current_pos = file.tellg();
    if (current_pos == -1) return false;

    file.seekg(0, std::ios::end);
    auto end_pos = file.tellg();
    if (end_pos == -1) return false;

    file.seekg(current_pos);
    if (!file.good()) return false;

    return static_cast<size_t>(end_pos - current_pos) >= expected_bytes;
}

void save_index_flat(const IndexFlat& index, const char* path) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        return; // todo: error handling
    }

    // write header
    FileHeader header{};
    header.magic = MAGIC_NUMBER;
    header.version = FORMAT_VERSION;
    header.index_type = static_cast<uint32_t>(IndexType::FLAT);
    header.metric_type = static_cast<uint32_t>(index.metric);
    header.dim = index.dim;
    header.dim_stride = index.vectors.col_stride;
    header.ntotal = index.ntotal;
    header.nlist = 0;
    header.nprobe = 0;

    file.write(reinterpret_cast<const char*>(&header), sizeof(header));

    // write vectors
    size_t vector_bytes = index.vectors.bytes();
    file.write(reinterpret_cast<const char*>(index.vectors.data), vector_bytes);

    // write ids
    if (index.ids) {
        file.write(reinterpret_cast<const char*>(index.ids),
                  index.ntotal * sizeof(int64_t));
    } else {
        // write sentinel value to indicate sequential ids
        int64_t sentinel = -1;
        file.write(reinterpret_cast<const char*>(&sentinel), sizeof(int64_t));
    }

    file.close();
}

IndexFlat* load_index_flat(const char* path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return nullptr;
    }

    // read header
    FileHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!file.good()) return nullptr;

    if (header.magic != MAGIC_NUMBER || header.version != FORMAT_VERSION ||
        header.index_type != static_cast<uint32_t>(IndexType::FLAT)) {
        return nullptr;
    }

    // SEC-1: validate header fields are within sane bounds
    if (!validate_header_bounds(header)) {
        return nullptr;
    }

    MetricType metric = static_cast<MetricType>(header.metric_type);
    IndexFlat* index = new IndexFlat(header.dim, metric);

    // read vectors (allocate returns false on overflow or OOM)
    if (!index->vectors.allocate(static_cast<uint32_t>(header.ntotal), header.dim)) {
        delete index;
        return nullptr;
    }

    // verify dim_stride matches to prevent silent data corruption
    if (index->vectors.col_stride != header.dim_stride) {
        delete index;
        file.close();
        return nullptr;
    }

    // SEC-3: verify file has enough data before reading
    size_t vector_bytes = index->vectors.bytes();
    if (!validate_file_remaining(file, vector_bytes)) {
        delete index;
        return nullptr;
    }

    file.read(reinterpret_cast<char*>(index->vectors.data), vector_bytes);
    if (!file.good()) {
        delete index;
        return nullptr;
    }
    index->ntotal = static_cast<uint32_t>(header.ntotal);

    // read ids
    if (!validate_file_remaining(file, sizeof(int64_t))) {
        delete index;
        return nullptr;
    }

    int64_t first_id;
    file.read(reinterpret_cast<char*>(&first_id), sizeof(int64_t));
    if (!file.good()) {
        delete index;
        return nullptr;
    }

    if (first_id == -1) {
        // sequential ids
        index->ids = nullptr;
    } else {
        // external ids
        if (index->ntotal == 0) {
            delete index;
            return nullptr;
        }
        size_t remaining_id_bytes = static_cast<size_t>(index->ntotal - 1) * sizeof(int64_t);
        if (!validate_file_remaining(file, remaining_id_bytes)) {
            delete index;
            return nullptr;
        }
        index->ids = new int64_t[index->ntotal];
        index->ids[0] = first_id;
        if (index->ntotal > 1) {
            file.read(reinterpret_cast<char*>(index->ids + 1), remaining_id_bytes);
            if (!file.good()) {
                delete index;
                return nullptr;
            }
        }
    }

    file.close();
    return index;
}

void save_index_ivf(const IndexIVF& index, const char* path) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        return;
    }

    // write header
    FileHeader header{};
    header.magic = MAGIC_NUMBER;
    header.version = FORMAT_VERSION;
    header.index_type = static_cast<uint32_t>(IndexType::IVF);
    header.metric_type = static_cast<uint32_t>(index.metric);
    header.dim = index.dim;
    header.dim_stride = index.centroids.col_stride;
    header.ntotal = index.ntotal;
    header.nlist = index.nlist;
    header.nprobe = index.nprobe;

    file.write(reinterpret_cast<const char*>(&header), sizeof(header));

    // write centroids
    file.write(reinterpret_cast<const char*>(index.centroids.data),
              index.centroids.bytes());

    // write list metadata
    file.write(reinterpret_cast<const char*>(index.lists.list_offsets),
              (index.nlist + 1) * sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(index.lists.list_sizes),
              index.nlist * sizeof(uint32_t));

    // write vectors
    if (index.ntotal > 0) {
        size_t vector_bytes = static_cast<size_t>(index.ntotal) *
                             index.lists.dim_stride * sizeof(float);
        file.write(reinterpret_cast<const char*>(index.lists.vectors),
                  vector_bytes);

        // write ids
        file.write(reinterpret_cast<const char*>(index.lists.ids),
                  index.ntotal * sizeof(int64_t));
    }

    file.close();
}

IndexIVF* load_index_ivf(const char* path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return nullptr;
    }

    // read header
    FileHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!file.good()) return nullptr;

    if (header.magic != MAGIC_NUMBER || header.version != FORMAT_VERSION ||
        header.index_type != static_cast<uint32_t>(IndexType::IVF)) {
        return nullptr;
    }

    // SEC-1: validate header fields are within sane bounds
    if (!validate_header_bounds(header)) {
        return nullptr;
    }

    MetricType metric = static_cast<MetricType>(header.metric_type);
    IndexIVF* index = new IndexIVF(header.dim, header.nlist, metric);

    // read centroids (allocate returns false on overflow or OOM)
    if (!index->centroids.allocate(header.nlist, header.dim)) {
        delete index;
        return nullptr;
    }

    // verify dim_stride matches for centroids
    if (index->centroids.col_stride != header.dim_stride) {
        delete index;
        file.close();
        return nullptr;
    }

    size_t centroid_bytes = index->centroids.bytes();
    if (!validate_file_remaining(file, centroid_bytes)) {
        delete index;
        return nullptr;
    }
    file.read(reinterpret_cast<char*>(index->centroids.data), centroid_bytes);
    if (!file.good()) {
        delete index;
        return nullptr;
    }
    index->is_trained = true;
    index->nprobe = header.nprobe;
    index->ntotal = static_cast<uint32_t>(header.ntotal);

    // read list metadata (allocate returns false on overflow or OOM)
    if (!index->lists.allocate(header.nlist, header.dim, static_cast<uint32_t>(header.ntotal))) {
        delete index;
        return nullptr;
    }

    // verify dim_stride matches for inverted lists
    if (index->lists.dim_stride != header.dim_stride) {
        delete index;
        file.close();
        return nullptr;
    }

    size_t offsets_bytes = static_cast<size_t>(header.nlist + 1) * sizeof(uint32_t);
    size_t sizes_bytes = static_cast<size_t>(header.nlist) * sizeof(uint32_t);
    if (!validate_file_remaining(file, offsets_bytes + sizes_bytes)) {
        delete index;
        return nullptr;
    }

    file.read(reinterpret_cast<char*>(index->lists.list_offsets), offsets_bytes);
    if (!file.good()) {
        delete index;
        return nullptr;
    }
    file.read(reinterpret_cast<char*>(index->lists.list_sizes), sizes_bytes);
    if (!file.good()) {
        delete index;
        return nullptr;
    }

    // read vectors and ids
    if (header.ntotal > 0) {
        // overflow-safe computation for vector_bytes
        size_t dim_stride = index->lists.dim_stride;
        if (header.ntotal > std::numeric_limits<size_t>::max() / dim_stride ||
            header.ntotal * dim_stride > std::numeric_limits<size_t>::max() / sizeof(float)) {
            delete index;
            return nullptr;
        }
        size_t vector_bytes = static_cast<size_t>(header.ntotal) * dim_stride * sizeof(float);
        size_t id_bytes = static_cast<size_t>(header.ntotal) * sizeof(int64_t);

        if (!validate_file_remaining(file, vector_bytes + id_bytes)) {
            delete index;
            return nullptr;
        }

        file.read(reinterpret_cast<char*>(index->lists.vectors), vector_bytes);
        if (!file.good()) {
            delete index;
            return nullptr;
        }
        file.read(reinterpret_cast<char*>(index->lists.ids), id_bytes);
        if (!file.good()) {
            delete index;
            return nullptr;
        }
    }

    file.close();
    return index;
}

} // namespace tinyaiss
