#include "serialization.h"
#include <fstream>
#include <cstring>

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

    if (header.magic != MAGIC_NUMBER || header.version != FORMAT_VERSION ||
        header.index_type != static_cast<uint32_t>(IndexType::FLAT)) {
        return nullptr;
    }

    MetricType metric = static_cast<MetricType>(header.metric_type);
    IndexFlat* index = new IndexFlat(header.dim, metric);

    // read vectors
    index->vectors.allocate(static_cast<uint32_t>(header.ntotal), header.dim);

    // verify dim_stride matches to prevent silent data corruption
    if (index->vectors.col_stride != header.dim_stride) {
        delete index;
        file.close();
        return nullptr;
    }

    file.read(reinterpret_cast<char*>(index->vectors.data), index->vectors.bytes());
    index->ntotal = static_cast<uint32_t>(header.ntotal);

    // read ids
    int64_t first_id;
    file.read(reinterpret_cast<char*>(&first_id), sizeof(int64_t));

    if (first_id == -1) {
        // sequential ids
        index->ids = nullptr;
    } else {
        // external ids
        index->ids = new int64_t[index->ntotal];
        index->ids[0] = first_id;
        file.read(reinterpret_cast<char*>(index->ids + 1),
                 (index->ntotal - 1) * sizeof(int64_t));
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

    if (header.magic != MAGIC_NUMBER || header.version != FORMAT_VERSION ||
        header.index_type != static_cast<uint32_t>(IndexType::IVF)) {
        return nullptr;
    }

    MetricType metric = static_cast<MetricType>(header.metric_type);
    IndexIVF* index = new IndexIVF(header.dim, header.nlist, metric);

    // read centroids
    index->centroids.allocate(header.nlist, header.dim);

    // verify dim_stride matches for centroids
    if (index->centroids.col_stride != header.dim_stride) {
        delete index;
        file.close();
        return nullptr;
    }

    file.read(reinterpret_cast<char*>(index->centroids.data),
             index->centroids.bytes());
    index->is_trained = true;
    index->nprobe = header.nprobe;
    index->ntotal = static_cast<uint32_t>(header.ntotal);

    // read list metadata
    index->lists.allocate(header.nlist, header.dim, static_cast<uint32_t>(header.ntotal));

    // verify dim_stride matches for inverted lists
    if (index->lists.dim_stride != header.dim_stride) {
        delete index;
        file.close();
        return nullptr;
    }
    file.read(reinterpret_cast<char*>(index->lists.list_offsets),
             (header.nlist + 1) * sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(index->lists.list_sizes),
             header.nlist * sizeof(uint32_t));

    // read vectors and ids
    if (header.ntotal > 0) {
        size_t vector_bytes = static_cast<size_t>(header.ntotal) *
                             index->lists.dim_stride * sizeof(float);
        file.read(reinterpret_cast<char*>(index->lists.vectors), vector_bytes);
        file.read(reinterpret_cast<char*>(index->lists.ids),
                 header.ntotal * sizeof(int64_t));
    }

    file.close();
    return index;
}

} // namespace tinyaiss
