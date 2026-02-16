#pragma once

#include <cstdint>

namespace tinyaiss {

// fixed-size max-heap for top-k nearest neighbor search
// maintains k smallest distances (max-heap so root is the k-th smallest)
struct TopKHeap {
    float* distances;     // max-heap of size k
    int64_t* ids;         // parallel ids
    uint32_t capacity;    // = k
    uint32_t size;        // current fill level

    TopKHeap(uint32_t k);
    ~TopKHeap();

    // non-copyable
    TopKHeap(const TopKHeap&) = delete;
    TopKHeap& operator=(const TopKHeap&) = delete;

    // reset heap for new query
    void reset();

    // insert if smaller than current worst (root)
    void push_if_smaller(float dist, int64_t id);

    // extract results in sorted order (ascending distance)
    // out_distances and out_ids must have capacity k
    void dump_sorted(float* out_distances, int64_t* out_ids);

private:
    void heapify();
    void sift_down(uint32_t idx);
};

} // namespace tinyaiss
