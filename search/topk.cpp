#include "topk.h"
#include <algorithm>
#include <cstring>
#include <limits>

namespace tinyaiss {

TopKHeap::TopKHeap(uint32_t k)
    : capacity(k), size(0) {
    distances = new float[k];
    ids = new int64_t[k];
}

TopKHeap::~TopKHeap() {
    delete[] distances;
    delete[] ids;
}

void TopKHeap::reset() {
    size = 0;
}

void TopKHeap::sift_down(uint32_t idx) {
    while (true) {
        uint32_t largest = idx;
        uint32_t left = 2 * idx + 1;
        uint32_t right = 2 * idx + 2;

        if (left < size && distances[left] > distances[largest]) {
            largest = left;
        }
        if (right < size && distances[right] > distances[largest]) {
            largest = right;
        }

        if (largest == idx) {
            break;
        }

        // swap
        std::swap(distances[idx], distances[largest]);
        std::swap(ids[idx], ids[largest]);
        idx = largest;
    }
}

void TopKHeap::heapify() {
    // build max-heap from bottom up
    if (size <= 1) return;
    for (int32_t i = (size / 2) - 1; i >= 0; i--) {
        sift_down(static_cast<uint32_t>(i));
    }
}

void TopKHeap::push_if_smaller(float dist, int64_t id) {
    if (size < capacity) {
        // heap not full: insert unconditionally
        distances[size] = dist;
        ids[size] = id;
        size++;
        if (size == capacity) {
            heapify();  // build heap once full
        }
    } else if (dist < distances[0]) {
        // replace root (current worst in top-k)
        distances[0] = dist;
        ids[0] = id;
        sift_down(0);
    }
    // else: dist >= worst in heap, skip
}

void TopKHeap::dump_sorted(float* out_distances, int64_t* out_ids) {
    // extract elements in descending order (repeatedly pop root)
    // then reverse to get ascending order
    uint32_t original_size = size;

    for (uint32_t i = 0; i < original_size; i++) {
        uint32_t extract_idx = original_size - 1 - i;
        out_distances[extract_idx] = distances[0];
        out_ids[extract_idx] = ids[0];

        // move last element to root and sift down
        if (size > 1) {
            distances[0] = distances[size - 1];
            ids[0] = ids[size - 1];
            size--;
            sift_down(0);
        } else {
            size--;
        }
    }

    // fill remaining positions with sentinel values if size < capacity
    for (uint32_t i = original_size; i < capacity; i++) {
        out_distances[i] = std::numeric_limits<float>::max();
        out_ids[i] = -1;
    }

    // reset for next query
    size = 0;
}

} // namespace tinyaiss
