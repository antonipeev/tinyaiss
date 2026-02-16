#include "../api/tinyAISS.h"
#include <iostream>
#include <vector>
#include <random>

using namespace tinyaiss;

// sample testing data
int main() {
    std::cout << "=== tinyAISS API Demo ===" << std::endl;

    // generate random test data
    const uint32_t dim = 128;
    const uint64_t n = 1000;
    std::vector<float> data(n * dim);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (uint64_t i = 0; i < n * dim; i++) {
        data[i] = dist(rng);
    }

    // ===== FLAT Index =====
    std::cout << "\n1. FLAT Index (exact search)" << std::endl;
    auto* flat_index = create_index(IndexType::FLAT, dim, MetricType::L2);
    add(flat_index, data.data(), nullptr, n);
    std::cout << "   Added " << ntotal(flat_index) << " vectors" << std::endl;

    SearchResult result = search(flat_index, data.data(), 1, 5);
    std::cout << "   Top 5 neighbors: ";
    for (int i = 0; i < 5; i++) std::cout << result.ids[i] << " ";
    std::cout << std::endl;
    free_result(result);

    // Save/Load
    save(flat_index, "test.bin");
    auto* loaded = load("test.bin");
    std::cout << "   Loaded index has " << ntotal(loaded) << " vectors" << std::endl;
    destroy_index(loaded);
    destroy_index(flat_index);

    // ===== IVF Index =====
    std::cout << "\n2. IVF Index (fast approximate search)" << std::endl;
    IVFConfig config;
    config.nlist = 64;
    config.nprobe = 8;
    auto* ivf_index = create_index(IndexType::IVF, dim, MetricType::L2, config);

    train(ivf_index, data.data(), n);
    add(ivf_index, data.data(), nullptr, n);
    std::cout << "   Trained and added " << ntotal(ivf_index) << " vectors" << std::endl;

    result = search(ivf_index, data.data(), 1, 5);
    std::cout << "   Top 5 neighbors: ";
    for (int i = 0; i < 5; i++) std::cout << result.ids[i] << " ";
    std::cout << std::endl;
    free_result(result);

    // adjust nprobe
    set_nprobe(ivf_index, 16);
    std::cout << "   nprobe set to " << get_nprobe(ivf_index) << std::endl;
    destroy_index(ivf_index);

    // ===== Custom IDs =====
    std::cout << "\n3. Custom IDs" << std::endl;
    auto* custom_index = create_index(IndexType::FLAT, dim, MetricType::L2);
    std::vector<int64_t> ids = {1000, 2000, 3000, 4000, 5000};
    add(custom_index, data.data(), ids.data(), 5);

    result = search(custom_index, data.data(), 1, 3);
    std::cout << "   Custom IDs returned: " << result.ids[0] << ", "
              << result.ids[1] << ", " << result.ids[2] << std::endl;
    free_result(result);
    destroy_index(custom_index);

    // ===== COSINE metric =====
    std::cout << "\n4. COSINE metric" << std::endl;
    auto* cosine_index = create_index(IndexType::FLAT, dim, MetricType::COSINE);
    add(cosine_index, data.data(), nullptr, 100);

    result = search(cosine_index, data.data(), 1, 3);
    std::cout << "   Distances: " << result.distances[0] << ", "
              << result.distances[1] << ", " << result.distances[2] << std::endl;
    free_result(result);
    destroy_index(cosine_index);

    // ===== Index info =====
    std::cout << "\n5. Index info functions" << std::endl;
    auto* info_index = create_index(IndexType::FLAT, 64, MetricType::L2);
    std::cout << "   dimension: " << dimension(info_index) << std::endl;
    std::cout << "   metric: " << (metric(info_index) == MetricType::L2 ? "L2" : "COSINE") << std::endl;
    std::cout << "   type: " << (index_type(info_index) == IndexType::FLAT ? "FLAT" : "IVF") << std::endl;
    destroy_index(info_index);

    std::cout << "\n=== Done ===" << std::endl;
    return 0;
}
