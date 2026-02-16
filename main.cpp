// tinyAISS.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <random>
#include <vector>
#include "api/tinyAISS.h"

using namespace tinyaiss;

// generate random test data
void generate_random_data(float* data, uint64_t n, uint32_t dim, uint64_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (uint64_t i = 0; i < n * dim; i++) {
        data[i] = dist(rng);
    }
}

int main() {
    std::cout << "=== tinyAISS Example Usage ===" << std::endl;

    const uint32_t dim = 128;
    const uint64_t n = 10000;
    const uint64_t nq = 10;
    const uint32_t k = 5;

    // generate random data
    std::cout << "Generating " << n << " vectors of dimension " << dim << "..." << std::endl;
    std::vector<float> data(n * dim);
    std::vector<float> queries(nq * dim);
    generate_random_data(data.data(), n, dim, 1234);
    generate_random_data(queries.data(), nq, dim, 5678);

    // test indexflat
    std::cout << "\n--- Testing IndexFlat ---" << std::endl;
    {
        auto* index = create_index(IndexType::FLAT, dim, MetricType::L2);

        std::cout << "Adding " << n << " vectors..." << std::endl;
        add(index, data.data(), nullptr, n);

        std::cout << "Index contains " << ntotal(index) << " vectors" << std::endl;

        std::cout << "Searching for " << k << " nearest neighbors for " << nq << " queries..." << std::endl;
        SearchResult result = search(index, queries.data(), nq, k);

        std::cout << "Results for first 3 queries:" << std::endl;
        for (uint64_t i = 0; i < std::min(3ULL, nq); i++) {
            std::cout << "Query " << i << ": ";
            for (uint32_t j = 0; j < k; j++) {
                std::cout << "[id=" << result.ids[i * k + j]
                         << " dist=" << result.distances[i * k + j] << "] ";
            }
            std::cout << std::endl;
        }

        free_result(result);

        // test save/load
        std::cout << "Saving index to flat_index.bin..." << std::endl;
        save(index, "flat_index.bin");

        std::cout << "Loading index from flat_index.bin..." << std::endl;
        auto* loaded_index = load("flat_index.bin");
        std::cout << "Loaded index contains " << ntotal(loaded_index) << " vectors" << std::endl;

        destroy_index(loaded_index);
        destroy_index(index);
    }

    // test indexivf
    std::cout << "\n--- Testing IndexIVF ---" << std::endl;
    {
        IVFConfig config;
        config.nlist = 128;
        config.nprobe = 8;

        auto* index = create_index(IndexType::IVF, dim, MetricType::L2, config);

        std::cout << "Training index on " << n << " vectors..." << std::endl;
        train(index, data.data(), n);

        std::cout << "Adding " << n << " vectors..." << std::endl;
        add(index, data.data(), nullptr, n);

        std::cout << "Index contains " << ntotal(index) << " vectors" << std::endl;
        std::cout << "nprobe = " << get_nprobe(index) << std::endl;

        std::cout << "Searching for " << k << " nearest neighbors for " << nq << " queries..." << std::endl;
        SearchResult result = search(index, queries.data(), nq, k);

        std::cout << "Results for first 3 queries:" << std::endl;
        for (uint64_t i = 0; i < std::min(3ULL, nq); i++) {
            std::cout << "Query " << i << ": ";
            for (uint32_t j = 0; j < k; j++) {
                std::cout << "[id=" << result.ids[i * k + j]
                         << " dist=" << result.distances[i * k + j] << "] ";
            }
            std::cout << std::endl;
        }

        free_result(result);

        // test save/load
        std::cout << "Saving index to ivf_index.bin..." << std::endl;
        save(index, "ivf_index.bin");

        std::cout << "Loading index from ivf_index.bin..." << std::endl;
        auto* loaded_index = load("ivf_index.bin");
        std::cout << "Loaded index contains " << ntotal(loaded_index) << " vectors" << std::endl;
        std::cout << "Loaded nprobe = " << get_nprobe(loaded_index) << std::endl;

        destroy_index(loaded_index);
        destroy_index(index);
    }

    std::cout << "\n=== All tests completed successfully ===" << std::endl;
    return 0;
}
