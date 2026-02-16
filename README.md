# tinyaiss

 A personal learning project inspired by [FAISS](https://github.com/facebookresearch/faiss).

## What's Included

- IndexFlat (brute-force exact search)
- IndexIVF (inverted file index with k-means clustering)
- L2 and cosine distance metrics
- Basic save/load functionality
- small codebase, zero dependencies

## Building

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

## Usage

```cpp
#include "api/tinyAISS.h"

using namespace tinyaiss;

// Create index
auto* index = create_index(IndexType::FLAT, dim, MetricType::L2);

// Add vectors
add(index, data, nullptr, n);

// Search
SearchResult result = search(index, query, 1, k);

// Cleanup
free_result(result);
destroy_index(index);
```

See `main.cpp` and `tests/example_usage.cpp` for complete examples.

## API

See `api/tinyAISS.h` for the complete API. Key functions:

- `create_index()` - Create IndexFlat or IndexIVF
- `add()` - Add vectors with optional custom IDs
- `search()` - k-NN search
- `save()`/`load()` - Persistence
- `train()` - Train IVF index (required before adding vectors)

## Limitations

- Single-threaded, CPU-only
- In-memory only (no mmap)
- Float32 vectors only
- Basic implementations (no SIMD, no advanced optimizations)

## License

MIT License - see [LICENSE](LICENSE).
