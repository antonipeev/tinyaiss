#pragma once

#include "../index_flat/index_flat.h"
#include "../index_ivf/index_ivf.h"
#include <cstdint>

namespace tinyaiss {

// file format constants
constexpr uint32_t MAGIC_NUMBER = 0x54464149; // "tfai"
constexpr uint32_t FORMAT_VERSION = 1;

// save indexflat to binary file
void save_index_flat(const IndexFlat& index, const char* path);

// load indexflat from binary file
IndexFlat* load_index_flat(const char* path);

// save indexivf to binary file
void save_index_ivf(const IndexIVF& index, const char* path);

// load indexivf from binary file
IndexIVF* load_index_ivf(const char* path);

} // namespace tinyaiss
