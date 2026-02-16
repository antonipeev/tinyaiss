#!/bin/bash

echo "=== tinyFAISS Structure Verification ==="
echo ""

ERRORS=0

# List of required files
REQUIRED_FILES=(
    "core/types.h"
    "core/memory.h"
    "core/aligned_matrix.h"
    "core/aligned_matrix.cpp"
    "math/platform.h"
    "math/distance.h"
    "math/distance_l2.cpp"
    "math/distance_ip.cpp"
    "search/topk.h"
    "search/topk.cpp"
    "clustering/kmeans.h"
    "clustering/kmeans.cpp"
    "index_flat/index_flat.h"
    "index_flat/index_flat.cpp"
    "index_ivf/ivf_lists.h"
    "index_ivf/ivf_lists.cpp"
    "index_ivf/index_ivf.h"
    "index_ivf/index_ivf.cpp"
    "io/serialization.h"
    "io/serialization.cpp"
    "api/tinyFAISS.h"
    "api/tinyFAISS.cpp"
    "tests/example_usage.cpp"
    "CMakeLists.txt"
    "README.md"
)

echo "Checking for required files..."
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file MISSING"
        ERRORS=$((ERRORS + 1))
    fi
done

echo ""
echo "Total files: ${#REQUIRED_FILES[@]}"
echo "Missing files: $ERRORS"

if [ $ERRORS -eq 0 ]; then
    echo ""
    echo "=== All files present! ==="
    echo ""
    echo "To build:"
    echo "  mkdir build && cd build"
    echo "  cmake .."
    echo "  cmake --build ."
    exit 0
else
    echo ""
    echo "=== Structure incomplete ==="
    exit 1
fi
