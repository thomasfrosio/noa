#include <cuda_runtime_api.h>

#include "noa/gpu/cuda/PtrDevice.h"

#include "../../../Helpers.h"
#include <catch2/catch.hpp>

using namespace ::Noa;

#ifdef NOA_DEBUG
#define REQUIRE_ALLOC_COUNT(count) REQUIRE(Noa::GPU::Allocator::debug_count_device == count)
#else
#define REQUIRE_ALLOC_COUNT(count)
#endif

// Just test the memory is allocated and cudaMemset doesn't return any error.
inline cudaError_t zero(void* d_ptr, size_t bytes) {
    cudaError_t err = cudaMemset(d_ptr, 0, bytes);
    if (err) {
        cudaDeviceSynchronize(); // cudaMemset is async if device memory.
        return err;
    }
    return cudaDeviceSynchronize();
}

// These are very simple tests. PtrDevice will be tested extensively since
// it is a dependency for many parts of the CUDA backend.
TEMPLATE_TEST_CASE("PtrDevice", "[noa][cuda][not_thread_safe]", int, uint, float, cfloat_t) {
    using PtrDevice = Noa::GPU::PtrDevice<TestType>;
    auto elements = static_cast<size_t>(Test::pseudoRandom(16, 255));
    REQUIRE_ALLOC_COUNT(0); // Assuming it is the only test running, hence [not_thread_safe] tag.

    AND_THEN("Alloc, dealloc and ownership") {
        {
            PtrDevice linear_memory(elements);
            REQUIRE_ALLOC_COUNT(1);
            REQUIRE(linear_memory);
            REQUIRE(linear_memory.get());
            REQUIRE(!linear_memory.empty());
            REQUIRE(linear_memory.get());
            REQUIRE(linear_memory.elements() == elements);
            REQUIRE(linear_memory.bytes() == elements * sizeof(TestType));
        }
        REQUIRE_ALLOC_COUNT(0);
        {
            PtrDevice linear_memory1(elements); // owner
            REQUIRE_ALLOC_COUNT(1);
            {
                PtrDevice linear_memory2(elements, linear_memory1.get(), false);
                REQUIRE(!linear_memory2.empty());
                REQUIRE(linear_memory2.get());
                REQUIRE(linear_memory2.elements() == elements);
                REQUIRE_ALLOC_COUNT(1);
            }
            {
                PtrDevice linear_memory2;
                linear_memory2.reset(elements, linear_memory1.get(), true);
                REQUIRE_ALLOC_COUNT(1);
                linear_memory2.is_owner = false;
                linear_memory2.dispose();
                REQUIRE_ALLOC_COUNT(1);
                REQUIRE(!linear_memory2.empty());
                REQUIRE(!linear_memory2.get());
                REQUIRE(!linear_memory2.elements());
            }
            REQUIRE_ALLOC_COUNT(1);
            REQUIRE(linear_memory1);
            REQUIRE(linear_memory1.get());
            REQUIRE(!linear_memory1.empty());
            REQUIRE(linear_memory1.get());
            REQUIRE(linear_memory1.elements() == elements);
            REQUIRE(linear_memory1.bytes() == elements * sizeof(TestType));
            REQUIRE(linear_memory1.is_owner);
        }
        REQUIRE_ALLOC_COUNT(0);

        {
            PtrDevice linear_memory1(elements); // transfer ownership to someone else.
            REQUIRE_ALLOC_COUNT(1);
            {
                PtrDevice linear_memory2(linear_memory1.elements(), linear_memory1.get(), true);
            }
            REQUIRE_ALLOC_COUNT(0);
            linear_memory1.is_owner = false;
        }
    }

    AND_THEN("Send data to device") {
        PtrDevice linear_memory(elements);
        REQUIRE_ALLOC_COUNT(1);
        cudaError_t err = zero(linear_memory.get(), linear_memory.bytes());
        REQUIRE(err == cudaSuccess);
    }
    REQUIRE_ALLOC_COUNT(0);
}
