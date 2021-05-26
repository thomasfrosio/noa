#include <cuda_runtime_api.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/memory/PtrArray.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::Noa;

TEMPLATE_TEST_CASE("PtrArray 1D: base", "[noa][cuda]", int32_t, uint32_t, float, cfloat_t) {
    Test::IntRandomizer<size_t> randomizer_small(1, 64);
    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape = Test::getRandomShape(ndim);
    size_t elements = getElements(shape);

    // test allocation and free
    AND_THEN("allocation, free, ownership") {
        CUDA::Memory::PtrArray<TestType> ptr1;
        REQUIRE_FALSE(ptr1);
        {
            CUDA::Memory::PtrArray<TestType> ptr2(shape);
            REQUIRE(ptr2);
            REQUIRE(ptr2.get());
            REQUIRE_FALSE(ptr2.empty());
            REQUIRE(ptr2.elements() == elements);
            REQUIRE(all(ptr2.shape() == shape));
            ptr1.reset(ptr2.release(), shape); // transfer ownership.
            REQUIRE_FALSE(ptr2);
            REQUIRE_FALSE(ptr2.get());
            REQUIRE(ptr2.empty());
            REQUIRE_FALSE(ptr2.elements());
            REQUIRE(all(ptr2.shape() == size_t{0}));
        }
        REQUIRE(ptr1);
        REQUIRE(ptr1.get());
        REQUIRE_FALSE(ptr1.empty());
        REQUIRE(ptr1.elements() == elements);
    }

    AND_THEN("empty states") {
        CUDA::Memory::PtrArray<TestType> ptr1(shape);
        ptr1.reset(shape);
        ptr1.dispose();
        ptr1.dispose(); // no double delete.
        REQUIRE(ptr1.empty());
        REQUIRE_FALSE(ptr1);
        ptr1.reset();
    }
}
