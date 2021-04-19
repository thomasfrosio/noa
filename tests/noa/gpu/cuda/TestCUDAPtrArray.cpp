#include <cuda_runtime_api.h>

#include <noa/cpu/PtrHost.h>
#include <noa/gpu/cuda/PtrArray.h>

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
        Noa::CUDA::PtrArray<TestType> ptr1;
        REQUIRE_FALSE(ptr1);
        {
            Noa::CUDA::PtrArray<TestType> ptr2(shape);
            REQUIRE(ptr2);
            REQUIRE(ptr2.get());
            REQUIRE_FALSE(ptr2.empty());
            REQUIRE(ptr2.elements() == elements);
            REQUIRE(ptr2.shape() == shape);
            REQUIRE(ptr2.ndim() == ndim);
            ptr1.reset(ptr2.release(), shape); // transfer ownership.
            REQUIRE_FALSE(ptr2);
            REQUIRE_FALSE(ptr2.get());
            REQUIRE(ptr2.empty());
            REQUIRE_FALSE(ptr2.elements());
            REQUIRE(ptr2.shape() == size3_t{0, 0, 0});
        }
        REQUIRE(ptr1);
        REQUIRE(ptr1.get());
        REQUIRE_FALSE(ptr1.empty());
        REQUIRE(ptr1.elements() == elements);
        REQUIRE(ptr1.ndim() == ndim);
    }

    AND_THEN("empty states") {
        Noa::CUDA::PtrArray<TestType> ptr1(shape);
        ptr1.reset(shape);
        ptr1.dispose();
        ptr1.dispose(); // no double delete.
        REQUIRE(ptr1.empty());
        REQUIRE_FALSE(ptr1);
        ptr1.reset();
    }
}
