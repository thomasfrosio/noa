#include <cuda_runtime_api.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/memory/PtrArray.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEMPLATE_TEST_CASE("cuda::memory::PtrArray", "[noa][cuda][memory]", int32_t, uint32_t, float, cfloat_t) {
    test::Randomizer<size_t> randomizer_small(1, 64);
    uint ndim = GENERATE(1U, 2U, 3U);
    const size3_t shape = size3_t{test::getRandomShape(ndim).get() + 1};
    const size_t elements = shape.elements();

    // test allocation and free
    AND_THEN("allocation, free, ownership") {
        cuda::memory::PtrArray<TestType> ptr1;
        REQUIRE_FALSE(ptr1);
        {
            cuda::memory::PtrArray<TestType> ptr2(shape);
            REQUIRE(ptr2);
            REQUIRE(ptr2.get());
            REQUIRE_FALSE(ptr2.empty());
            REQUIRE(ptr2.elements() == elements);
            REQUIRE(all(ptr2.shape() == shape));
            ptr1 = std::move(ptr2);
        }
        REQUIRE(ptr1);
        REQUIRE(ptr1.get());
        REQUIRE_FALSE(ptr1.empty());
        REQUIRE(ptr1.elements() == elements);
    }

    AND_THEN("empty states") {
        cuda::memory::PtrArray<TestType> ptr1(shape);
        ptr1 = cuda::memory::PtrArray<TestType>(shape);
        ptr1 = nullptr; // no double delete.
        ptr1 = nullptr;
        REQUIRE(ptr1.empty());
        REQUIRE_FALSE(ptr1);
        ptr1.release();
    }
}
