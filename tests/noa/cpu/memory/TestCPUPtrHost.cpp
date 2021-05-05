#include <noa/cpu/memory/PtrHost.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::Noa;

// These tests are not very good... mostly make sure it compiles and runs... This is OK since
// PtrHost is massively used in the code base and is therefore tested indirectly elsewhere.

TEMPLATE_TEST_CASE("PtrHost", "[noa][cpu]", int32_t, uint32_t, int64_t, uint64_t, float, double, cfloat_t, cdouble_t) {
    Test::IntRandomizer<size_t> randomizer(1, 128);
    Memory::PtrHost<TestType> ptr;

    AND_THEN("allocation, free, ownership") {
        ptr.reset(randomizer.get());
        if constexpr (std::is_same_v<TestType, cfloat_t>) {
            for (auto& elements: ptr)
                elements = TestType{static_cast<float>(randomizer.get())};
        } else if constexpr (std::is_same_v<TestType, cdouble_t>) {
            for (auto& elements: ptr)
                elements = TestType{static_cast<double>(randomizer.get())};
        } else {
            for (auto& elements: ptr)
                elements = static_cast<TestType>(randomizer.get());
        }
        Memory::PtrHost<TestType> ptr1(ptr.elements());
        std::memcpy(ptr1.get(), ptr.get(), ptr.bytes());
        TestType diff{0};
        for (size_t idx{0}; idx < ptr.elements(); ++idx)
            diff += ptr1[idx] - ptr[idx];
        REQUIRE(diff == TestType{0});

        ptr1.dispose();
        size_t elements = randomizer.get();
        {
            Memory::PtrHost<TestType> ptr2(elements);
            REQUIRE(ptr2);
            REQUIRE(ptr2.get());
            REQUIRE_FALSE(ptr2.empty());
            REQUIRE(ptr2.elements() == elements);
            REQUIRE(ptr2.bytes() == elements * sizeof(TestType));
            ptr1.reset(ptr2.release(), elements); // transfer ownership. ptr1 will dealloc its data.

            REQUIRE_FALSE(ptr2);
            REQUIRE_FALSE(ptr2.get());
            REQUIRE(ptr2.empty());
            REQUIRE_FALSE(ptr2.elements());
            REQUIRE_FALSE(ptr2.bytes());
        }
        REQUIRE(ptr1);
        REQUIRE(ptr1.get());
        REQUIRE_FALSE(ptr1.empty());
        REQUIRE(ptr1.elements() == elements);
        REQUIRE(ptr1.bytes() == elements * sizeof(TestType));
    }

    AND_THEN("transfer data") {
        Memory::PtrHost<TestType> ptr1;
        REQUIRE_FALSE(ptr1);
        size_t elements = randomizer.get();
        {
            Memory::PtrHost<TestType> ptr2(elements);
            ptr1 = std::move(ptr2);
        }
        REQUIRE(ptr1);
        REQUIRE(ptr1.elements() == elements);
    }

    AND_THEN("empty states") {
        Memory::PtrHost<TestType> ptr1(randomizer.get());
        ptr1.reset(randomizer.get());
        ptr1.dispose();
        ptr1.dispose(); // no double delete.

        ptr1.reset(0); // allocate but 0 elements...
        REQUIRE(ptr1.empty());
        REQUIRE_FALSE(ptr1);
        ptr1.reset();
    }
}
