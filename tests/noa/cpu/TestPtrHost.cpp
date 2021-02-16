#include <noa/cpu/PtrHost.h>

#include "../../Helpers.h"
#include <catch2/catch.hpp>

using namespace ::Noa;

// These tests are not very good... mostly make sure it compiles and runs...
TEMPLATE_TEST_CASE("PtrHost: real types", "[noa][cpu]", int32_t, uint32_t, int64_t, uint64_t, float, double) {
    PtrHost<TestType> ptr;
    Test::IntRandomizer<size_t> randomizer(1, 255);

    // test allocation and free
    AND_THEN("allocation, free, ownership") {
        ptr.reset(randomizer.get());
        for (auto& elements: ptr)
            elements = static_cast<TestType>(1);
        TestType sum{0};
        for (auto elements: ptr)
            sum += elements;
        REQUIRE(static_cast<size_t>(sum) == ptr.elements());

        PtrHost<TestType> ptr1(randomizer.get());
        size_t elements = randomizer.get();
        {
            PtrHost<TestType> ptr2(elements);
            REQUIRE(ptr2);
            REQUIRE(ptr2.get());
            REQUIRE_FALSE(ptr2.empty());
            REQUIRE(ptr2.elements() == elements);
            REQUIRE(ptr2.bytes() == elements * sizeof(TestType));
            TestType* data = ptr2.release();
            ptr1.reset(data, elements, true); // transfer ownership. ptr1 will dealloc its data.

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
        ptr.reset(randomizer.get());
        for (auto& elements: ptr)
            elements = static_cast<TestType>(3);
        TestType sum{0};
        for (size_t idx{0}; idx < ptr.elements(); ++idx)
            sum += ptr[idx];
        REQUIRE(static_cast<size_t>(sum) == 3 * ptr.elements());

        PtrHost<TestType> ptr1;
        size_t elements = randomizer.get();
        {
            PtrHost<TestType> ptr2(elements);
            ptr2.is_owner = false;
            ptr1.reset(ptr2.get(), ptr2.elements(), true);
            ptr1.dispose(); // should not deallocate since isn't owner.
        }
    }

    AND_THEN("empty states") {
        PtrHost<TestType> ptr1(randomizer.get());
        ptr1.reset(randomizer.get());
        ptr1.dispose();
        ptr1.dispose(); // no double delete.

        ptr1.reset(0); // allocate but 0 elements...
        REQUIRE(ptr.empty());
        REQUIRE_FALSE(ptr);
        ptr1.reset();
    }

    AND_THEN("const qualifier") {
        const PtrHost<TestType> ptr1;
        PtrHost<const TestType> ptr2(ptr1.get(), 0, false);
        REQUIRE_FALSE(ptr1);
        REQUIRE_FALSE(ptr2);
    }

    AND_THEN("toString") {
        INFO(ptr); // uses ostream <<
        INFO(fmt::format("{}", ptr));
        REQUIRE_FALSE(ptr);
    }
}

// These tests are not very good... mostly make sure it compiles and runs...
TEMPLATE_TEST_CASE("PtrHost: complex types", "[noa][cpu]", cfloat_t, cdouble_t) {
    PtrHost<TestType> ptr;
    Test::IntRandomizer<size_t> randomizer(1, 255);

    AND_THEN("allocation, free, ownership") {
        ptr.reset(randomizer.get());
        for (auto& elements: ptr)
            elements = TestType{1, 2};

        PtrHost<TestType> ptr1(randomizer.get());
        size_t elements = randomizer.get();
        {
            PtrHost<TestType> ptr2(elements);
            REQUIRE(ptr2);
            REQUIRE(ptr2.get());
            REQUIRE_FALSE(ptr2.empty());
            REQUIRE(ptr2.elements() == elements);
            REQUIRE(ptr2.bytes() == elements * sizeof(TestType));
            ptr1.reset(ptr2.release(), elements, true); // transfer ownership. ptr1 will dealloc its data.

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
        ptr.reset(randomizer.get());
        for (auto& elements: ptr)
            elements = static_cast<TestType>(3);
        PtrHost<TestType> ptr1;
        size_t elements = randomizer.get();
        {
            PtrHost<TestType> ptr2(elements);
            ptr1.reset(ptr2.get(), ptr2.elements(), true);
            ptr2.is_owner = false;
            ptr.dispose(); // should not deallocate since isn't owner.
        }
    }

    AND_THEN("empty states") {
        PtrHost<TestType> ptr1(randomizer.get());
        ptr1.reset(randomizer.get());
        ptr1.dispose();
        ptr1.dispose(); // no double delete.

        ptr1.reset(0); // allocate but 0 elements...
        REQUIRE(ptr.empty());
        REQUIRE_FALSE(ptr);
        ptr1.reset();
    }

    AND_THEN("const qualifier") {
        const PtrHost<TestType> ptr1;
        PtrHost<const TestType> ptr2(ptr1.get(), 0, false);
        REQUIRE_FALSE(ptr1);
        REQUIRE_FALSE(ptr2);
    }

    AND_THEN("toString") {
        INFO(ptr); // uses ostream <<
        INFO(fmt::format("{}", ptr));
        REQUIRE_FALSE(ptr);
    }
}
