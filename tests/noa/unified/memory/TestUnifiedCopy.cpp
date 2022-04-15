#include <noa/unified/Array.h>
#include <noa/unified/memory/Copy.h>
#include <noa/unified/memory/Initialize.h>

#include <catch2/catch.hpp>

#include "Helpers.h"

using namespace ::noa;

TEMPLATE_TEST_CASE("unified::memory::copy", "[noa][unified]", int32_t, float, double, cfloat_t) {
    const size4_t shape = test::getRandomShapeBatched(3);
    Device cpu{"cpu"};

    AND_THEN("cpu -> cpu") {
        Array<TestType> a{shape};
        memory::fill(a, TestType{3});

        Array<TestType> b{shape};
        memory::copy(a, b);
        b.stream().synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, a.get(), b.get(), shape.elements(), 1e-8));

        memory::fill(a, TestType{4});
        b = a.to(cpu);
        b.stream().synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, a.get(), b.get(), shape.elements(), 1e-8));
    }

    if (!Device::any(Device::GPU))
        return;

    AND_THEN("cpu -> gpu -> gpu -> cpu") {
        Device gpu{"gpu"};

        Array<TestType> a{shape};
        memory::fill(a, TestType{3});

        Array<TestType> b{shape, {gpu, Allocator::DEFAULT_ASYNC}};
        Array<TestType> c{shape, {gpu, Allocator::PITCHED}};
        memory::copy(a, b);
        memory::copy(b, c);

        Array<TestType> d = c.to(cpu);
        b.stream().synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, a.get(), d.get(), shape.elements(), 1e-8));
    }
}
