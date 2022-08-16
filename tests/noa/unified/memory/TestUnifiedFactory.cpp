#include <noa/unified/Array.h>
#include <noa/unified/memory/Factory.h>

#include <catch2/catch.hpp>
#include "Helpers.h"

using namespace ::noa;

TEMPLATE_TEST_CASE("unified::memory::arange()", "[noa][unified]",
                   int32_t, uint32_t, int64_t, uint64_t, float, double, cfloat_t, cdouble_t) {
    Device::Type type = GENERATE(Device::CPU, Device::GPU);
    if (!Device::any(type))
        return;

    const TestType start = 1;
    const TestType step = 2;
    const size4_t shape = test::getRandomShapeBatched(3);
    const size_t elements = shape.elements();

    Array<TestType> results = memory::arange(shape, start, step, {Device(type), Allocator::MANAGED});

    cpu::memory::PtrHost<TestType> expected(elements);
    cpu::memory::arange(expected.get(), elements, start, step);

    results.eval();
    REQUIRE(test::Matcher(test::MATCH_ABS, results.get(), expected.get(), elements, 1e-5));
}

TEST_CASE("unified::memory::linspace()", "[noa][unified]") {
    Device::Type type = GENERATE(Device::CPU, Device::GPU);
    if (!Device::any(type))
        return;

    const bool endpoint = GENERATE(true, false);
    const double start = test::Randomizer<double>(0, 5).get();
    const double stop = test::Randomizer<double>(5, 50).get();
    const size4_t shape = test::getRandomShapeBatched(3);
    const size_t elements = shape.elements();

    Array<double> results = memory::linspace(shape, start, stop, endpoint, {Device(type), Allocator::MANAGED});

    cpu::memory::PtrHost<double> expected(elements);
    cpu::memory::linspace(expected.get(), elements, start, stop, endpoint);

    results.eval();
    REQUIRE(test::Matcher(test::MATCH_ABS, results.get(), expected.get(), elements, 1e-6));
}

TEMPLATE_TEST_CASE("unified::memory::{zeros|ones|fill}()", "[noa][unified]",
                   int32_t, uint32_t, int64_t, uint64_t, float, double, cfloat_t, cdouble_t) {
    Device::Type type = GENERATE(Device::CPU, Device::GPU);
    if (!Device::any(type))
        return;

    const size4_t shape = test::getRandomShapeBatched(3);
    const ArrayOption options{Device(type), Allocator::MANAGED};
    Array<TestType> results;

    results = memory::zeros<TestType>(shape, options);
    results.eval();
    REQUIRE(test::Matcher(test::MATCH_ABS, results, TestType{0}, 1e-6));

    results = memory::ones<TestType>(shape, options);
    results.eval();
    REQUIRE(test::Matcher(test::MATCH_ABS, results, TestType{1}, 1e-6));

    results = memory::fill(shape, TestType{5}, options);
    results.eval();
    REQUIRE(test::Matcher(test::MATCH_ABS, results, TestType{5}, 1e-6));
}
