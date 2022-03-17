#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/memory/Arange.h>
#include <noa/cpu/memory/Linspace.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEMPLATE_TEST_CASE("cpu::memory::arange()", "[noa][cpu][memory]",
                   int32_t, uint32_t, int64_t, uint64_t, float, double, cfloat_t, cdouble_t) {
    cpu::Stream stream;
    {
        const size_t elements = 100;
        cpu::memory::PtrHost<TestType> results(elements);
        cpu::memory::PtrHost<TestType> expected(elements);

        cpu::memory::arange(results.get(), elements, stream);
        for (size_t i = 0; i < elements; ++i)
            expected[i] = static_cast<TestType>(i);

        REQUIRE(test::Matcher(test::MATCH_ABS, results.get(), expected.get(), elements, 1e-10f));
    }

    {
        const size_t elements = 55;
        cpu::memory::PtrHost<TestType> results(elements);
        cpu::memory::PtrHost<TestType> expected(elements);

        cpu::memory::arange(results.get(), elements, TestType(3), TestType(5), stream);
        TestType v = 3;
        for (size_t i = 0; i < elements; ++i, v += TestType(5))
            expected[i] = v;

        REQUIRE(test::Matcher(test::MATCH_ABS, results.get(), expected.get(), elements, 1e-10f));
    }

    {
        const size4_t shape = test::getRandomShapeBatched(3);
        const size_t elements = shape.elements();
        cpu::memory::PtrHost<TestType> results(elements);
        cpu::memory::PtrHost<TestType> expected(elements);

        cpu::memory::arange(results.get(), shape.stride(), shape, TestType(3), TestType(5), stream);
        TestType v = 3;
        for (size_t i = 0; i < elements; ++i, v += TestType(5))
            expected[i] = v;

        REQUIRE(test::Matcher(test::MATCH_ABS, results.get(), expected.get(), elements, 1e-10f));
    }
}

TEST_CASE("cpu::memory::linspace()", "[noa][cpu][memory]") {
    cpu::Stream stream;
    {
        const size_t elements = 5;
        cpu::memory::PtrHost<double> results(elements);
        cpu::memory::linspace(results.get(), elements, 0., 5., false, stream);
        std::array<double, 5> expected = {0, 1, 2, 3, 4};
        REQUIRE(test::Matcher(test::MATCH_ABS, results.get(), expected.data(), elements, 1e-7));
    }

    {
        const size4_t shape = test::getRandomShapeBatched(3);
        const size_t elements = shape.elements();
        cpu::memory::PtrHost<double> results(elements);
        cpu::memory::PtrHost<double> expected(elements);
        cpu::memory::linspace(expected.get(), elements, 0., 5., false, stream);
        cpu::memory::linspace(results.get(), shape.stride(), shape, 0., 5., false, stream);
        REQUIRE(test::Matcher(test::MATCH_ABS, results.get(), expected.data(), elements, 1e-7));
    }

    {
        const size_t elements = 5;
        cpu::memory::PtrHost<double> results(elements);
        double step = cpu::memory::linspace(results.get(), elements, 0., 5.);
        std::array<double, 5> expected = {0., 1.25, 2.5, 3.75, 5.};
        REQUIRE(step == 1.25);
        REQUIRE(test::Matcher(test::MATCH_ABS, results.get(), expected.data(), elements, 1e-7));
    }

    {
        const size_t elements = 8;
        cpu::memory::PtrHost<double> results(elements);
        double step = cpu::memory::linspace(results.get(), elements, 0., 1.);
        std::array<double, 8> expected = {0., 0.14285714, 0.28571429, 0.42857143,
                                          0.57142857, 0.71428571, 0.85714286, 1.};
        REQUIRE_THAT(step, Catch::WithinAbs(0.14285714285714285, 1e-9));
        REQUIRE(test::Matcher(test::MATCH_ABS, results.get(), expected.data(), elements, 1e-7));
    }

    {
        const size_t elements = 8;
        cpu::memory::PtrHost<double> results(elements);
        double step = cpu::memory::linspace(results.get(), elements, 0., 1., false);
        std::array<double, 8> expected = {0., 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875};
        REQUIRE(step == 0.125);
        REQUIRE(test::Matcher(test::MATCH_ABS, results.get(), expected.data(), elements, 1e-7));
    }

    {
        const size_t elements = 9;
        cpu::memory::PtrHost<double> results(elements);
        double step = cpu::memory::linspace(results.get(), elements, 3., 40.);
        std::array<double, 9> expected = {3., 7.625, 12.25, 16.875, 21.5, 26.125, 30.75, 35.375, 40.};
        REQUIRE(step == 4.625);
        REQUIRE(test::Matcher(test::MATCH_ABS, results.get(), expected.data(), elements, 1e-7));
    }

    {
        const size_t elements = 1;
        cpu::memory::PtrHost<double> results(elements);
        double step = cpu::memory::linspace(results.get(), elements, 0., 1., false);
        std::array<double, 1> expected = {0.};
        REQUIRE(step == 0.);
        REQUIRE(test::Matcher(test::MATCH_ABS, results.get(), expected.data(), elements, 1e-7));
    }
}
