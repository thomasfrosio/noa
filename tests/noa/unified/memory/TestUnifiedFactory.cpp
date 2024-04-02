#include <noa/unified/Array.hpp>
#include <noa/unified/Factory.hpp>

#include <catch2/catch.hpp>
#include "Helpers.h"

using namespace ::noa::types;

TEMPLATE_TEST_CASE("unified::arange(), cpu", "[noa][unified]",
                   i32, i64, u32, u64, f32, f64, c32, c64) {
    {
        const i64 elements = 100;
        const Array<TestType> results(elements);
        const Array<TestType> expected(elements);

        noa::arange(results, TestType(0), TestType(1));
        const auto expected_1d = expected.template accessor_contiguous_1d();
        for (i64 i = 0; auto& e: expected.span())
            e = static_cast<TestType>(i++);

        REQUIRE(test::allclose_abs(results, expected, 1e-10f));
    }

    {
        const i64 elements = 55;
        const Array<TestType> results(elements);
        const Array<TestType> expected(elements);

        noa::arange(results, TestType(3), TestType(5));
        const auto expected_1d = expected.template accessor_contiguous_1d();
        for (i64 i = 0; auto& e: expected.span())
            e = TestType(3) + static_cast<TestType>(i) * TestType(5);

        REQUIRE(test::allclose_abs(results, expected, 1e-10f));
    }

    {
        const auto shape = test::get_random_shape4_batched(3);
        const i64 elements = shape.elements();
        const Array<TestType> results(shape);
        const Array<TestType> expected(shape);

        noa::arange(results, TestType(3), TestType(5));
        for (i64 i = 0; auto& e: expected.span())
            e = TestType(3) + static_cast<TestType>(i) * TestType(5);

        REQUIRE(test::allclose_abs(results, expected, 1e-10f));
    }
}

TEST_CASE("unified::memory::linspace()", "[noa][unified]") {
    std::vector<Device> devices{"cpu"};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for(const auto& device: devices) {
        const StreamGuard guard(device, StreamMode::DEFAULT);
        const ArrayOption options(device, MemoryResource::MANAGED);

        {
            const i64 elements = 5;
            const auto results = noa::linspace(elements, 0., 5., false, options);
            std::array<double, 5> expected = {0, 1, 2, 3, 4};
            REQUIRE(test::Matcher(test::MATCH_ABS, results.eval().get(), expected.data(), elements, 1e-7));
        }

        {
            const i64 elements = 5;
            const Array<double> results(elements, options);
            const double step = noa::linspace(results.view(), 0., 5.);
            std::array<double, 5> expected = {0., 1.25, 2.5, 3.75, 5.};
            REQUIRE(step == 1.25);
            REQUIRE(test::Matcher(test::MATCH_ABS, results.eval().get(), expected.data(), elements, 1e-7));
        }

        {
            const i64 elements = 8;
            const Array<double> results(elements);
            const double step = noa::linspace(results, 0., 1.);
            std::array<double, 8> expected = {0., 0.14285714, 0.28571429, 0.42857143,
                                              0.57142857, 0.71428571, 0.85714286, 1.};
            REQUIRE_THAT(step, Catch::WithinAbs(0.14285714285714285, 1e-9));
            REQUIRE(test::Matcher(test::MATCH_ABS, results.eval().get(), expected.data(), elements, 1e-7));
        }

        {
            const i64 elements = 8;
            const Array<double> results(elements, options);
            const double step = noa::linspace(results, 0., 1., false);
            std::array<double, 8> expected = {0., 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875};
            REQUIRE(step == 0.125);
            REQUIRE(test::Matcher(test::MATCH_ABS, results.eval().get(), expected.data(), elements, 1e-7));
        }

        {
            const i64 elements = 9;
            const Array<double> results(elements, options);
            const double step = noa::linspace(results, 3., 40.);
            std::array<double, 9> expected = {3., 7.625, 12.25, 16.875, 21.5, 26.125, 30.75, 35.375, 40.};
            REQUIRE(step == 4.625);
            REQUIRE(test::Matcher(test::MATCH_ABS, results.eval().get(), expected.data(), elements, 1e-7));
        }

        {
            const i64 elements = 1;
            const Array<double> results(elements, options);
            const double step = noa::linspace(results, 0., 1., false);
            std::array<double, 1> expected = {0.};
            REQUIRE(step == 1.);
            REQUIRE(test::Matcher(test::MATCH_ABS, results.eval().get(), expected.data(), elements, 1e-7));
        }
    }
}

TEMPLATE_TEST_CASE("unified:: factories, cpu vs gpu", "[noa][unified]",
                   i32, i64, u32, u64, f32, f64, c32, c64) {
    const bool pad = GENERATE(false, true);
    const DeviceType type = GENERATE(DeviceType::CPU, DeviceType::GPU);
    INFO(pad);

    if (not Device::is_any(type))
        return;

    const auto subregion_shape = test::get_random_shape4_batched(3);
    auto shape = subregion_shape;
    if (pad) {
        shape[1] += 10;
        shape[2] += 11;
        shape[3] += 12;
    }

    const auto options = ArrayOption(Device(type), MemoryResource::MANAGED);
    INFO(options.device);

    auto results = Array<TestType>(shape, options);
    results = results.subregion(
            noa::indexing::Ellipsis{},
            noa::indexing::Slice{0, subregion_shape[1]},
            noa::indexing::Slice{0, subregion_shape[2]},
            noa::indexing::Slice{0, subregion_shape[3]});

    AND_THEN("arange") {
        const TestType start = test::Randomizer<TestType>(0, 3).get();
        const TestType step = test::Randomizer<TestType>(1, 2).get();
        noa::arange(results, start, step);

        const auto expected = Array<TestType>(results.shape());
        cpu::memory::arange(expected.get(), expected.elements(), start, step);

        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, results, expected, 1e-5));
    }

    AND_THEN("linspace") {
        if constexpr (noa::traits::is_real_v<TestType>) {
            const bool endpoint = GENERATE(true, false);
            const TestType start = test::Randomizer<TestType>(-5, 5).get();
            const TestType stop = test::Randomizer<TestType>(10, 50).get();

            noa::memory::linspace(results, start, stop, endpoint);

            const auto expected = Array<TestType>(results.shape());
            cpu::memory::linspace(expected.get(), expected.elements(), start, stop, endpoint);

            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, results, expected, 1e-6));
        }
    }

    AND_THEN("zeros/ones/fill") {
        const auto value = test::Randomizer<TestType>(0, 400).get();

        noa::fill(results, value);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, results, value, 1e-6));

        const auto zeros = noa::zeros<TestType>(shape, options);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, zeros, TestType{0}, 1e-6));

        const auto ones = noa::ones<TestType>(shape, options);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, ones, TestType{1}, 1e-6));

        const auto fill = noa::fill(shape, value, options);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, fill, value, 1e-6));
    }
}
