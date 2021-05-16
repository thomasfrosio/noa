#include <noa/gpu/cuda/math/Generics.h>

#include <noa/cpu/math/Generics.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace Noa;

TEMPLATE_TEST_CASE("CUDA: Generics: contiguous", "[noa][cuda][math]",
                   int, float, double, cfloat_t, cdouble_t) {
    size_t elements = Test::IntRandomizer<size_t>(1, 50000).get();
    Memory::PtrHost<TestType> data(elements);
    Memory::PtrHost<TestType> expected(elements);
    CUDA::Memory::PtrDevice<TestType> d_data(elements);
    CUDA::Memory::PtrDevice<TestType> d_results(elements);
    Memory::PtrHost<TestType> cuda_results(elements);

    AND_THEN("oneMinus, abs") {
        Test::Randomizer<TestType> randomizer(-10., 10.);
        Test::initDataRandom(data.get(), data.elements(), randomizer);
        Test::initDataZero(expected.get(), expected.elements());

        CUDA::Stream stream(CUDA::Stream::CONCURRENT);
        CUDA::Memory::copy(data.get(), d_data.get(), elements, stream);
        CUDA::Memory::copy(expected.get(), d_results.get(), elements, stream);
        CUDA::Math::oneMinus(d_data.get(), d_results.get(), elements, stream);
        CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements, stream);
        Math::oneMinus(data.get(), expected.get(), elements);
        CUDA::Stream::synchronize(stream);

        TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));

        CUDA::Math::abs(d_data.get(), d_data.get(), elements, stream); // in-place
        CUDA::Memory::copy(d_data.get(), cuda_results.get(), elements, stream);
        Math::abs(data.get(), expected.get(), elements);
        CUDA::Stream::synchronize(stream);

        diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
    }

    AND_THEN("square") {
        Test::Randomizer<TestType> randomizer(-10., 10.);
        Test::initDataRandom(data.get(), data.elements(), randomizer);
        Test::initDataZero(expected.get(), expected.elements());

        CUDA::Stream stream(CUDA::Stream::CONCURRENT);
        CUDA::Memory::copy(data.get(), d_data.get(), elements, stream);
        CUDA::Memory::copy(expected.get(), d_results.get(), elements, stream);
        CUDA::Math::square(d_data.get(), d_results.get(), elements, stream);
        CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements, stream);
        Math::square(data.get(), expected.get(), elements);
        CUDA::Stream::synchronize(stream);

        TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
    }

    AND_THEN("normalize") {
        if constexpr (Noa::Traits::is_complex_v<TestType>) {
            Test::Randomizer<TestType> randomizer(-10., 10.);
            Test::initDataRandom(data.get(), data.elements(), randomizer);
            Test::initDataZero(expected.get(), expected.elements());

            CUDA::Stream stream(CUDA::Stream::CONCURRENT);
            CUDA::Memory::copy(data.get(), d_data.get(), elements, stream);
            CUDA::Memory::copy(expected.get(), d_results.get(), elements, stream);
            CUDA::Math::normalize(d_data.get(), d_results.get(), elements, stream);
            CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements, stream);
            Math::normalize(data.get(), expected.get(), elements);
            CUDA::Stream::synchronize(stream);

            TestType diff = Test::getAverageNormalizedDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-6));
        }
    }

    AND_THEN("min, max, clamp") {
        if constexpr (!Noa::Traits::is_complex_v<TestType>) {
            Test::Randomizer<TestType> randomizer(-1000., 1000.);
            Test::initDataRandom(data.get(), data.elements(), randomizer);
            Test::initDataZero(expected.get(), expected.elements());

            TestType low = randomizer.get(), high = low + 200;
            Memory::PtrHost<TestType> rhs(elements);
            Test::initDataRandom(rhs.get(), rhs.elements(), randomizer);
            CUDA::Memory::PtrDevice<TestType> d_rhs(elements);
            CUDA::Memory::copy(rhs.get(), d_rhs.get(), elements);

            CUDA::Stream stream(CUDA::Stream::CONCURRENT);
            CUDA::Memory::copy(data.get(), d_data.get(), elements, stream);
            TestType diff;

            // min
            CUDA::Memory::copy(expected.get(), d_results.get(), elements, stream);
            CUDA::Math::min(d_data.get(), d_rhs.get(), d_results.get(), elements, stream);
            CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements, stream);
            Math::min(data.get(), rhs.get(), expected.get(), elements);
            CUDA::Stream::synchronize(stream);

            diff = Test::getDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-12));

            CUDA::Memory::copy(expected.get(), d_results.get(), elements, stream);
            CUDA::Math::min(d_data.get(), low, d_results.get(), elements, stream);
            CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements, stream);
            Math::min(data.get(), low, expected.get(), elements);
            CUDA::Stream::synchronize(stream);

            diff = Test::getDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-12));

            // max
            CUDA::Memory::copy(expected.get(), d_results.get(), elements, stream);
            CUDA::Math::max(d_data.get(), d_rhs.get(), d_results.get(), elements, stream);
            CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements, stream);
            Math::max(data.get(), rhs.get(), expected.get(), elements);
            CUDA::Stream::synchronize(stream);

            diff = Test::getDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-12));

            CUDA::Memory::copy(expected.get(), d_results.get(), elements, stream);
            CUDA::Math::max(d_data.get(), low, d_results.get(), elements, stream);
            CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements, stream);
            Math::max(data.get(), low, expected.get(), elements);
            CUDA::Stream::synchronize(stream);

            diff = Test::getDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-12));

            // clamp
            CUDA::Memory::copy(expected.get(), d_results.get(), elements, stream);
            CUDA::Math::clamp(d_data.get(), low, high, d_results.get(), elements, stream);
            CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements, stream);
            Math::clamp(data.get(), low, high, expected.get(), elements);
            CUDA::Stream::synchronize(stream);

            diff = Test::getDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-12));
        }
    }

    AND_THEN("others") {
        if constexpr (std::is_floating_point_v<TestType>) {
            Test::Randomizer<TestType> randomizer(1., 10.);
            Test::initDataRandom(data.get(), data.elements(), randomizer);
            Test::initDataZero(expected.get(), expected.elements());

            CUDA::Stream stream(CUDA::Stream::CONCURRENT);
            CUDA::Memory::copy(data.get(), d_data.get(), elements, stream);
            CUDA::Memory::copy(expected.get(), d_results.get(), elements, stream);
            CUDA::Stream::synchronize(stream);
            TestType diff;

            // inverse
            CUDA::Math::inverse(d_data.get(), d_results.get(), elements, stream);
            CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements, stream);
            Math::inverse(data.get(), expected.get(), elements);
            CUDA::Stream::synchronize(stream);
            diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-6));

            // sqrt
            CUDA::Math::sqrt(d_data.get(), d_results.get(), elements, stream);
            CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements, stream);
            Math::sqrt(data.get(), expected.get(), elements);
            CUDA::Stream::synchronize(stream);
            diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-6));

            // rsqrt
            CUDA::Math::rsqrt(d_data.get(), d_results.get(), elements, stream);
            CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements, stream);
            Math::rsqrt(data.get(), expected.get(), elements);
            CUDA::Stream::synchronize(stream);
            diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-6));

            // exp
            CUDA::Math::exp(d_data.get(), d_results.get(), elements, stream);
            CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements, stream);
            Math::exp(data.get(), expected.get(), elements);
            CUDA::Stream::synchronize(stream);
            diff = Test::getAverageNormalizedDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-6));

            // log
            CUDA::Math::log(d_data.get(), d_results.get(), elements, stream);
            CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements, stream);
            Math::log(data.get(), expected.get(), elements);
            CUDA::Stream::synchronize(stream);
            diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-6));

            // cos
            CUDA::Math::cos(d_data.get(), d_results.get(), elements, stream);
            CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements, stream);
            Math::cos(data.get(), expected.get(), elements);
            CUDA::Stream::synchronize(stream);
            diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-6));

            // sin
            CUDA::Math::sin(d_data.get(), d_results.get(), elements, stream);
            CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements, stream);
            Math::sin(data.get(), expected.get(), elements);
            CUDA::Stream::synchronize(stream);
            diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-6));

            // pow
            TestType exponent = Test::Randomizer<TestType>(-2, 2).get();
            CUDA::Math::pow(d_data.get(), exponent, d_results.get(), elements, stream);
            CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements, stream);
            Math::pow(data.get(), exponent, expected.get(), elements);
            CUDA::Stream::synchronize(stream);
            diff = Test::getAverageNormalizedDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-6));
        }
    }
}

TEMPLATE_TEST_CASE("CUDA: Generics: padded", "[noa][cuda][math]",
                   int, float, double, cfloat_t, cdouble_t) {
    size3_t shape = Test::getRandomShape(2);
    size_t elements = getElements(shape);
    Memory::PtrHost<TestType> data(elements);
    Memory::PtrHost<TestType> expected(elements);
    CUDA::Memory::PtrDevicePadded<TestType> d_data(shape);
    CUDA::Memory::PtrDevicePadded<TestType> d_results(shape);
    Memory::PtrHost<TestType> cuda_results(elements);

    AND_THEN("oneMinus, abs") {
        Test::Randomizer<TestType> randomizer(-10., 10.);
        Test::initDataRandom(data.get(), data.elements(), randomizer);
        Test::initDataZero(expected.get(), expected.elements());

        CUDA::Stream stream(CUDA::Stream::CONCURRENT);
        CUDA::Memory::copy(data.get(), shape.x, d_data.get(), d_data.pitch(), shape, stream);
        CUDA::Memory::copy(expected.get(), shape.x, d_results.get(), d_results.pitch(), shape, stream);
        CUDA::Math::oneMinus(d_data.get(), d_data.pitch(),
                             d_results.get(), d_results.pitch(), shape, stream);
        CUDA::Memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape, stream);
        Math::oneMinus(data.get(), expected.get(), elements);
        CUDA::Stream::synchronize(stream);

        TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));

        CUDA::Math::abs(d_data.get(), d_data.pitch(), d_data.get(), d_data.pitch(), shape,
                        stream); // in-place
        CUDA::Memory::copy(d_data.get(), d_data.pitch(), cuda_results.get(), shape.x, shape, stream);
        Math::abs(data.get(), expected.get(), elements);
        CUDA::Stream::synchronize(stream);

        diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
    }

    AND_THEN("square") {
        Test::Randomizer<TestType> randomizer(-10., 10.);
        Test::initDataRandom(data.get(), data.elements(), randomizer);
        Test::initDataZero(expected.get(), expected.elements());

        CUDA::Stream stream(CUDA::Stream::CONCURRENT);
        CUDA::Memory::copy(data.get(), shape.x, d_data.get(), d_data.pitch(), shape, stream);
        CUDA::Memory::copy(expected.get(), shape.x, d_results.get(), d_results.pitch(), shape, stream);
        CUDA::Math::square(d_data.get(), d_data.pitch(),
                           d_results.get(), d_results.pitch(), shape, stream);
        CUDA::Memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape, stream);
        Math::square(data.get(), expected.get(), elements);
        CUDA::Stream::synchronize(stream);

        TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
    }

    AND_THEN("normalize") {
        if constexpr (Noa::Traits::is_complex_v<TestType>) {
            Test::Randomizer<TestType> randomizer(-10., 10.);
            Test::initDataRandom(data.get(), data.elements(), randomizer);
            Test::initDataZero(expected.get(), expected.elements());

            CUDA::Stream stream(CUDA::Stream::CONCURRENT);
            CUDA::Memory::copy(data.get(), shape.x, d_data.get(), d_data.pitch(), shape, stream);
            CUDA::Memory::copy(expected.get(), shape.x, d_results.get(), d_results.pitch(), shape, stream);
            CUDA::Math::normalize(d_data.get(), d_data.pitch(),
                                  d_results.get(), d_results.pitch(), shape, stream);
            CUDA::Memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape, stream);
            Math::normalize(data.get(), expected.get(), elements);
            CUDA::Stream::synchronize(stream);

            TestType diff = Test::getAverageNormalizedDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-6));
        }
    }

    AND_THEN("min, max, clamp") {
        if constexpr (!Noa::Traits::is_complex_v<TestType>) {
            Test::Randomizer<TestType> randomizer(-1000., 1000.);
            Test::initDataRandom(data.get(), data.elements(), randomizer);
            Test::initDataZero(expected.get(), expected.elements());

            TestType low = randomizer.get(), high = low + 200;
            Memory::PtrHost<TestType> rhs(elements);
            Test::initDataRandom(rhs.get(), rhs.elements(), randomizer);
            CUDA::Memory::PtrDevice<TestType> d_rhs(elements);
            CUDA::Memory::copy(rhs.get(), d_rhs.get(), elements);

            CUDA::Stream stream(CUDA::Stream::CONCURRENT);
            CUDA::Memory::copy(data.get(), shape.x, d_data.get(), d_data.pitch(), shape, stream);
            TestType diff;

            // min
            CUDA::Memory::copy(expected.get(), shape.x, d_results.get(), d_results.pitch(), shape, stream);
            CUDA::Math::min(d_data.get(), d_data.pitch(),
                            d_rhs.get(), shape.x,
                            d_results.get(), d_results.pitch(), shape, stream);
            CUDA::Memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape, stream);
            Math::min(data.get(), rhs.get(), expected.get(), elements);
            CUDA::Stream::synchronize(stream);

            diff = Test::getDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-12));

            CUDA::Memory::copy(expected.get(), shape.x, d_results.get(), d_results.pitch(), shape, stream);
            CUDA::Math::min(d_data.get(), d_data.pitch(), low,
                            d_results.get(), d_results.pitch(), shape, stream);
            CUDA::Memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape, stream);
            Math::min(data.get(), low, expected.get(), elements);
            CUDA::Stream::synchronize(stream);

            diff = Test::getDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-12));

            // max
            CUDA::Memory::copy(expected.get(), shape.x, d_results.get(), d_results.pitch(), shape, stream);
            CUDA::Math::max(d_data.get(), d_data.pitch(),
                            d_rhs.get(), shape.x,
                            d_results.get(), d_results.pitch(), shape, stream);
            CUDA::Memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape, stream);
            Math::max(data.get(), rhs.get(), expected.get(), elements);
            CUDA::Stream::synchronize(stream);

            diff = Test::getDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-12));

            CUDA::Memory::copy(expected.get(), shape.x, d_results.get(), d_results.pitch(), shape, stream);
            CUDA::Math::max(d_data.get(), d_data.pitch(), low,
                            d_results.get(), d_results.pitch(), shape, stream);
            CUDA::Memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape, stream);
            Math::max(data.get(), low, expected.get(), elements);
            CUDA::Stream::synchronize(stream);

            diff = Test::getDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-12));

            // clamp
            CUDA::Memory::copy(expected.get(), shape.x, d_results.get(), d_results.pitch(), shape, stream);
            CUDA::Math::clamp(d_data.get(), d_data.pitch(), low, high,
                              d_results.get(), d_results.pitch(), shape, stream);
            CUDA::Memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape, stream);
            Math::clamp(data.get(), low, high, expected.get(), elements);
            CUDA::Stream::synchronize(stream);

            diff = Test::getDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-12));
        }
    }

    AND_THEN("others") {
        if constexpr (std::is_floating_point_v<TestType>) {
            Test::Randomizer<TestType> randomizer(1., 10.);
            Test::initDataRandom(data.get(), data.elements(), randomizer);
            Test::initDataZero(expected.get(), expected.elements());

            CUDA::Stream stream(CUDA::Stream::CONCURRENT);
            CUDA::Memory::copy(data.get(), shape.x, d_data.get(), d_data.pitch(), shape, stream);
            CUDA::Memory::copy(expected.get(), shape.x, d_results.get(), d_results.pitch(), shape, stream);
            CUDA::Stream::synchronize(stream);
            TestType diff;

            // inverse
            CUDA::Math::inverse(d_data.get(), d_data.pitch(),
                                d_results.get(), d_results.pitch(), shape, stream);
            CUDA::Memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape, stream);
            Math::inverse(data.get(), expected.get(), elements);
            CUDA::Stream::synchronize(stream);
            diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-6));

            // sqrt
            CUDA::Math::sqrt(d_data.get(), d_data.pitch(),
                             d_results.get(), d_results.pitch(), shape, stream);
            CUDA::Memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape, stream);
            Math::sqrt(data.get(), expected.get(), elements);
            CUDA::Stream::synchronize(stream);
            diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-6));

            // rsqrt
            CUDA::Math::rsqrt(d_data.get(), d_data.pitch(),
                              d_results.get(), d_results.pitch(), shape, stream);
            CUDA::Memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape, stream);
            Math::rsqrt(data.get(), expected.get(), elements);
            CUDA::Stream::synchronize(stream);
            diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-6));

            // exp
            CUDA::Math::exp(d_data.get(), d_data.pitch(),
                            d_results.get(), d_results.pitch(), shape, stream);
            CUDA::Memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape, stream);
            Math::exp(data.get(), expected.get(), elements);
            CUDA::Stream::synchronize(stream);
            diff = Test::getAverageNormalizedDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-6));

            // log
            CUDA::Math::log(d_data.get(), d_data.pitch(),
                            d_results.get(), d_results.pitch(), shape, stream);
            CUDA::Memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape, stream);
            Math::log(data.get(), expected.get(), elements);
            CUDA::Stream::synchronize(stream);
            diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-6));

            // cos
            CUDA::Math::cos(d_data.get(), d_data.pitch(),
                            d_results.get(), d_results.pitch(), shape, stream);
            CUDA::Memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape, stream);
            Math::cos(data.get(), expected.get(), elements);
            CUDA::Stream::synchronize(stream);
            diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-6));

            // sin
            CUDA::Math::sin(d_data.get(), d_data.pitch(),
                            d_results.get(), d_results.pitch(), shape, stream);
            CUDA::Memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape, stream);
            Math::sin(data.get(), expected.get(), elements);
            CUDA::Stream::synchronize(stream);
            diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-6));

            // pow
            TestType exponent = Test::Randomizer<TestType>(-2, 2).get();
            CUDA::Math::pow(d_data.get(), d_data.pitch(), exponent,
                            d_results.get(), d_results.pitch(), shape, stream);
            CUDA::Memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape, stream);
            Math::pow(data.get(), exponent, expected.get(), elements);
            CUDA::Stream::synchronize(stream);
            diff = Test::getAverageNormalizedDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-6));
        }
    }
}
