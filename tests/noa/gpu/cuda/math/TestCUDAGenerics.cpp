#include <noa/gpu/cuda/math/Generics.h>

#include <noa/cpu/math/Generics.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("CUDA: Generics: contiguous", "[noa][cuda][math]",
                   int, float, double, cfloat_t, cdouble_t) {
    size_t elements = test::IntRandomizer<size_t>(1, 50000).get();
    memory::PtrHost<TestType> data(elements);
    memory::PtrHost<TestType> expected(elements);
    cuda::memory::PtrDevice<TestType> d_data(elements);
    cuda::memory::PtrDevice<TestType> d_results(elements);
    memory::PtrHost<TestType> cuda_results(elements);

    AND_THEN("oneMinus, abs") {
        test::Randomizer<TestType> randomizer(-10., 10.);
        test::initDataRandom(data.get(), data.elements(), randomizer);
        test::initDataZero(expected.get(), expected.elements());

        cuda::Stream stream(cuda::STREAM_CONCURRENT);
        cuda::memory::copy(data.get(), d_data.get(), elements, stream);
        cuda::memory::copy(expected.get(), d_results.get(), elements, stream);
        cuda::math::oneMinus(d_data.get(), d_results.get(), elements, stream);
        cuda::memory::copy(d_results.get(), cuda_results.get(), elements, stream);
        math::oneMinus(data.get(), expected.get(), elements);
        cuda::Stream::synchronize(stream);

        TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));

        cuda::math::abs(d_data.get(), d_data.get(), elements, stream); // in-place
        cuda::memory::copy(d_data.get(), cuda_results.get(), elements, stream);
        math::abs(data.get(), expected.get(), elements);
        cuda::Stream::synchronize(stream);

        diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
    }

    AND_THEN("square") {
        test::Randomizer<TestType> randomizer(-10., 10.);
        test::initDataRandom(data.get(), data.elements(), randomizer);
        test::initDataZero(expected.get(), expected.elements());

        cuda::Stream stream(cuda::STREAM_CONCURRENT);
        cuda::memory::copy(data.get(), d_data.get(), elements, stream);
        cuda::memory::copy(expected.get(), d_results.get(), elements, stream);
        cuda::math::square(d_data.get(), d_results.get(), elements, stream);
        cuda::memory::copy(d_results.get(), cuda_results.get(), elements, stream);
        math::square(data.get(), expected.get(), elements);
        cuda::Stream::synchronize(stream);

        TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
    }

    AND_THEN("normalize") {
        if constexpr (noa::traits::is_complex_v<TestType>) {
            test::Randomizer<TestType> randomizer(-10., 10.);
            test::initDataRandom(data.get(), data.elements(), randomizer);
            test::initDataZero(expected.get(), expected.elements());

            cuda::Stream stream(cuda::STREAM_CONCURRENT);
            cuda::memory::copy(data.get(), d_data.get(), elements, stream);
            cuda::memory::copy(expected.get(), d_results.get(), elements, stream);
            cuda::math::normalize(d_data.get(), d_results.get(), elements, stream);
            cuda::memory::copy(d_results.get(), cuda_results.get(), elements, stream);
            math::normalize(data.get(), expected.get(), elements);
            cuda::Stream::synchronize(stream);

            TestType diff = test::getAverageNormalizedDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-6));
        }
    }

    AND_THEN("min, max, clamp") {
        if constexpr (!noa::traits::is_complex_v<TestType>) {
            test::Randomizer<TestType> randomizer(-1000., 1000.);
            test::initDataRandom(data.get(), data.elements(), randomizer);
            test::initDataZero(expected.get(), expected.elements());

            TestType low = randomizer.get(), high = low + 200;
            memory::PtrHost<TestType> rhs(elements);
            test::initDataRandom(rhs.get(), rhs.elements(), randomizer);
            cuda::memory::PtrDevice<TestType> d_rhs(elements);
            cuda::memory::copy(rhs.get(), d_rhs.get(), elements);

            cuda::Stream stream(cuda::STREAM_CONCURRENT);
            cuda::memory::copy(data.get(), d_data.get(), elements, stream);
            TestType diff;

            // min
            cuda::memory::copy(expected.get(), d_results.get(), elements, stream);
            cuda::math::min(d_data.get(), d_rhs.get(), d_results.get(), elements, stream);
            cuda::memory::copy(d_results.get(), cuda_results.get(), elements, stream);
            math::min(data.get(), rhs.get(), expected.get(), elements);
            cuda::Stream::synchronize(stream);

            diff = test::getDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-12));

            cuda::memory::copy(expected.get(), d_results.get(), elements, stream);
            cuda::math::min(d_data.get(), low, d_results.get(), elements, stream);
            cuda::memory::copy(d_results.get(), cuda_results.get(), elements, stream);
            math::min(data.get(), low, expected.get(), elements);
            cuda::Stream::synchronize(stream);

            diff = test::getDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-12));

            // max
            cuda::memory::copy(expected.get(), d_results.get(), elements, stream);
            cuda::math::max(d_data.get(), d_rhs.get(), d_results.get(), elements, stream);
            cuda::memory::copy(d_results.get(), cuda_results.get(), elements, stream);
            math::max(data.get(), rhs.get(), expected.get(), elements);
            cuda::Stream::synchronize(stream);

            diff = test::getDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-12));

            cuda::memory::copy(expected.get(), d_results.get(), elements, stream);
            cuda::math::max(d_data.get(), low, d_results.get(), elements, stream);
            cuda::memory::copy(d_results.get(), cuda_results.get(), elements, stream);
            math::max(data.get(), low, expected.get(), elements);
            cuda::Stream::synchronize(stream);

            diff = test::getDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-12));

            // clamp
            cuda::memory::copy(expected.get(), d_results.get(), elements, stream);
            cuda::math::clamp(d_data.get(), low, high, d_results.get(), elements, stream);
            cuda::memory::copy(d_results.get(), cuda_results.get(), elements, stream);
            math::clamp(data.get(), low, high, expected.get(), elements);
            cuda::Stream::synchronize(stream);

            diff = test::getDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-12));
        }
    }

    AND_THEN("others") {
        if constexpr (std::is_floating_point_v<TestType>) {
            test::Randomizer<TestType> randomizer(1., 10.);
            test::initDataRandom(data.get(), data.elements(), randomizer);
            test::initDataZero(expected.get(), expected.elements());

            cuda::Stream stream(cuda::STREAM_CONCURRENT);
            cuda::memory::copy(data.get(), d_data.get(), elements, stream);
            cuda::memory::copy(expected.get(), d_results.get(), elements, stream);
            cuda::Stream::synchronize(stream);
            TestType diff;

            // inverse
            cuda::math::inverse(d_data.get(), d_results.get(), elements, stream);
            cuda::memory::copy(d_results.get(), cuda_results.get(), elements, stream);
            math::inverse(data.get(), expected.get(), elements);
            cuda::Stream::synchronize(stream);
            diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-6));

            // sqrt
            cuda::math::sqrt(d_data.get(), d_results.get(), elements, stream);
            cuda::memory::copy(d_results.get(), cuda_results.get(), elements, stream);
            math::sqrt(data.get(), expected.get(), elements);
            cuda::Stream::synchronize(stream);
            diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-6));

            // rsqrt
            cuda::math::rsqrt(d_data.get(), d_results.get(), elements, stream);
            cuda::memory::copy(d_results.get(), cuda_results.get(), elements, stream);
            math::rsqrt(data.get(), expected.get(), elements);
            cuda::Stream::synchronize(stream);
            diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-6));

            // exp
            cuda::math::exp(d_data.get(), d_results.get(), elements, stream);
            cuda::memory::copy(d_results.get(), cuda_results.get(), elements, stream);
            math::exp(data.get(), expected.get(), elements);
            cuda::Stream::synchronize(stream);
            diff = test::getAverageNormalizedDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-6));

            // log
            cuda::math::log(d_data.get(), d_results.get(), elements, stream);
            cuda::memory::copy(d_results.get(), cuda_results.get(), elements, stream);
            math::log(data.get(), expected.get(), elements);
            cuda::Stream::synchronize(stream);
            diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-6));

            // cos
            cuda::math::cos(d_data.get(), d_results.get(), elements, stream);
            cuda::memory::copy(d_results.get(), cuda_results.get(), elements, stream);
            math::cos(data.get(), expected.get(), elements);
            cuda::Stream::synchronize(stream);
            diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-6));

            // sin
            cuda::math::sin(d_data.get(), d_results.get(), elements, stream);
            cuda::memory::copy(d_results.get(), cuda_results.get(), elements, stream);
            math::sin(data.get(), expected.get(), elements);
            cuda::Stream::synchronize(stream);
            diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-6));

            // pow
            TestType exponent = test::Randomizer<TestType>(-2, 2).get();
            cuda::math::pow(d_data.get(), exponent, d_results.get(), elements, stream);
            cuda::memory::copy(d_results.get(), cuda_results.get(), elements, stream);
            math::pow(data.get(), exponent, expected.get(), elements);
            cuda::Stream::synchronize(stream);
            diff = test::getAverageNormalizedDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-6));
        }
    }
}

TEMPLATE_TEST_CASE("CUDA: Generics: padded", "[noa][cuda][math]",
                   int, float, double, cfloat_t, cdouble_t) {
    size3_t shape = test::getRandomShape(2);
    size_t elements = getElements(shape);
    memory::PtrHost<TestType> data(elements);
    memory::PtrHost<TestType> expected(elements);
    cuda::memory::PtrDevicePadded<TestType> d_data(shape);
    cuda::memory::PtrDevicePadded<TestType> d_results(shape);
    memory::PtrHost<TestType> cuda_results(elements);

    AND_THEN("oneMinus, abs") {
        test::Randomizer<TestType> randomizer(-10., 10.);
        test::initDataRandom(data.get(), data.elements(), randomizer);
        test::initDataZero(expected.get(), expected.elements());

        cuda::Stream stream(cuda::STREAM_CONCURRENT);
        cuda::memory::copy(data.get(), shape.x, d_data.get(), d_data.pitch(), shape, stream);
        cuda::memory::copy(expected.get(), shape.x, d_results.get(), d_results.pitch(), shape, stream);
        cuda::math::oneMinus(d_data.get(), d_data.pitch(),
                             d_results.get(), d_results.pitch(), shape, stream);
        cuda::memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape, stream);
        math::oneMinus(data.get(), expected.get(), elements);
        cuda::Stream::synchronize(stream);

        TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));

        cuda::math::abs(d_data.get(), d_data.pitch(), d_data.get(), d_data.pitch(), shape,
                        stream); // in-place
        cuda::memory::copy(d_data.get(), d_data.pitch(), cuda_results.get(), shape.x, shape, stream);
        math::abs(data.get(), expected.get(), elements);
        cuda::Stream::synchronize(stream);

        diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
    }

    AND_THEN("square") {
        test::Randomizer<TestType> randomizer(-10., 10.);
        test::initDataRandom(data.get(), data.elements(), randomizer);
        test::initDataZero(expected.get(), expected.elements());

        cuda::Stream stream(cuda::STREAM_CONCURRENT);
        cuda::memory::copy(data.get(), shape.x, d_data.get(), d_data.pitch(), shape, stream);
        cuda::memory::copy(expected.get(), shape.x, d_results.get(), d_results.pitch(), shape, stream);
        cuda::math::square(d_data.get(), d_data.pitch(),
                           d_results.get(), d_results.pitch(), shape, stream);
        cuda::memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape, stream);
        math::square(data.get(), expected.get(), elements);
        cuda::Stream::synchronize(stream);

        TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
    }

    AND_THEN("normalize") {
        if constexpr (noa::traits::is_complex_v<TestType>) {
            test::Randomizer<TestType> randomizer(-10., 10.);
            test::initDataRandom(data.get(), data.elements(), randomizer);
            test::initDataZero(expected.get(), expected.elements());

            cuda::Stream stream(cuda::STREAM_CONCURRENT);
            cuda::memory::copy(data.get(), shape.x, d_data.get(), d_data.pitch(), shape, stream);
            cuda::memory::copy(expected.get(), shape.x, d_results.get(), d_results.pitch(), shape, stream);
            cuda::math::normalize(d_data.get(), d_data.pitch(),
                                  d_results.get(), d_results.pitch(), shape, stream);
            cuda::memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape, stream);
            math::normalize(data.get(), expected.get(), elements);
            cuda::Stream::synchronize(stream);

            TestType diff = test::getAverageNormalizedDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-6));
        }
    }

    AND_THEN("min, max, clamp") {
        if constexpr (!noa::traits::is_complex_v<TestType>) {
            test::Randomizer<TestType> randomizer(-1000., 1000.);
            test::initDataRandom(data.get(), data.elements(), randomizer);
            test::initDataZero(expected.get(), expected.elements());

            TestType low = randomizer.get(), high = low + 200;
            memory::PtrHost<TestType> rhs(elements);
            test::initDataRandom(rhs.get(), rhs.elements(), randomizer);
            cuda::memory::PtrDevice<TestType> d_rhs(elements);
            cuda::memory::copy(rhs.get(), d_rhs.get(), elements);

            cuda::Stream stream(cuda::STREAM_CONCURRENT);
            cuda::memory::copy(data.get(), shape.x, d_data.get(), d_data.pitch(), shape, stream);
            TestType diff;

            // min
            cuda::memory::copy(expected.get(), shape.x, d_results.get(), d_results.pitch(), shape, stream);
            cuda::math::min(d_data.get(), d_data.pitch(),
                            d_rhs.get(), shape.x,
                            d_results.get(), d_results.pitch(), shape, stream);
            cuda::memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape, stream);
            math::min(data.get(), rhs.get(), expected.get(), elements);
            cuda::Stream::synchronize(stream);

            diff = test::getDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-12));

            cuda::memory::copy(expected.get(), shape.x, d_results.get(), d_results.pitch(), shape, stream);
            cuda::math::min(d_data.get(), d_data.pitch(), low,
                            d_results.get(), d_results.pitch(), shape, stream);
            cuda::memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape, stream);
            math::min(data.get(), low, expected.get(), elements);
            cuda::Stream::synchronize(stream);

            diff = test::getDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-12));

            // max
            cuda::memory::copy(expected.get(), shape.x, d_results.get(), d_results.pitch(), shape, stream);
            cuda::math::max(d_data.get(), d_data.pitch(),
                            d_rhs.get(), shape.x,
                            d_results.get(), d_results.pitch(), shape, stream);
            cuda::memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape, stream);
            math::max(data.get(), rhs.get(), expected.get(), elements);
            cuda::Stream::synchronize(stream);

            diff = test::getDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-12));

            cuda::memory::copy(expected.get(), shape.x, d_results.get(), d_results.pitch(), shape, stream);
            cuda::math::max(d_data.get(), d_data.pitch(), low,
                            d_results.get(), d_results.pitch(), shape, stream);
            cuda::memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape, stream);
            math::max(data.get(), low, expected.get(), elements);
            cuda::Stream::synchronize(stream);

            diff = test::getDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-12));

            // clamp
            cuda::memory::copy(expected.get(), shape.x, d_results.get(), d_results.pitch(), shape, stream);
            cuda::math::clamp(d_data.get(), d_data.pitch(), low, high,
                              d_results.get(), d_results.pitch(), shape, stream);
            cuda::memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape, stream);
            math::clamp(data.get(), low, high, expected.get(), elements);
            cuda::Stream::synchronize(stream);

            diff = test::getDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-12));
        }
    }

    AND_THEN("others") {
        if constexpr (std::is_floating_point_v<TestType>) {
            test::Randomizer<TestType> randomizer(1., 10.);
            test::initDataRandom(data.get(), data.elements(), randomizer);
            test::initDataZero(expected.get(), expected.elements());

            cuda::Stream stream(cuda::STREAM_CONCURRENT);
            cuda::memory::copy(data.get(), shape.x, d_data.get(), d_data.pitch(), shape, stream);
            cuda::memory::copy(expected.get(), shape.x, d_results.get(), d_results.pitch(), shape, stream);
            cuda::Stream::synchronize(stream);
            TestType diff;

            // inverse
            cuda::math::inverse(d_data.get(), d_data.pitch(),
                                d_results.get(), d_results.pitch(), shape, stream);
            cuda::memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape, stream);
            math::inverse(data.get(), expected.get(), elements);
            cuda::Stream::synchronize(stream);
            diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-6));

            // sqrt
            cuda::math::sqrt(d_data.get(), d_data.pitch(),
                             d_results.get(), d_results.pitch(), shape, stream);
            cuda::memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape, stream);
            math::sqrt(data.get(), expected.get(), elements);
            cuda::Stream::synchronize(stream);
            diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-6));

            // rsqrt
            cuda::math::rsqrt(d_data.get(), d_data.pitch(),
                              d_results.get(), d_results.pitch(), shape, stream);
            cuda::memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape, stream);
            math::rsqrt(data.get(), expected.get(), elements);
            cuda::Stream::synchronize(stream);
            diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-6));

            // exp
            cuda::math::exp(d_data.get(), d_data.pitch(),
                            d_results.get(), d_results.pitch(), shape, stream);
            cuda::memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape, stream);
            math::exp(data.get(), expected.get(), elements);
            cuda::Stream::synchronize(stream);
            diff = test::getAverageNormalizedDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-6));

            // log
            cuda::math::log(d_data.get(), d_data.pitch(),
                            d_results.get(), d_results.pitch(), shape, stream);
            cuda::memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape, stream);
            math::log(data.get(), expected.get(), elements);
            cuda::Stream::synchronize(stream);
            diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-6));

            // cos
            cuda::math::cos(d_data.get(), d_data.pitch(),
                            d_results.get(), d_results.pitch(), shape, stream);
            cuda::memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape, stream);
            math::cos(data.get(), expected.get(), elements);
            cuda::Stream::synchronize(stream);
            diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-6));

            // sin
            cuda::math::sin(d_data.get(), d_data.pitch(),
                            d_results.get(), d_results.pitch(), shape, stream);
            cuda::memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape, stream);
            math::sin(data.get(), expected.get(), elements);
            cuda::Stream::synchronize(stream);
            diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-6));

            // pow
            TestType exponent = test::Randomizer<TestType>(-2, 2).get();
            cuda::math::pow(d_data.get(), d_data.pitch(), exponent,
                            d_results.get(), d_results.pitch(), shape, stream);
            cuda::memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape, stream);
            math::pow(data.get(), exponent, expected.get(), elements);
            cuda::Stream::synchronize(stream);
            diff = test::getAverageNormalizedDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-6));
        }
    }
}
