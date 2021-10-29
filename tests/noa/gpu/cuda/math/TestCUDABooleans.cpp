#include <noa/gpu/cuda/math/Booleans.h>

#include <noa/cpu/math/Booleans.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("cuda::math:: booleans, contiguous", "[noa][cuda][math]", int, uint, float, double) {
    test::Randomizer<TestType> randomizer(1., 10.);

    size_t elements = test::IntRandomizer<size_t>(1, 16384).get();

    cpu::memory::PtrHost<TestType> data(elements);
    cpu::memory::PtrHost<bool> expected(elements);
    TestType value = randomizer.get();

    cuda::memory::PtrDevice<TestType> d_data(elements);
    cuda::memory::PtrDevice<bool> d_results(elements);
    cpu::memory::PtrHost<bool> cuda_results(elements);

    test::initDataRandom(data.get(), data.elements(), randomizer);
    test::initDataZero(expected.get(), expected.elements());

    cuda::Stream stream(cuda::Stream::CONCURRENT);
    cuda::memory::copy(data.get(), d_data.get(), elements);
    cuda::memory::copy(expected.get(), d_results.get(), elements);

    AND_THEN("isLess") {
        cuda::math::isLess(d_data.get(), value, d_results.get(), elements, stream);
        cuda::memory::copy(d_results.get(), cuda_results.get(), elements, stream);
        cpu::math::isLess(data.get(), value, expected.get(), elements);
        cuda::Stream::synchronize(stream);

        TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
    }

    AND_THEN("isGreater") {
        cuda::math::isGreater(d_data.get(), value, d_results.get(), elements, stream);
        cuda::memory::copy(d_results.get(), cuda_results.get(), elements, stream);
        cpu::math::isGreater(data.get(), value, expected.get(), elements);
        cuda::Stream::synchronize(stream);

        TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
    }

    AND_THEN("isWithin") {
        TestType low = test::Randomizer<TestType>(1., 5.).get(), high = low + 3;
        cuda::math::isWithin(d_data.get(), low, high, d_results.get(), elements, stream);
        cuda::memory::copy(d_results.get(), cuda_results.get(), elements, stream);
        cpu::math::isWithin(data.get(), low, high, expected.get(), elements);
        cuda::Stream::synchronize(stream);

        TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
    }

    AND_THEN("logicNOT") {
        if constexpr (!noa::traits::is_float_v<TestType>) {
            test::IntRandomizer<TestType> randomizer_int(0, 5);
            test::initDataRandom(data.get(), data.elements(), randomizer_int);

            cuda::memory::copy(data.get(), d_data.get(), elements, stream);
            cuda::math::logicNOT(d_data.get(), d_results.get(), elements, stream);
            cuda::memory::copy(d_results.get(), cuda_results.get(), elements, stream);
            cpu::math::logicNOT(data.get(), expected.get(), elements);
            cuda::Stream::synchronize(stream);

            TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
        }
    }
}

TEMPLATE_TEST_CASE("cuda::math:: booleans, padded", "[noa][cuda][math]", int, uint, float, double) {
    test::Randomizer<TestType> randomizer(1., 10.);

    size3_t shape = test::getRandomShape(2);
    size_t elements = noa::elements(shape);
    INFO(shape);
    cpu::memory::PtrHost<TestType> data(elements);
    cpu::memory::PtrHost<TestType> expected(elements);
    TestType value = randomizer.get();

    cuda::memory::PtrDevicePadded<TestType> d_data(shape);
    cuda::memory::PtrDevicePadded<TestType> d_results(shape);
    cpu::memory::PtrHost<TestType> cuda_results(elements);

    test::initDataRandom(data.get(), data.elements(), randomizer);
    test::initDataZero(expected.get(), expected.elements());

    cuda::Stream stream(cuda::Stream::CONCURRENT);
    cuda::memory::copy(data.get(), shape.x, d_data.get(), d_data.pitch(), shape, stream);
    cuda::memory::copy(expected.get(), shape.x, d_results.get(), d_results.pitch(), shape, stream);

    AND_THEN("isLess") {
        cuda::math::isLess(d_data.get(), d_data.pitch(), value,
                           d_results.get(), d_results.pitch(), shape, stream);
        cuda::memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape, stream);
        cpu::math::isLess(data.get(), value, expected.get(), elements);
        cuda::Stream::synchronize(stream);

        TestType diff = test::getDifference(expected.get(), cuda_results.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
    }

    AND_THEN("isGreater") {
        cuda::math::isGreater(d_data.get(), d_data.pitch(), value,
                              d_results.get(), d_results.pitch(), shape, stream);
        cuda::memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape, stream);
        cpu::math::isGreater(data.get(), value, expected.get(), elements);
        cuda::Stream::synchronize(stream);

        TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
    }

    AND_THEN("isWithin") {
        TestType low = test::Randomizer<TestType>(1., 5.).get(), high = low + 3;
        cuda::math::isWithin(d_data.get(), d_data.pitch(), low, high,
                             d_results.get(), d_results.pitch(), shape, stream);
        cuda::memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape, stream);
        cpu::math::isWithin(data.get(), low, high, expected.get(), elements);
        cuda::Stream::synchronize(stream);

        TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
    }

    AND_THEN("logicNOT") {
        if constexpr (!noa::traits::is_float_v<TestType>) {
            test::IntRandomizer<TestType> randomizer_int(0, 5);
            test::initDataRandom(data.get(), data.elements(), randomizer_int);

            cuda::memory::copy(data.get(), shape.x, d_data.get(), d_data.pitch(), shape, stream);
            cuda::math::logicNOT(d_data.get(), d_data.pitch(),
                                 d_results.get(), d_results.pitch(), shape, stream);
            cuda::memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape, stream);
            cpu::math::logicNOT(data.get(), expected.get(), elements);
            cuda::Stream::synchronize(stream);

            TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
        }
    }
}
