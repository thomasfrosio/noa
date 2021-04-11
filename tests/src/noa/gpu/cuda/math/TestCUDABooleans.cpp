#include <noa/gpu/cuda/math/Booleans.h>

#include <noa/cpu/math/Booleans.h>
#include <noa/cpu/PtrHost.h>
#include <noa/gpu/cuda/PtrDevice.h>
#include <noa/gpu/cuda/PtrDevicePadded.h>
#include <noa/gpu/cuda/Memory.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace Noa;

TEMPLATE_TEST_CASE("CUDA: Booleans: contiguous", "[noa][cuda][math]", int, uint, float, double) {
    Test::Randomizer<TestType> randomizer(1., 10.);

    size_t elements = Test::IntRandomizer<size_t>(1, 16384).get();

    PtrHost<TestType> data(elements);
    PtrHost<bool> expected(elements);
    TestType value = randomizer.get();

    CUDA::PtrDevice<TestType> d_data(elements);
    CUDA::PtrDevice<bool> d_results(elements);
    PtrHost<bool> cuda_results(elements);

    Test::initDataRandom(data.get(), data.elements(), randomizer);
    Test::initDataZero(expected.get(), expected.elements());

    CUDA::Stream stream(CUDA::Stream::CONCURRENT);
    CUDA::Memory::copy(data.get(), d_data.get(), elements * sizeof(TestType));
    CUDA::Memory::copy(expected.get(), d_results.get(), elements * sizeof(bool));

    AND_THEN("isLess") {
        CUDA::Math::isLess(d_data.get(), value, d_results.get(), elements, stream);
        CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements, stream);
        Math::isLess(data.get(), value, expected.get(), elements);
        CUDA::Stream::synchronize(stream);

        int diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
    }

    AND_THEN("isGreater") {
        CUDA::Math::isGreater(d_data.get(), value, d_results.get(), elements, stream);
        CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements, stream);
        Math::isGreater(data.get(), value, expected.get(), elements);
        CUDA::Stream::synchronize(stream);

        int diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
    }

    AND_THEN("isWithin") {
        TestType low = Test::Randomizer<TestType>(1., 5.).get(), high = low + 3;
        CUDA::Math::isWithin(d_data.get(), low, high, d_results.get(), elements, stream);
        CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements, stream);
        Math::isWithin(data.get(), low, high, expected.get(), elements);
        CUDA::Stream::synchronize(stream);

        int diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
    }

    AND_THEN("logicNOT") {
        if constexpr (!Noa::Traits::is_float_v<TestType>) {
            Test::IntRandomizer<TestType> randomizer_int(0, 5);
            Test::initDataRandom(data.get(), data.elements(), randomizer_int);

            CUDA::Memory::copy(data.get(), d_data.get(), elements * sizeof(TestType), stream);
            CUDA::Math::logicNOT(d_data.get(), d_results.get(), elements, stream);
            CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements, stream);
            Math::logicNOT(data.get(), expected.get(), elements);
            CUDA::Stream::synchronize(stream);

            int diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }
    }
}

TEMPLATE_TEST_CASE("CUDA: Booleans: padded", "[noa][cuda][math]", int, uint, float, double) {
    Test::Randomizer<TestType> randomizer(1., 10.);

    size3_t shape = Test::getRandomShape(2);
    size_t elements = getElements(shape);

    PtrHost<TestType> data(elements);
    PtrHost<bool> expected(elements);
    TestType value = randomizer.get();

    CUDA::PtrDevicePadded<TestType> d_data(shape);
    CUDA::PtrDevicePadded<bool> d_results(shape);
    PtrHost<bool> cuda_results(elements);

    Test::initDataRandom(data.get(), data.elements(), randomizer);
    Test::initDataZero(expected.get(), expected.elements());

    CUDA::Stream stream(CUDA::Stream::CONCURRENT);
    CUDA::Memory::copy(data.get(), shape.x * sizeof(TestType), d_data.get(), d_data.pitch(), shape);
    CUDA::Memory::copy(expected.get(), shape.x, d_results.get(), d_results.pitch(), shape);

    AND_THEN("isLess") {
        CUDA::Math::isLess(d_data.get(), d_data.pitchElements(), value,
                           d_results.get(), d_results.pitchElements(), shape, stream);
        CUDA::Memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape, stream);
        Math::isLess(data.get(), value, expected.get(), elements);
        CUDA::Stream::synchronize(stream);

        int diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
    }

    AND_THEN("isGreater") {
        CUDA::Math::isGreater(d_data.get(), d_data.pitchElements(), value,
                           d_results.get(), d_results.pitchElements(), shape, stream);
        CUDA::Memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape, stream);
        Math::isGreater(data.get(), value, expected.get(), elements);
        CUDA::Stream::synchronize(stream);

        int diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
    }

    AND_THEN("isWithin") {
        TestType low = Test::Randomizer<TestType>(1., 5.).get(), high = low + 3;
        CUDA::Math::isWithin(d_data.get(), d_data.pitchElements(), low, high,
                              d_results.get(), d_results.pitchElements(), shape, stream);
        CUDA::Memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape, stream);
        Math::isWithin(data.get(), low, high, expected.get(), elements);
        CUDA::Stream::synchronize(stream);

        int diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
    }

    AND_THEN("logicNOT") {
        if constexpr (!Noa::Traits::is_float_v<TestType>) {
            Test::IntRandomizer<TestType> randomizer_int(0, 5);
            Test::initDataRandom(data.get(), data.elements(), randomizer_int);

            CUDA::Memory::copy(data.get(), shape.x * sizeof(TestType), d_data.get(), d_data.pitch(), shape, stream);
            CUDA::Math::logicNOT(d_data.get(), d_data.pitchElements(),
                                 d_results.get(), d_results.pitchElements(), shape, stream);
            CUDA::Memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape, stream);
            Math::logicNOT(data.get(), expected.get(), elements);
            CUDA::Stream::synchronize(stream);

            int diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }
    }
}
