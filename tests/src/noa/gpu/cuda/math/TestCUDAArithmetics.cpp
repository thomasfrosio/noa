#include <noa/gpu/cuda/math/Arithmetics.h>

#include <noa/cpu/math/Arithmetics.h>
#include <noa/cpu/PtrHost.h>
#include <noa/gpu/cuda/PtrDevice.h>
#include <noa/gpu/cuda/PtrDevicePadded.h>
#include <noa/gpu/cuda/Memory.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace Noa;

TEMPLATE_TEST_CASE("CUDA: Arithmetics: contiguous", "[noa][cuda][math]",
                   int, uint, float, double, cfloat_t, cdouble_t) {
    Test::Randomizer<TestType> randomizer(1., 10.);

    size_t elements = Test::IntRandomizer<size_t>(1, 16384).get();
    uint batches = Test::IntRandomizer<uint>(1, 5).get();

    PtrHost<TestType> data(elements * batches);
    PtrHost<TestType> expected(elements * batches);
    PtrHost<TestType> values(batches);
    PtrHost<TestType> array(elements);
    TestType value = randomizer.get();

    CUDA::PtrDevice<TestType> d_data(elements * batches);
    CUDA::PtrDevice<TestType> d_values(batches);
    CUDA::PtrDevice<TestType> d_array(elements);
    CUDA::PtrDevice<TestType> d_results(elements * batches);
    PtrHost<TestType> cuda_results(elements * batches);

    Test::initDataRandom(data.get(), data.elements(), randomizer);
    Test::initDataZero(expected.get(), expected.elements());
    Test::initDataRandom(values.get(), values.elements(), randomizer);
    Test::initDataRandom(array.get(), array.elements(), randomizer);
    CUDA::Stream stream;

    CUDA::Memory::copy(data.get(), d_data.get(), elements * batches * sizeof(TestType));
    CUDA::Memory::copy(expected.get(), d_results.get(), elements * batches * sizeof(TestType));
    CUDA::Memory::copy(values.get(), d_values.get(), batches * sizeof(TestType));
    CUDA::Memory::copy(array.get(), d_array.get(), elements * sizeof(TestType));

    AND_GIVEN("multiply") {
        AND_THEN("value") {
            CUDA::Math::multiplyByValue(d_data.get(), value, d_results.get(), elements, stream);
            CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements * sizeof(TestType), stream);
            Math::multiplyByValue(data.get(), value, expected.get(), elements);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("values") {
            CUDA::Math::multiplyByValue(d_data.get(), d_values.get(), d_results.get(), elements, batches, stream);
            CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements * batches * sizeof(TestType), stream);
            Math::multiplyByValue(data.get(), values.get(), expected.get(), elements, batches);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("array") {
            CUDA::Math::multiplyByArray(d_data.get(), d_array.get(), d_results.get(), elements, batches, stream);
            CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements * batches * sizeof(TestType), stream);
            Math::multiplyByArray(data.get(), array.get(), expected.get(), elements, batches);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }
    }

    AND_GIVEN("divide") {
        AND_THEN("value") {
            CUDA::Math::divideByValue(d_data.get(), value, d_results.get(), elements, stream);
            CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements * sizeof(TestType), stream);
            Math::divideByValue(data.get(), value, expected.get(), elements);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("values") {
            CUDA::Math::divideByValue(d_data.get(), d_values.get(), d_results.get(), elements, batches, stream);
            CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements * batches * sizeof(TestType), stream);
            Math::divideByValue(data.get(), values.get(), expected.get(), elements, batches);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("array") {
            CUDA::Math::divideByArray(d_data.get(), d_array.get(), d_results.get(), elements, batches, stream);
            CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements * batches * sizeof(TestType), stream);
            Math::divideByArray(data.get(), array.get(), expected.get(), elements, batches);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }
    }

    AND_GIVEN("add") {
        AND_THEN("value") {
            CUDA::Math::addValue(d_data.get(), value, d_results.get(), elements, stream);
            CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements * sizeof(TestType), stream);
            Math::addValue(data.get(), value, expected.get(), elements);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("values") {
            CUDA::Math::addValue(d_data.get(), d_values.get(), d_results.get(), elements, batches, stream);
            CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements * batches * sizeof(TestType), stream);
            Math::addValue(data.get(), values.get(), expected.get(), elements, batches);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("array") {
            CUDA::Math::addArray(d_data.get(), d_array.get(), d_results.get(), elements, batches, stream);
            CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements * batches * sizeof(TestType), stream);
            Math::addArray(data.get(), array.get(), expected.get(), elements, batches);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }
    }

    AND_GIVEN("subtract") {
        AND_THEN("value") {
            CUDA::Math::subtractValue(d_data.get(), value, d_results.get(), elements, stream);
            CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements * sizeof(TestType), stream);
            Math::subtractValue(data.get(), value, expected.get(), elements);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("values") {
            CUDA::Math::subtractValue(d_data.get(), d_values.get(), d_results.get(), elements, batches, stream);
            CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements * batches * sizeof(TestType), stream);
            Math::subtractValue(data.get(), values.get(), expected.get(), elements, batches);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("array") {
            CUDA::Math::subtractArray(d_data.get(), d_array.get(), d_results.get(), elements, batches, stream);
            CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements * batches * sizeof(TestType), stream);
            Math::subtractArray(data.get(), array.get(), expected.get(), elements, batches);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }
    }
}

TEMPLATE_TEST_CASE("CUDA: Arithmetics: padded", "[noa][cuda][math]",
                   int, uint, float, double, cfloat_t, cdouble_t) {
    Test::Randomizer<TestType> randomizer(1., 10.);

    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape = Test::getRandomShape(ndim);
    size_t elements = getElements(shape);
    uint batches = Test::IntRandomizer<uint>(1, 5).get();

    PtrHost<TestType> data(elements * batches);
    PtrHost<TestType> expected(elements * batches);
    PtrHost<TestType> values(batches);
    PtrHost<TestType> array(elements);
    TestType value = randomizer.get();

    size3_t shape_batch = {shape.x, shape.y, shape.z * batches};
    CUDA::PtrDevicePadded<TestType> d_data(shape_batch);
    CUDA::PtrDevice<TestType> d_values(batches);
    CUDA::PtrDevicePadded<TestType> d_array(shape);
    CUDA::PtrDevicePadded<TestType> d_results(shape_batch);
    PtrHost<TestType> cuda_results(elements * batches);

    Test::initDataRandom(data.get(), data.elements(), randomizer);
    Test::initDataZero(expected.get(), expected.elements());
    Test::initDataRandom(values.get(), values.elements(), randomizer);
    Test::initDataRandom(array.get(), array.elements(), randomizer);
    CUDA::Stream stream;

    CUDA::Memory::copy(data.get(), shape.x * sizeof(TestType), d_data.get(), d_data.pitch(), shape_batch);
    CUDA::Memory::copy(expected.get(), shape.x * sizeof(TestType), d_results.get(), d_results.pitch(), shape_batch);
    CUDA::Memory::copy(values.get(), d_values.get(), batches * sizeof(TestType));
    CUDA::Memory::copy(array.get(), shape.x * sizeof(TestType), d_array.get(), d_array.pitch(), shape);

    AND_GIVEN("multiply") {
        AND_THEN("value") {
            CUDA::Math::multiplyByValue(d_data.get(), d_data.pitchElements(), value,
                                        d_results.get(), d_results.pitchElements(), shape, stream);
            CUDA::Memory::copy(d_results.get(), d_results.pitch(),
                               cuda_results.get(), shape.x * sizeof(TestType), shape, stream);
            Math::multiplyByValue(data.get(), value, expected.get(), elements);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("values") {
            CUDA::Math::multiplyByValue(d_data.get(), d_data.pitchElements(), d_values.get(),
                                        d_results.get(), d_results.pitchElements(), shape, batches, stream);
            CUDA::Memory::copy(d_results.get(), d_results.pitch(),
                               cuda_results.get(), shape.x * sizeof(TestType), shape_batch, stream);
            Math::multiplyByValue(data.get(), values.get(), expected.get(), elements, batches);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("array") {
            CUDA::Math::multiplyByArray(d_data.get(), d_data.pitchElements(), d_array.get(), d_array.pitchElements(),
                                        d_results.get(), d_array.pitchElements(), shape, batches, stream);
            CUDA::Memory::copy(d_results.get(), d_results.pitch(),
                               cuda_results.get(), shape.x * sizeof(TestType), shape_batch, stream);
            Math::multiplyByArray(data.get(), array.get(), expected.get(), elements, batches);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }
    }

    AND_GIVEN("divide") {
        AND_THEN("value") {
            CUDA::Math::divideByValue(d_data.get(), d_data.pitchElements(), value,
                                      d_results.get(), d_results.pitchElements(), shape, stream);
            CUDA::Memory::copy(d_results.get(), d_results.pitch(),
                               cuda_results.get(), shape.x * sizeof(TestType), shape, stream);
            Math::divideByValue(data.get(), value, expected.get(), elements);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("values") {
            CUDA::Math::divideByValue(d_data.get(), d_data.pitchElements(), d_values.get(),
                                      d_results.get(), d_results.pitchElements(), shape, batches, stream);
            CUDA::Memory::copy(d_results.get(), d_results.pitch(),
                               cuda_results.get(), shape.x * sizeof(TestType), shape_batch, stream);
            Math::divideByValue(data.get(), values.get(), expected.get(), elements, batches);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("array") {
            CUDA::Math::divideByArray(d_data.get(), d_data.pitchElements(), d_array.get(), d_array.pitchElements(),
                                      d_results.get(), d_array.pitchElements(), shape, batches, stream);
            CUDA::Memory::copy(d_results.get(), d_results.pitch(),
                               cuda_results.get(), shape.x * sizeof(TestType), shape_batch, stream);
            Math::divideByArray(data.get(), array.get(), expected.get(), elements, batches);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }
    }

    AND_GIVEN("add") {
        AND_THEN("value") {
            CUDA::Math::addValue(d_data.get(), d_data.pitchElements(), value,
                                 d_results.get(), d_results.pitchElements(), shape, stream);
            CUDA::Memory::copy(d_results.get(), d_results.pitch(),
                               cuda_results.get(), shape.x * sizeof(TestType), shape, stream);
            Math::addValue(data.get(), value, expected.get(), elements);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("values") {
            CUDA::Math::addValue(d_data.get(), d_data.pitchElements(), d_values.get(),
                                 d_results.get(), d_results.pitchElements(), shape, batches, stream);
            CUDA::Memory::copy(d_results.get(), d_results.pitch(),
                               cuda_results.get(), shape.x * sizeof(TestType), shape_batch, stream);
            Math::addValue(data.get(), values.get(), expected.get(), elements, batches);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("array") {
            CUDA::Math::addArray(d_data.get(), d_data.pitchElements(), d_array.get(), d_array.pitchElements(),
                                 d_results.get(), d_array.pitchElements(), shape, batches, stream);
            CUDA::Memory::copy(d_results.get(), d_results.pitch(),
                               cuda_results.get(), shape.x * sizeof(TestType), shape_batch, stream);
            Math::addArray(data.get(), array.get(), expected.get(), elements, batches);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }
    }

    AND_GIVEN("subtract") {
        AND_THEN("value") {
            CUDA::Math::subtractValue(d_data.get(), d_data.pitchElements(), value,
                                      d_results.get(), d_results.pitchElements(), shape, stream);
            CUDA::Memory::copy(d_results.get(), d_results.pitch(),
                               cuda_results.get(), shape.x * sizeof(TestType), shape, stream);
            Math::subtractValue(data.get(), value, expected.get(), elements);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("values") {
            CUDA::Math::subtractValue(d_data.get(), d_data.pitchElements(), d_values.get(),
                                      d_results.get(), d_results.pitchElements(), shape, batches, stream);
            CUDA::Memory::copy(d_results.get(), d_results.pitch(),
                               cuda_results.get(), shape.x * sizeof(TestType), shape_batch, stream);
            Math::subtractValue(data.get(), values.get(), expected.get(), elements, batches);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("array") {
            CUDA::Math::subtractArray(d_data.get(), d_data.pitchElements(), d_array.get(), d_array.pitchElements(),
                                      d_results.get(), d_array.pitchElements(), shape, batches, stream);
            CUDA::Memory::copy(d_results.get(), d_results.pitch(),
                               cuda_results.get(), shape.x * sizeof(TestType), shape_batch, stream);
            Math::subtractArray(data.get(), array.get(), expected.get(), elements, batches);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }
    }
}
