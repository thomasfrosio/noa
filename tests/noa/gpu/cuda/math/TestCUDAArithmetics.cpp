#include <noa/gpu/cuda/math/Arithmetics.h>

#include <noa/cpu/math/Arithmetics.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace Noa;

TEMPLATE_TEST_CASE("CUDA: Arithmetics: contiguous", "[noa][cuda][math]",
                   int, uint, float, double, cfloat_t, cdouble_t) {
    Test::Randomizer<TestType> randomizer(1., 10.);

    size_t elements = Test::IntRandomizer<size_t>(1, 16384).get();
    uint batches = Test::IntRandomizer<uint>(1, 5).get();

    Memory::PtrHost<TestType> data(elements * batches);
    Memory::PtrHost<TestType> expected(elements * batches);
    Memory::PtrHost<TestType> values(batches);
    Memory::PtrHost<TestType> array(elements);
    TestType value = randomizer.get();

    CUDA::Memory::PtrDevice<TestType> d_data(elements * batches);
    CUDA::Memory::PtrDevice<TestType> d_values(batches);
    CUDA::Memory::PtrDevice<TestType> d_array(elements);
    CUDA::Memory::PtrDevice<TestType> d_results(elements * batches);
    Memory::PtrHost<TestType> cuda_results(elements * batches);

    Test::initDataRandom(data.get(), data.elements(), randomizer);
    Test::initDataZero(expected.get(), expected.elements());
    Test::initDataRandom(values.get(), values.elements(), randomizer);
    Test::initDataRandom(array.get(), array.elements(), randomizer);
    CUDA::Stream stream;

    CUDA::Memory::copy(data.get(), d_data.get(), elements * batches);
    CUDA::Memory::copy(expected.get(), d_results.get(), elements * batches);
    CUDA::Memory::copy(values.get(), d_values.get(), batches);
    CUDA::Memory::copy(array.get(), d_array.get(), elements);

    AND_GIVEN("multiply") {
        AND_THEN("value") {
            CUDA::Math::multiplyByValue(d_data.get(), value, d_results.get(), elements, stream);
            CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements, stream);
            Math::multiplyByValue(data.get(), value, expected.get(), elements);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("values") {
            CUDA::Math::multiplyByValue(d_data.get(), d_values.get(), d_results.get(), elements, batches, stream);
            CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements * batches, stream);
            Math::multiplyByValue(data.get(), values.get(), expected.get(), elements, batches);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("array") {
            CUDA::Math::multiplyByArray(d_data.get(), d_array.get(), d_results.get(), elements, batches, stream);
            CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements * batches, stream);
            Math::multiplyByArray(data.get(), array.get(), expected.get(), elements, batches);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }
    }

    AND_GIVEN("divide") {
        AND_THEN("value") {
            CUDA::Math::divideByValue(d_data.get(), value, d_results.get(), elements, stream);
            CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements, stream);
            Math::divideByValue(data.get(), value, expected.get(), elements);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("values") {
            CUDA::Math::divideByValue(d_data.get(), d_values.get(), d_results.get(), elements, batches, stream);
            CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements * batches, stream);
            Math::divideByValue(data.get(), values.get(), expected.get(), elements, batches);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("array") {
            CUDA::Math::divideByArray(d_data.get(), d_array.get(), d_results.get(), elements, batches, stream);
            CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements * batches, stream);
            Math::divideByArray(data.get(), array.get(), expected.get(), elements, batches);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }
    }

    AND_GIVEN("add") {
        AND_THEN("value") {
            CUDA::Math::addValue(d_data.get(), value, d_results.get(), elements, stream);
            CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements, stream);
            Math::addValue(data.get(), value, expected.get(), elements);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("values") {
            CUDA::Math::addValue(d_data.get(), d_values.get(), d_results.get(), elements, batches, stream);
            CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements * batches, stream);
            Math::addValue(data.get(), values.get(), expected.get(), elements, batches);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("array") {
            CUDA::Math::addArray(d_data.get(), d_array.get(), d_results.get(), elements, batches, stream);
            CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements * batches, stream);
            Math::addArray(data.get(), array.get(), expected.get(), elements, batches);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }
    }

    AND_GIVEN("subtract") {
        AND_THEN("value") {
            CUDA::Math::subtractValue(d_data.get(), value, d_results.get(), elements, stream);
            CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements, stream);
            Math::subtractValue(data.get(), value, expected.get(), elements);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("values") {
            CUDA::Math::subtractValue(d_data.get(), d_values.get(), d_results.get(), elements, batches, stream);
            CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements * batches, stream);
            Math::subtractValue(data.get(), values.get(), expected.get(), elements, batches);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("array") {
            CUDA::Math::subtractArray(d_data.get(), d_array.get(), d_results.get(), elements, batches, stream);
            CUDA::Memory::copy(d_results.get(), cuda_results.get(), elements * batches, stream);
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

    Memory::PtrHost<TestType> data(elements * batches);
    Memory::PtrHost<TestType> expected(elements * batches);
    Memory::PtrHost<TestType> values(batches);
    Memory::PtrHost<TestType> array(elements);
    TestType value = randomizer.get();

    size3_t shape_batch = {shape.x, shape.y, shape.z * batches};
    CUDA::Memory::PtrDevicePadded<TestType> d_data(shape_batch);
    CUDA::Memory::PtrDevice<TestType> d_values(batches);
    CUDA::Memory::PtrDevicePadded<TestType> d_array(shape);
    CUDA::Memory::PtrDevicePadded<TestType> d_results(shape_batch);
    Memory::PtrHost<TestType> cuda_results(elements * batches);

    Test::initDataRandom(data.get(), data.elements(), randomizer);
    Test::initDataZero(expected.get(), expected.elements());
    Test::initDataRandom(values.get(), values.elements(), randomizer);
    Test::initDataRandom(array.get(), array.elements(), randomizer);
    CUDA::Stream stream;

    CUDA::Memory::copy(data.get(), shape.x, d_data.get(), d_data.pitch(), shape_batch);
    CUDA::Memory::copy(expected.get(), shape.x, d_results.get(), d_results.pitch(), shape_batch);
    CUDA::Memory::copy(values.get(), d_values.get(), batches);
    CUDA::Memory::copy(array.get(), shape.x, d_array.get(), d_array.pitch(), shape);

    AND_GIVEN("multiply") {
        AND_THEN("value") {
            CUDA::Math::multiplyByValue(d_data.get(), d_data.pitch(), value,
                                        d_results.get(), d_results.pitch(), shape, stream);
            CUDA::Memory::copy(d_results.get(), d_results.pitch(),
                               cuda_results.get(), shape.x, shape, stream);
            Math::multiplyByValue(data.get(), value, expected.get(), elements);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("values") {
            CUDA::Math::multiplyByValue(d_data.get(), d_data.pitch(), d_values.get(),
                                        d_results.get(), d_results.pitch(), shape, batches, stream);
            CUDA::Memory::copy(d_results.get(), d_results.pitch(),
                               cuda_results.get(), shape.x, shape_batch, stream);
            Math::multiplyByValue(data.get(), values.get(), expected.get(), elements, batches);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("array") {
            CUDA::Math::multiplyByArray(d_data.get(), d_data.pitch(), d_array.get(), d_array.pitch(),
                                        d_results.get(), d_array.pitch(), shape, batches, stream);
            CUDA::Memory::copy(d_results.get(), d_results.pitch(),
                               cuda_results.get(), shape.x, shape_batch, stream);
            Math::multiplyByArray(data.get(), array.get(), expected.get(), elements, batches);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }
    }

    AND_GIVEN("divide") {
        AND_THEN("value") {
            CUDA::Math::divideByValue(d_data.get(), d_data.pitch(), value,
                                      d_results.get(), d_results.pitch(), shape, stream);
            CUDA::Memory::copy(d_results.get(), d_results.pitch(),
                               cuda_results.get(), shape.x, shape, stream);
            Math::divideByValue(data.get(), value, expected.get(), elements);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("values") {
            CUDA::Math::divideByValue(d_data.get(), d_data.pitch(), d_values.get(),
                                      d_results.get(), d_results.pitch(), shape, batches, stream);
            CUDA::Memory::copy(d_results.get(), d_results.pitch(),
                               cuda_results.get(), shape.x, shape_batch, stream);
            Math::divideByValue(data.get(), values.get(), expected.get(), elements, batches);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("array") {
            CUDA::Math::divideByArray(d_data.get(), d_data.pitch(), d_array.get(), d_array.pitch(),
                                      d_results.get(), d_array.pitch(), shape, batches, stream);
            CUDA::Memory::copy(d_results.get(), d_results.pitch(),
                               cuda_results.get(), shape.x, shape_batch, stream);
            Math::divideByArray(data.get(), array.get(), expected.get(), elements, batches);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }
    }

    AND_GIVEN("add") {
        AND_THEN("value") {
            CUDA::Math::addValue(d_data.get(), d_data.pitch(), value,
                                 d_results.get(), d_results.pitch(), shape, stream);
            CUDA::Memory::copy(d_results.get(), d_results.pitch(),
                               cuda_results.get(), shape.x, shape, stream);
            Math::addValue(data.get(), value, expected.get(), elements);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("values") {
            CUDA::Math::addValue(d_data.get(), d_data.pitch(), d_values.get(),
                                 d_results.get(), d_results.pitch(), shape, batches, stream);
            CUDA::Memory::copy(d_results.get(), d_results.pitch(),
                               cuda_results.get(), shape.x, shape_batch, stream);
            Math::addValue(data.get(), values.get(), expected.get(), elements, batches);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("array") {
            CUDA::Math::addArray(d_data.get(), d_data.pitch(), d_array.get(), d_array.pitch(),
                                 d_results.get(), d_array.pitch(), shape, batches, stream);
            CUDA::Memory::copy(d_results.get(), d_results.pitch(),
                               cuda_results.get(), shape.x, shape_batch, stream);
            Math::addArray(data.get(), array.get(), expected.get(), elements, batches);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }
    }

    AND_GIVEN("subtract") {
        AND_THEN("value") {
            CUDA::Math::subtractValue(d_data.get(), d_data.pitch(), value,
                                      d_results.get(), d_results.pitch(), shape, stream);
            CUDA::Memory::copy(d_results.get(), d_results.pitch(),
                               cuda_results.get(), shape.x, shape, stream);
            Math::subtractValue(data.get(), value, expected.get(), elements);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("values") {
            CUDA::Math::subtractValue(d_data.get(), d_data.pitch(), d_values.get(),
                                      d_results.get(), d_results.pitch(), shape, batches, stream);
            CUDA::Memory::copy(d_results.get(), d_results.pitch(),
                               cuda_results.get(), shape.x, shape_batch, stream);
            Math::subtractValue(data.get(), values.get(), expected.get(), elements, batches);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("array") {
            CUDA::Math::subtractArray(d_data.get(), d_data.pitch(), d_array.get(), d_array.pitch(),
                                      d_results.get(), d_array.pitch(), shape, batches, stream);
            CUDA::Memory::copy(d_results.get(), d_results.pitch(),
                               cuda_results.get(), shape.x, shape_batch, stream);
            Math::subtractArray(data.get(), array.get(), expected.get(), elements, batches);
            CUDA::Stream::synchronize(stream);
            TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
        }
    }
}
