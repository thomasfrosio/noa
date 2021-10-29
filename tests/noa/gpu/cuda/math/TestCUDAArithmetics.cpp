#include <noa/gpu/cuda/math/Arithmetics.h>

#include <noa/cpu/math/Arithmetics.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("cuda::math:: arithmetics, contiguous", "[noa][cuda][math]",
                   int, uint, float, double, cfloat_t, cdouble_t) {
    test::Randomizer<TestType> randomizer(1., 10.);

    size_t elements = test::IntRandomizer<size_t>(1, 16384).get();
    uint batches = test::IntRandomizer<uint>(1, 5).get();

    cpu::memory::PtrHost<TestType> data(elements * batches);
    cpu::memory::PtrHost<TestType> expected(elements * batches);
    cpu::memory::PtrHost<TestType> values(batches);
    cpu::memory::PtrHost<TestType> array(elements);
    TestType value = randomizer.get();

    cuda::memory::PtrDevice<TestType> d_data(elements * batches);
    cuda::memory::PtrDevice<TestType> d_values(batches);
    cuda::memory::PtrDevice<TestType> d_array(elements);
    cuda::memory::PtrDevice<TestType> d_results(elements * batches);
    cpu::memory::PtrHost<TestType> cuda_results(elements * batches);

    test::initDataRandom(data.get(), data.elements(), randomizer);
    test::initDataZero(expected.get(), expected.elements());
    test::initDataRandom(values.get(), values.elements(), randomizer);
    test::initDataRandom(array.get(), array.elements(), randomizer);
    cuda::Stream stream;

    cuda::memory::copy(data.get(), d_data.get(), elements * batches);
    cuda::memory::copy(expected.get(), d_results.get(), elements * batches);
    cuda::memory::copy(values.get(), d_values.get(), batches);
    cuda::memory::copy(array.get(), d_array.get(), elements);

    AND_GIVEN("multiply") {
        AND_THEN("value") {
            cuda::math::multiplyByValue(d_data.get(), value, d_results.get(), elements, stream);
            cuda::memory::copy(d_results.get(), cuda_results.get(), elements, stream);
            cpu::math::multiplyByValue(data.get(), value, expected.get(), elements);
            cuda::Stream::synchronize(stream);
            TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("values") {
            cuda::math::multiplyByValue(d_data.get(), d_values.get(), d_results.get(), elements, batches, stream);
            cuda::memory::copy(d_results.get(), cuda_results.get(), elements * batches, stream);
            cpu::math::multiplyByValue(data.get(), values.get(), expected.get(), elements, batches);
            cuda::Stream::synchronize(stream);
            TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("array") {
            cuda::math::multiplyByArray(d_data.get(), d_array.get(), d_results.get(), elements, batches, stream);
            cuda::memory::copy(d_results.get(), cuda_results.get(), elements * batches, stream);
            cpu::math::multiplyByArray(data.get(), array.get(), expected.get(), elements, batches);
            cuda::Stream::synchronize(stream);
            TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
        }
    }

    AND_GIVEN("divide") {
        AND_THEN("value") {
            cuda::math::divideByValue(d_data.get(), value, d_results.get(), elements, stream);
            cuda::memory::copy(d_results.get(), cuda_results.get(), elements, stream);
            cpu::math::divideByValue(data.get(), value, expected.get(), elements);
            cuda::Stream::synchronize(stream);
            TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("values") {
            cuda::math::divideByValue(d_data.get(), d_values.get(), d_results.get(), elements, batches, stream);
            cuda::memory::copy(d_results.get(), cuda_results.get(), elements * batches, stream);
            cpu::math::divideByValue(data.get(), values.get(), expected.get(), elements, batches);
            cuda::Stream::synchronize(stream);
            TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("array") {
            cuda::math::divideByArray(d_data.get(), d_array.get(), d_results.get(), elements, batches, stream);
            cuda::memory::copy(d_results.get(), cuda_results.get(), elements * batches, stream);
            cpu::math::divideByArray(data.get(), array.get(), expected.get(), elements, batches);
            cuda::Stream::synchronize(stream);
            TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
        }
    }

    AND_GIVEN("add") {
        AND_THEN("value") {
            cuda::math::addValue(d_data.get(), value, d_results.get(), elements, stream);
            cuda::memory::copy(d_results.get(), cuda_results.get(), elements, stream);
            cpu::math::addValue(data.get(), value, expected.get(), elements);
            cuda::Stream::synchronize(stream);
            TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("values") {
            cuda::math::addValue(d_data.get(), d_values.get(), d_results.get(), elements, batches, stream);
            cuda::memory::copy(d_results.get(), cuda_results.get(), elements * batches, stream);
            cpu::math::addValue(data.get(), values.get(), expected.get(), elements, batches);
            cuda::Stream::synchronize(stream);
            TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("array") {
            cuda::math::addArray(d_data.get(), d_array.get(), d_results.get(), elements, batches, stream);
            cuda::memory::copy(d_results.get(), cuda_results.get(), elements * batches, stream);
            cpu::math::addArray(data.get(), array.get(), expected.get(), elements, batches);
            cuda::Stream::synchronize(stream);
            TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
        }
    }

    AND_GIVEN("subtract") {
        AND_THEN("value") {
            cuda::math::subtractValue(d_data.get(), value, d_results.get(), elements, stream);
            cuda::memory::copy(d_results.get(), cuda_results.get(), elements, stream);
            cpu::math::subtractValue(data.get(), value, expected.get(), elements);
            cuda::Stream::synchronize(stream);
            TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("values") {
            cuda::math::subtractValue(d_data.get(), d_values.get(), d_results.get(), elements, batches, stream);
            cuda::memory::copy(d_results.get(), cuda_results.get(), elements * batches, stream);
            cpu::math::subtractValue(data.get(), values.get(), expected.get(), elements, batches);
            cuda::Stream::synchronize(stream);
            TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("array") {
            cuda::math::subtractArray(d_data.get(), d_array.get(), d_results.get(), elements, batches, stream);
            cuda::memory::copy(d_results.get(), cuda_results.get(), elements * batches, stream);
            cpu::math::subtractArray(data.get(), array.get(), expected.get(), elements, batches);
            cuda::Stream::synchronize(stream);
            TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
        }
    }
}

TEMPLATE_TEST_CASE("cuda::math:: arithmetics: padded", "[noa][cuda][math]",
                   int, uint, float, double, cfloat_t, cdouble_t) {
    test::Randomizer<TestType> randomizer(1., 10.);

    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape = test::getRandomShape(ndim);
    size_t elements = noa::elements(shape);
    uint batches = test::IntRandomizer<uint>(1, 5).get();

    cpu::memory::PtrHost<TestType> data(elements * batches);
    cpu::memory::PtrHost<TestType> expected(elements * batches);
    cpu::memory::PtrHost<TestType> values(batches);
    cpu::memory::PtrHost<TestType> array(elements);
    TestType value = randomizer.get();

    size3_t shape_batch = {shape.x, shape.y, shape.z * batches};
    cuda::memory::PtrDevicePadded<TestType> d_data(shape_batch);
    cuda::memory::PtrDevice<TestType> d_values(batches);
    cuda::memory::PtrDevicePadded<TestType> d_array(shape);
    cuda::memory::PtrDevicePadded<TestType> d_results(shape_batch);
    cpu::memory::PtrHost<TestType> cuda_results(elements * batches);

    test::initDataRandom(data.get(), data.elements(), randomizer);
    test::initDataZero(expected.get(), expected.elements());
    test::initDataRandom(values.get(), values.elements(), randomizer);
    test::initDataRandom(array.get(), array.elements(), randomizer);
    cuda::Stream stream;

    cuda::memory::copy(data.get(), shape.x, d_data.get(), d_data.pitch(), shape_batch);
    cuda::memory::copy(expected.get(), shape.x, d_results.get(), d_results.pitch(), shape_batch);
    cuda::memory::copy(values.get(), d_values.get(), batches);
    cuda::memory::copy(array.get(), shape.x, d_array.get(), d_array.pitch(), shape);

    AND_GIVEN("multiply") {
        AND_THEN("value") {
            cuda::math::multiplyByValue(d_data.get(), d_data.pitch(), value,
                                        d_results.get(), d_results.pitch(), shape, stream);
            cuda::memory::copy(d_results.get(), d_results.pitch(),
                               cuda_results.get(), shape.x, shape, stream);
            cpu::math::multiplyByValue(data.get(), value, expected.get(), elements);
            cuda::Stream::synchronize(stream);
            TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("values") {
            cuda::math::multiplyByValue(d_data.get(), d_data.pitch(), d_values.get(),
                                        d_results.get(), d_results.pitch(), shape, batches, stream);
            cuda::memory::copy(d_results.get(), d_results.pitch(),
                               cuda_results.get(), shape.x, shape_batch, stream);
            cpu::math::multiplyByValue(data.get(), values.get(), expected.get(), elements, batches);
            cuda::Stream::synchronize(stream);
            TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("array") {
            cuda::math::multiplyByArray(d_data.get(), d_data.pitch(), d_array.get(), d_array.pitch(),
                                        d_results.get(), d_array.pitch(), shape, batches, stream);
            cuda::memory::copy(d_results.get(), d_results.pitch(),
                               cuda_results.get(), shape.x, shape_batch, stream);
            cpu::math::multiplyByArray(data.get(), array.get(), expected.get(), elements, batches);
            cuda::Stream::synchronize(stream);
            TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
        }
    }

    AND_GIVEN("divide") {
        AND_THEN("value") {
            cuda::math::divideByValue(d_data.get(), d_data.pitch(), value,
                                      d_results.get(), d_results.pitch(), shape, stream);
            cuda::memory::copy(d_results.get(), d_results.pitch(),
                               cuda_results.get(), shape.x, shape, stream);
            cpu::math::divideByValue(data.get(), value, expected.get(), elements);
            cuda::Stream::synchronize(stream);
            TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("values") {
            cuda::math::divideByValue(d_data.get(), d_data.pitch(), d_values.get(),
                                      d_results.get(), d_results.pitch(), shape, batches, stream);
            cuda::memory::copy(d_results.get(), d_results.pitch(),
                               cuda_results.get(), shape.x, shape_batch, stream);
            cpu::math::divideByValue(data.get(), values.get(), expected.get(), elements, batches);
            cuda::Stream::synchronize(stream);
            TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("array") {
            cuda::math::divideByArray(d_data.get(), d_data.pitch(), d_array.get(), d_array.pitch(),
                                      d_results.get(), d_array.pitch(), shape, batches, stream);
            cuda::memory::copy(d_results.get(), d_results.pitch(),
                               cuda_results.get(), shape.x, shape_batch, stream);
            cpu::math::divideByArray(data.get(), array.get(), expected.get(), elements, batches);
            cuda::Stream::synchronize(stream);
            TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
        }
    }

    AND_GIVEN("add") {
        AND_THEN("value") {
            cuda::math::addValue(d_data.get(), d_data.pitch(), value,
                                 d_results.get(), d_results.pitch(), shape, stream);
            cuda::memory::copy(d_results.get(), d_results.pitch(),
                               cuda_results.get(), shape.x, shape, stream);
            cpu::math::addValue(data.get(), value, expected.get(), elements);
            cuda::Stream::synchronize(stream);
            TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("values") {
            cuda::math::addValue(d_data.get(), d_data.pitch(), d_values.get(),
                                 d_results.get(), d_results.pitch(), shape, batches, stream);
            cuda::memory::copy(d_results.get(), d_results.pitch(),
                               cuda_results.get(), shape.x, shape_batch, stream);
            cpu::math::addValue(data.get(), values.get(), expected.get(), elements, batches);
            cuda::Stream::synchronize(stream);
            TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("array") {
            cuda::math::addArray(d_data.get(), d_data.pitch(), d_array.get(), d_array.pitch(),
                                 d_results.get(), d_array.pitch(), shape, batches, stream);
            cuda::memory::copy(d_results.get(), d_results.pitch(),
                               cuda_results.get(), shape.x, shape_batch, stream);
            cpu::math::addArray(data.get(), array.get(), expected.get(), elements, batches);
            cuda::Stream::synchronize(stream);
            TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
        }
    }

    AND_GIVEN("subtract") {
        AND_THEN("value") {
            cuda::math::subtractValue(d_data.get(), d_data.pitch(), value,
                                      d_results.get(), d_results.pitch(), shape, stream);
            cuda::memory::copy(d_results.get(), d_results.pitch(),
                               cuda_results.get(), shape.x, shape, stream);
            cpu::math::subtractValue(data.get(), value, expected.get(), elements);
            cuda::Stream::synchronize(stream);
            TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("values") {
            cuda::math::subtractValue(d_data.get(), d_data.pitch(), d_values.get(),
                                      d_results.get(), d_results.pitch(), shape, batches, stream);
            cuda::memory::copy(d_results.get(), d_results.pitch(),
                               cuda_results.get(), shape.x, shape_batch, stream);
            cpu::math::subtractValue(data.get(), values.get(), expected.get(), elements, batches);
            cuda::Stream::synchronize(stream);
            TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
        }

        AND_THEN("array") {
            cuda::math::subtractArray(d_data.get(), d_data.pitch(), d_array.get(), d_array.pitch(),
                                      d_results.get(), d_array.pitch(), shape, batches, stream);
            cuda::memory::copy(d_results.get(), d_results.pitch(),
                               cuda_results.get(), shape.x, shape_batch, stream);
            cpu::math::subtractArray(data.get(), array.get(), expected.get(), elements, batches);
            cuda::Stream::synchronize(stream);
            TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
        }
    }
}
