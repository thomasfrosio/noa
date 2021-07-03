#include <noa/gpu/cuda/math/ArithmeticsComposite.h>

#include <noa/cpu/math/ArithmeticsComposite.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("cuda::math:: arithmeticsComposite, contiguous", "[noa][cuda][math]",
                   int, uint, float, double) {
    test::Randomizer<TestType> randomizer(1., 10.);

    size_t elements = test::IntRandomizer<size_t>(1, 16384).get();
    uint batches = test::IntRandomizer<uint>(1, 5).get();

    AND_THEN("multiplyAddArray") {
        memory::PtrHost<TestType> data(elements * batches);
        memory::PtrHost<TestType> expected(elements * batches);
        memory::PtrHost<TestType> multipliers(elements);
        memory::PtrHost<TestType> addends(elements);

        test::initDataRandom(data.get(), elements * batches, randomizer);
        test::initDataZero(expected.get(), elements * batches);
        test::initDataRandom(multipliers.get(), elements, randomizer);
        test::initDataRandom(addends.get(), elements, randomizer);

        cuda::memory::PtrDevice<TestType> d_data(elements * batches);
        cuda::memory::PtrDevice<TestType> d_results(elements * batches);
        cuda::memory::PtrDevice<TestType> d_multipliers(elements);
        cuda::memory::PtrDevice<TestType> d_addends(elements);
        memory::PtrHost<TestType> cuda_results(elements * batches);

        cuda::memory::copy(data.get(), d_data.get(), data.size());
        cuda::memory::copy(expected.get(), d_results.get(), expected.size());
        cuda::memory::copy(multipliers.get(), d_multipliers.get(), multipliers.size());
        cuda::memory::copy(addends.get(), d_addends.get(), addends.size());

        cuda::Stream stream(cuda::Stream::SERIAL);
        cuda::math::multiplyAddArray(d_data.get(), d_multipliers.get(), d_addends.get(), d_results.get(),
                                     elements, batches, stream);
        cuda::memory::copy(d_results.get(), cuda_results.get(), d_results.size(), stream);
        math::multiplyAddArray(data.get(), multipliers.get(), addends.get(), expected.get(), elements, batches);
        cuda::Stream::synchronize(stream);

        TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
    }

    AND_THEN("squaredDistanceFromValue") {
        memory::PtrHost<TestType> data(elements * batches);
        memory::PtrHost<TestType> expected(elements * batches);
        memory::PtrHost<TestType> values(batches);

        test::initDataRandom(data.get(), elements * batches, randomizer);
        test::initDataZero(expected.get(), elements * batches);
        test::initDataRandom(values.get(), batches, randomizer);

        cuda::memory::PtrDevice<TestType> d_data(elements * batches);
        cuda::memory::PtrDevice<TestType> d_results(elements * batches);
        cuda::memory::PtrDevice<TestType> d_values(batches);
        memory::PtrHost<TestType> cuda_results(elements * batches);

        cuda::memory::copy(data.get(), d_data.get(), data.size());
        cuda::memory::copy(expected.get(), d_results.get(), expected.size());
        cuda::memory::copy(values.get(), d_values.get(), values.size());

        cuda::Stream stream(cuda::Stream::SERIAL);
        cuda::math::squaredDistanceFromValue(d_data.get(), d_values.get(), d_results.get(),
                                     elements, batches, stream);
        cuda::memory::copy(d_results.get(), cuda_results.get(), d_results.size(), stream);
        math::squaredDistanceFromValue(data.get(), values.get(), expected.get(), elements, batches);
        cuda::Stream::synchronize(stream);

        TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
    }

    AND_THEN("squaredDistanceFromArray") {
        memory::PtrHost<TestType> data(elements * batches);
        memory::PtrHost<TestType> expected(elements * batches);
        memory::PtrHost<TestType> array(elements);

        test::initDataRandom(data.get(), elements * batches, randomizer);
        test::initDataZero(expected.get(), elements * batches);
        test::initDataRandom(array.get(), elements, randomizer);

        cuda::memory::PtrDevice<TestType> d_data(elements * batches);
        cuda::memory::PtrDevice<TestType> d_results(elements * batches);
        cuda::memory::PtrDevice<TestType> d_array(elements);
        memory::PtrHost<TestType> cuda_results(elements * batches);

        cuda::memory::copy(data.get(), d_data.get(), data.size());
        cuda::memory::copy(expected.get(), d_results.get(), expected.size());
        cuda::memory::copy(array.get(), d_array.get(), array.size());

        cuda::Stream stream(cuda::Stream::SERIAL);
        cuda::math::squaredDistanceFromArray(d_data.get(), d_array.get(), d_results.get(),
                                             elements, batches, stream);
        cuda::memory::copy(d_results.get(), cuda_results.get(), d_results.size(), stream);
        math::squaredDistanceFromArray(data.get(), array.get(), expected.get(), elements, batches);
        cuda::Stream::synchronize(stream);

        TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
    }
}

TEMPLATE_TEST_CASE("cuda::math:: arithmeticsComposite, padded", "[noa][cuda][math]",
                   int, uint, float, double) {
    test::Randomizer<TestType> randomizer(1., 10.);

    size3_t shape = test::getRandomShape(3);
    size_t elements = getElements(shape);
    uint batches = test::IntRandomizer<uint>(1, 5).get();
    size3_t shape_batched(shape.x, shape.y * shape.z, batches);

    AND_THEN("multiplyAddArray") {
        memory::PtrHost<TestType> data(elements * batches);
        memory::PtrHost<TestType> expected(elements * batches);
        memory::PtrHost<TestType> multipliers(elements);
        memory::PtrHost<TestType> addends(elements);

        test::initDataRandom(data.get(), elements * batches, randomizer);
        test::initDataZero(expected.get(), elements * batches);
        test::initDataRandom(multipliers.get(), elements, randomizer);
        test::initDataRandom(addends.get(), elements, randomizer);

        // The API allows all inputs to have their own pitch. This is just an example...
        cuda::memory::PtrDevicePadded<TestType> d_data(shape_batched);
        cuda::memory::PtrDevice<TestType> d_results(elements * batches);
        cuda::memory::PtrDevicePadded<TestType> d_multipliers(shape);
        cuda::memory::PtrDevice<TestType> d_addends(elements);
        memory::PtrHost<TestType> cuda_results(elements * batches);

        cuda::memory::copy(data.get(), shape.x, d_data.get(), d_data.pitch(), shape_batched);
        cuda::memory::copy(expected.get(), d_results.get(), expected.size());
        cuda::memory::copy(multipliers.get(), shape.x,
                           d_multipliers.get(), d_multipliers.pitch(), shape);
        cuda::memory::copy(addends.get(), d_addends.get(), addends.size());

        cuda::Stream stream(cuda::Stream::SERIAL);
        cuda::math::multiplyAddArray(d_data.get(), d_data.pitch(),
                                     d_multipliers.get(), d_multipliers.pitch(),
                                     d_addends.get(), shape.x,
                                     d_results.get(), shape.x,
                                     shape, batches, stream);
        cuda::memory::copy(d_results.get(), cuda_results.get(), d_results.size(), stream);
        math::multiplyAddArray(data.get(), multipliers.get(), addends.get(), expected.get(), elements, batches);
        cuda::Stream::synchronize(stream);

        TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
    }

    AND_THEN("squaredDistanceFromValue") {
        memory::PtrHost<TestType> data(elements * batches);
        memory::PtrHost<TestType> expected(elements * batches);
        memory::PtrHost<TestType> values(batches);

        test::initDataRandom(data.get(), elements * batches, randomizer);
        test::initDataZero(expected.get(), elements * batches);
        test::initDataRandom(values.get(), batches, randomizer);

        cuda::memory::PtrDevicePadded<TestType> d_data(shape_batched);
        cuda::memory::PtrDevicePadded<TestType> d_results(shape_batched);
        cuda::memory::PtrDevice<TestType> d_values(batches);
        memory::PtrHost<TestType> cuda_results(elements * batches);

        cuda::memory::copy(data.get(), shape.x, d_data.get(), d_data.pitch(), shape_batched);
        cuda::memory::copy(expected.get(), shape.x, d_results.get(), d_results.pitch(), shape_batched);
        cuda::memory::copy(values.get(), d_values.get(), values.size());

        cuda::Stream stream(cuda::Stream::SERIAL);
        cuda::math::squaredDistanceFromValue(d_data.get(), d_data.pitch(), d_values.get(),
                                             d_results.get(), d_results.pitch(),
                                             shape, batches, stream);
        cuda::memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape_batched, stream);
        math::squaredDistanceFromValue(data.get(), values.get(), expected.get(), elements, batches);
        cuda::Stream::synchronize(stream);

        TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
    }

    AND_THEN("squaredDistanceFromArray") {
        memory::PtrHost<TestType> data(elements * batches);
        memory::PtrHost<TestType> expected(elements * batches);
        memory::PtrHost<TestType> array(elements);

        test::initDataRandom(data.get(), elements * batches, randomizer);
        test::initDataZero(expected.get(), elements * batches);
        test::initDataRandom(array.get(), elements, randomizer);

        cuda::memory::PtrDevicePadded<TestType> d_data(shape_batched);
        cuda::memory::PtrDevicePadded<TestType> d_results(shape_batched);
        cuda::memory::PtrDevicePadded<TestType> d_array(shape);
        memory::PtrHost<TestType> cuda_results(elements * batches);

        cuda::memory::copy(data.get(), shape.x, d_data.get(), d_data.pitch(), shape_batched);
        cuda::memory::copy(expected.get(), shape.x, d_results.get(), d_results.pitch(), shape_batched);
        cuda::memory::copy(array.get(), shape.x, d_array.get(), d_array.pitch(), shape);

        cuda::Stream stream(cuda::Stream::SERIAL);
        cuda::math::squaredDistanceFromArray(d_data.get(), d_data.pitch(),
                                             d_array.get(), d_array.pitch(),
                                             d_results.get(), d_results.pitch(),
                                             shape, batches, stream);
        cuda::memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape_batched, stream);
        math::squaredDistanceFromArray(data.get(), array.get(), expected.get(), elements, batches);
        cuda::Stream::synchronize(stream);

        TestType diff = test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-5));
    }
}
