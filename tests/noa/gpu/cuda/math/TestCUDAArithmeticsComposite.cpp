#include <noa/gpu/cuda/math/ArithmeticsComposite.h>

#include <noa/cpu/math/ArithmeticsComposite.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace Noa;

TEMPLATE_TEST_CASE("CUDA: ArithmeticsComposite: contiguous", "[noa][cuda][math]",
                   int, uint, float, double) {
    Test::Randomizer<TestType> randomizer(1., 10.);

    size_t elements = Test::IntRandomizer<size_t>(1, 16384).get();
    uint batches = Test::IntRandomizer<uint>(1, 5).get();

    AND_THEN("multiplyAddArray") {
        Memory::PtrHost<TestType> data(elements * batches);
        Memory::PtrHost<TestType> expected(elements * batches);
        Memory::PtrHost<TestType> multipliers(elements);
        Memory::PtrHost<TestType> addends(elements);

        Test::initDataRandom(data.get(), elements * batches, randomizer);
        Test::initDataZero(expected.get(), elements * batches);
        Test::initDataRandom(multipliers.get(), elements, randomizer);
        Test::initDataRandom(addends.get(), elements, randomizer);

        CUDA::Memory::PtrDevice<TestType> d_data(elements * batches);
        CUDA::Memory::PtrDevice<TestType> d_results(elements * batches);
        CUDA::Memory::PtrDevice<TestType> d_multipliers(elements);
        CUDA::Memory::PtrDevice<TestType> d_addends(elements);
        Memory::PtrHost<TestType> cuda_results(elements * batches);

        CUDA::Memory::copy(data.get(), d_data.get(), data.bytes());
        CUDA::Memory::copy(expected.get(), d_results.get(), expected.bytes());
        CUDA::Memory::copy(multipliers.get(), d_multipliers.get(), multipliers.bytes());
        CUDA::Memory::copy(addends.get(), d_addends.get(), addends.bytes());

        CUDA::Stream stream(CUDA::Stream::SERIAL);
        CUDA::Math::multiplyAddArray(d_data.get(), d_multipliers.get(), d_addends.get(), d_results.get(),
                                     elements, batches, stream);
        CUDA::Memory::copy(d_results.get(), cuda_results.get(), d_results.bytes(), stream);
        Math::multiplyAddArray(data.get(), multipliers.get(), addends.get(), expected.get(), elements, batches);
        CUDA::Stream::synchronize(stream);

        TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
    }

    AND_THEN("squaredDistanceFromValue") {
        Memory::PtrHost<TestType> data(elements * batches);
        Memory::PtrHost<TestType> expected(elements * batches);
        Memory::PtrHost<TestType> values(batches);

        Test::initDataRandom(data.get(), elements * batches, randomizer);
        Test::initDataZero(expected.get(), elements * batches);
        Test::initDataRandom(values.get(), batches, randomizer);

        CUDA::Memory::PtrDevice<TestType> d_data(elements * batches);
        CUDA::Memory::PtrDevice<TestType> d_results(elements * batches);
        CUDA::Memory::PtrDevice<TestType> d_values(batches);
        Memory::PtrHost<TestType> cuda_results(elements * batches);

        CUDA::Memory::copy(data.get(), d_data.get(), data.bytes());
        CUDA::Memory::copy(expected.get(), d_results.get(), expected.bytes());
        CUDA::Memory::copy(values.get(), d_values.get(), values.bytes());

        CUDA::Stream stream(CUDA::Stream::SERIAL);
        CUDA::Math::squaredDistanceFromValue(d_data.get(), d_values.get(), d_results.get(),
                                     elements, batches, stream);
        CUDA::Memory::copy(d_results.get(), cuda_results.get(), d_results.bytes(), stream);
        Math::squaredDistanceFromValue(data.get(), values.get(), expected.get(), elements, batches);
        CUDA::Stream::synchronize(stream);

        TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
    }

    AND_THEN("squaredDistanceFromArray") {
        Memory::PtrHost<TestType> data(elements * batches);
        Memory::PtrHost<TestType> expected(elements * batches);
        Memory::PtrHost<TestType> array(elements);

        Test::initDataRandom(data.get(), elements * batches, randomizer);
        Test::initDataZero(expected.get(), elements * batches);
        Test::initDataRandom(array.get(), elements, randomizer);

        CUDA::Memory::PtrDevice<TestType> d_data(elements * batches);
        CUDA::Memory::PtrDevice<TestType> d_results(elements * batches);
        CUDA::Memory::PtrDevice<TestType> d_array(elements);
        Memory::PtrHost<TestType> cuda_results(elements * batches);

        CUDA::Memory::copy(data.get(), d_data.get(), data.bytes());
        CUDA::Memory::copy(expected.get(), d_results.get(), expected.bytes());
        CUDA::Memory::copy(array.get(), d_array.get(), array.bytes());

        CUDA::Stream stream(CUDA::Stream::SERIAL);
        CUDA::Math::squaredDistanceFromArray(d_data.get(), d_array.get(), d_results.get(),
                                             elements, batches, stream);
        CUDA::Memory::copy(d_results.get(), cuda_results.get(), d_results.bytes(), stream);
        Math::squaredDistanceFromArray(data.get(), array.get(), expected.get(), elements, batches);
        CUDA::Stream::synchronize(stream);

        TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
    }
}

TEMPLATE_TEST_CASE("CUDA: ArithmeticsComposite: padded", "[noa][cuda][math]",
                   int, uint, float, double) {
    Test::Randomizer<TestType> randomizer(1., 10.);

    size3_t shape = Test::getRandomShape(3);
    size_t elements = getElements(shape);
    uint batches = Test::IntRandomizer<uint>(1, 5).get();
    size3_t shape_batched(shape.x, shape.y * shape.z, batches);
    size_t pitch_bytes = shape.x * sizeof(TestType);

    AND_THEN("multiplyAddArray") {
        Memory::PtrHost<TestType> data(elements * batches);
        Memory::PtrHost<TestType> expected(elements * batches);
        Memory::PtrHost<TestType> multipliers(elements);
        Memory::PtrHost<TestType> addends(elements);

        Test::initDataRandom(data.get(), elements * batches, randomizer);
        Test::initDataZero(expected.get(), elements * batches);
        Test::initDataRandom(multipliers.get(), elements, randomizer);
        Test::initDataRandom(addends.get(), elements, randomizer);

        // The API allows all inputs to have their own pitch. This is just an example...
        CUDA::Memory::PtrDevicePadded<TestType> d_data(shape_batched);
        CUDA::Memory::PtrDevice<TestType> d_results(elements * batches);
        CUDA::Memory::PtrDevicePadded<TestType> d_multipliers(shape);
        CUDA::Memory::PtrDevice<TestType> d_addends(elements);
        Memory::PtrHost<TestType> cuda_results(elements * batches);

        CUDA::Memory::copy(data.get(), pitch_bytes, d_data.get(), d_data.pitch(), shape_batched);
        CUDA::Memory::copy(expected.get(), d_results.get(), expected.bytes());
        CUDA::Memory::copy(multipliers.get(), pitch_bytes,
                           d_multipliers.get(), d_multipliers.pitch(), shape);
        CUDA::Memory::copy(addends.get(), d_addends.get(), addends.bytes());

        CUDA::Stream stream(CUDA::Stream::SERIAL);
        CUDA::Math::multiplyAddArray(d_data.get(), d_data.pitchElements(),
                                     d_multipliers.get(), d_multipliers.pitchElements(),
                                     d_addends.get(), shape.x,
                                     d_results.get(), shape.x,
                                     shape, batches, stream);
        CUDA::Memory::copy(d_results.get(), cuda_results.get(), d_results.bytes(), stream);
        Math::multiplyAddArray(data.get(), multipliers.get(), addends.get(), expected.get(), elements, batches);
        CUDA::Stream::synchronize(stream);

        TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
    }

    AND_THEN("squaredDistanceFromValue") {
        Memory::PtrHost<TestType> data(elements * batches);
        Memory::PtrHost<TestType> expected(elements * batches);
        Memory::PtrHost<TestType> values(batches);

        Test::initDataRandom(data.get(), elements * batches, randomizer);
        Test::initDataZero(expected.get(), elements * batches);
        Test::initDataRandom(values.get(), batches, randomizer);

        CUDA::Memory::PtrDevicePadded<TestType> d_data(shape_batched);
        CUDA::Memory::PtrDevicePadded<TestType> d_results(shape_batched);
        CUDA::Memory::PtrDevice<TestType> d_values(batches);
        Memory::PtrHost<TestType> cuda_results(elements * batches);

        CUDA::Memory::copy(data.get(), pitch_bytes, d_data.get(), d_data.pitch(), shape_batched);
        CUDA::Memory::copy(expected.get(), pitch_bytes, d_results.get(), d_results.pitch(), shape_batched);
        CUDA::Memory::copy(values.get(), d_values.get(), values.bytes());

        CUDA::Stream stream(CUDA::Stream::SERIAL);
        CUDA::Math::squaredDistanceFromValue(d_data.get(), d_data.pitchElements(), d_values.get(),
                                             d_results.get(), d_results.pitchElements(),
                                             shape, batches, stream);
        CUDA::Memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), pitch_bytes, shape_batched, stream);
        Math::squaredDistanceFromValue(data.get(), values.get(), expected.get(), elements, batches);
        CUDA::Stream::synchronize(stream);

        TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
    }

    AND_THEN("squaredDistanceFromArray") {
        Memory::PtrHost<TestType> data(elements * batches);
        Memory::PtrHost<TestType> expected(elements * batches);
        Memory::PtrHost<TestType> array(elements);

        Test::initDataRandom(data.get(), elements * batches, randomizer);
        Test::initDataZero(expected.get(), elements * batches);
        Test::initDataRandom(array.get(), elements, randomizer);

        CUDA::Memory::PtrDevicePadded<TestType> d_data(shape_batched);
        CUDA::Memory::PtrDevicePadded<TestType> d_results(shape_batched);
        CUDA::Memory::PtrDevicePadded<TestType> d_array(shape);
        Memory::PtrHost<TestType> cuda_results(elements * batches);

        CUDA::Memory::copy(data.get(), pitch_bytes, d_data.get(), d_data.pitch(), shape_batched);
        CUDA::Memory::copy(expected.get(), pitch_bytes, d_results.get(), d_results.pitch(), shape_batched);
        CUDA::Memory::copy(array.get(), pitch_bytes, d_array.get(), d_array.pitch(), shape);

        CUDA::Stream stream(CUDA::Stream::SERIAL);
        CUDA::Math::squaredDistanceFromArray(d_data.get(), d_data.pitchElements(),
                                             d_array.get(), d_array.pitchElements(),
                                             d_results.get(), d_results.pitchElements(),
                                             shape, batches, stream);
        CUDA::Memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), pitch_bytes, shape_batched, stream);
        Math::squaredDistanceFromArray(data.get(), array.get(), expected.get(), elements, batches);
        CUDA::Stream::synchronize(stream);

        TestType diff = Test::getAverageDifference(expected.get(), cuda_results.get(), elements * batches);
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-5));
    }
}
