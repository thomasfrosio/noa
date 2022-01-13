#include <noa/cpu/math/Ewise.h>
#include <noa/cpu/memory/PtrHost.h>

#include <noa/gpu/cuda/math/ArithmeticsComposite.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("cuda::math:: arithmeticsComposite, contiguous", "[noa][cuda][math]",
                   int, uint, float, double) {
    test::Randomizer<TestType> randomizer(1., 10.);

    const size3_t shape = test::getRandomShape(3);
    const size_t elements = noa::elements(shape);
    const size_t batches = test::Randomizer<size_t>(1, 5).get();
    const size3_t pitch{shape.x, shape.y, 0};
    cpu::Stream cpu_stream;

    AND_THEN("multiplyAddArray") {
        cpu::memory::PtrHost<TestType> data(elements * batches);
        cpu::memory::PtrHost<TestType> expected(elements * batches);
        cpu::memory::PtrHost<TestType> multipliers(elements);
        cpu::memory::PtrHost<TestType> addends(elements);

        test::randomize(data.get(), elements * batches, randomizer);
        test::memset(expected.get(), elements * batches, 0);
        test::randomize(multipliers.get(), elements, randomizer);
        test::randomize(addends.get(), elements, randomizer);

        cuda::memory::PtrDevice<TestType> d_data(elements * batches);
        cuda::memory::PtrDevice<TestType> d_results(elements * batches);
        cuda::memory::PtrDevice<TestType> d_multipliers(elements);
        cuda::memory::PtrDevice<TestType> d_addends(elements);
        cpu::memory::PtrHost<TestType> cuda_results(elements * batches);

        cuda::memory::copy(data.get(), d_data.get(), data.size());
        cuda::memory::copy(expected.get(), d_results.get(), expected.size());
        cuda::memory::copy(multipliers.get(), d_multipliers.get(), multipliers.size());
        cuda::memory::copy(addends.get(), d_addends.get(), addends.size());

        cuda::Stream stream(cuda::Stream::SERIAL);
        cuda::math::multiplyAddArray(d_data.get(), d_multipliers.get(), d_addends.get(), d_results.get(),
                                     elements, batches, stream);
        cuda::memory::copy(d_results.get(), cuda_results.get(), d_results.size(), stream);
        cpu::math::ewise(data.get(), shape, multipliers.get(), pitch, addends.get(), pitch,
                         expected.get(), shape, shape, batches, math::fma_t{}, cpu_stream);
        stream.synchronize();

        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), cuda_results.get(), elements * batches, 5e-5));
    }

    AND_THEN("squaredDistanceFromValue") {
        cpu::memory::PtrHost<TestType> data(elements * batches);
        cpu::memory::PtrHost<TestType> expected(elements * batches);
        cpu::memory::PtrHost<TestType> values(batches);

        test::randomize(data.get(), elements * batches, randomizer);
        test::memset(expected.get(), elements * batches, 0);
        test::randomize(values.get(), batches, randomizer);

        cuda::memory::PtrDevice<TestType> d_data(elements * batches);
        cuda::memory::PtrDevice<TestType> d_results(elements * batches);
        cuda::memory::PtrDevice<TestType> d_values(batches);
        cpu::memory::PtrHost<TestType> cuda_results(elements * batches);

        cuda::memory::copy(data.get(), d_data.get(), data.size());
        cuda::memory::copy(expected.get(), d_results.get(), expected.size());
        cuda::memory::copy(values.get(), d_values.get(), values.size());

        cuda::Stream stream(cuda::Stream::SERIAL);
        cuda::math::squaredDistanceFromValue(d_data.get(), d_values.get(), d_results.get(),
                                             elements, batches, stream);
        cuda::memory::copy(d_results.get(), cuda_results.get(), d_results.size(), stream);
        cpu::math::ewise(data.get(), shape, values.get(),
                         expected.get(), shape, shape, batches, math::dist2_t{}, cpu_stream);
        stream.synchronize();

        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), cuda_results.get(), elements * batches, 5e-5));
    }

    AND_THEN("squaredDistanceFromArray") {
        cpu::memory::PtrHost<TestType> data(elements * batches);
        cpu::memory::PtrHost<TestType> expected(elements * batches);
        cpu::memory::PtrHost<TestType> array(elements);

        test::randomize(data.get(), elements * batches, randomizer);
        test::memset(expected.get(), elements * batches, 0);
        test::randomize(array.get(), elements, randomizer);

        cuda::memory::PtrDevice<TestType> d_data(elements * batches);
        cuda::memory::PtrDevice<TestType> d_results(elements * batches);
        cuda::memory::PtrDevice<TestType> d_array(elements);
        cpu::memory::PtrHost<TestType> cuda_results(elements * batches);

        cuda::memory::copy(data.get(), d_data.get(), data.size());
        cuda::memory::copy(expected.get(), d_results.get(), expected.size());
        cuda::memory::copy(array.get(), d_array.get(), array.size());

        cuda::Stream stream(cuda::Stream::SERIAL);
        cuda::math::squaredDistanceFromArray(d_data.get(), d_array.get(), d_results.get(),
                                             elements, batches, stream);
        cuda::memory::copy(d_results.get(), cuda_results.get(), d_results.size(), stream);
        cpu::math::ewise(data.get(), shape, array.get(), pitch,
                         expected.get(), shape, shape, batches, math::dist2_t{}, cpu_stream);
        stream.synchronize();

        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), cuda_results.get(), elements * batches, 5e-5));
    }
}

TEMPLATE_TEST_CASE("cuda::math:: arithmeticsComposite, padded", "[noa][cuda][math]",
                   int, uint, float, double) {
    test::Randomizer<TestType> randomizer(1., 10.);

    cpu::Stream cpu_stream;
    size3_t shape = test::getRandomShape(3);
    size_t elements = noa::elements(shape);
    size_t batches = test::Randomizer<size_t>(1, 5).get();
    size3_t shape_batched(shape.x, shape.y * shape.z, batches);
    const size3_t pitch{shape.x, shape.y, 0};

    AND_THEN("multiplyAddArray") {
        cpu::memory::PtrHost<TestType> data(elements * batches);
        cpu::memory::PtrHost<TestType> expected(elements * batches);
        cpu::memory::PtrHost<TestType> multipliers(elements);
        cpu::memory::PtrHost<TestType> addends(elements);

        test::randomize(data.get(), elements * batches, randomizer);
        test::memset(expected.get(), elements * batches, 0);
        test::randomize(multipliers.get(), elements, randomizer);
        test::randomize(addends.get(), elements, randomizer);

        // The API allows all inputs to have their own pitch. This is just an example...
        cuda::memory::PtrDevicePadded<TestType> d_data(shape_batched);
        cuda::memory::PtrDevice<TestType> d_results(elements * batches);
        cuda::memory::PtrDevicePadded<TestType> d_multipliers(shape);
        cuda::memory::PtrDevice<TestType> d_addends(elements);
        cpu::memory::PtrHost<TestType> cuda_results(elements * batches);

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
        cpu::math::ewise(data.get(), shape, multipliers.get(), pitch, addends.get(), pitch,
                         expected.get(), shape, shape, batches, math::fma_t{}, cpu_stream);
        stream.synchronize();

        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), cuda_results.get(), elements * batches, 5e-5));
    }

    AND_THEN("squaredDistanceFromValue") {
        cpu::memory::PtrHost<TestType> data(elements * batches);
        cpu::memory::PtrHost<TestType> expected(elements * batches);
        cpu::memory::PtrHost<TestType> values(batches);

        test::randomize(data.get(), elements * batches, randomizer);
        test::memset(expected.get(), elements * batches, 0);
        test::randomize(values.get(), batches, randomizer);

        cuda::memory::PtrDevicePadded<TestType> d_data(shape_batched);
        cuda::memory::PtrDevicePadded<TestType> d_results(shape_batched);
        cuda::memory::PtrDevice<TestType> d_values(batches);
        cpu::memory::PtrHost<TestType> cuda_results(elements * batches);

        cuda::memory::copy(data.get(), shape.x, d_data.get(), d_data.pitch(), shape_batched);
        cuda::memory::copy(expected.get(), shape.x, d_results.get(), d_results.pitch(), shape_batched);
        cuda::memory::copy(values.get(), d_values.get(), values.size());

        cuda::Stream stream(cuda::Stream::SERIAL);
        cuda::math::squaredDistanceFromValue(d_data.get(), d_data.pitch(), d_values.get(),
                                             d_results.get(), d_results.pitch(),
                                             shape, batches, stream);
        cuda::memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape_batched, stream);
        cpu::math::ewise(data.get(), shape, values.get(),
                         expected.get(), shape, shape, batches, math::dist2_t{}, cpu_stream);
        stream.synchronize();

        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), cuda_results.get(), elements * batches, 5e-5));
    }

    AND_THEN("squaredDistanceFromArray") {
        cpu::memory::PtrHost<TestType> data(elements * batches);
        cpu::memory::PtrHost<TestType> expected(elements * batches);
        cpu::memory::PtrHost<TestType> array(elements);

        test::randomize(data.get(), elements * batches, randomizer);
        test::memset(expected.get(), elements * batches, 0);
        test::randomize(array.get(), elements, randomizer);

        cuda::memory::PtrDevicePadded<TestType> d_data(shape_batched);
        cuda::memory::PtrDevicePadded<TestType> d_results(shape_batched);
        cuda::memory::PtrDevicePadded<TestType> d_array(shape);
        cpu::memory::PtrHost<TestType> cuda_results(elements * batches);

        cuda::memory::copy(data.get(), shape.x, d_data.get(), d_data.pitch(), shape_batched);
        cuda::memory::copy(expected.get(), shape.x, d_results.get(), d_results.pitch(), shape_batched);
        cuda::memory::copy(array.get(), shape.x, d_array.get(), d_array.pitch(), shape);

        cuda::Stream stream(cuda::Stream::SERIAL);
        cuda::math::squaredDistanceFromArray(d_data.get(), d_data.pitch(),
                                             d_array.get(), d_array.pitch(),
                                             d_results.get(), d_results.pitch(),
                                             shape, batches, stream);
        cuda::memory::copy(d_results.get(), d_results.pitch(), cuda_results.get(), shape.x, shape_batched, stream);
        cpu::math::ewise(data.get(), shape, array.get(), pitch,
                         expected.get(), shape, shape, batches, math::dist2_t{}, cpu_stream);
        stream.synchronize();

        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), cuda_results.get(), elements * batches, 5e-5));
    }
}
