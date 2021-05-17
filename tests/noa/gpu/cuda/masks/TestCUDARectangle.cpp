#include <noa/gpu/cuda/masks/Rectangle.h>
#include <noa/cpu/masks/Rectangle.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace Noa;

TEMPLATE_TEST_CASE("CUDA::Mask - rectangle - contiguous", "[noa][cuda][masks]", float, double) {
    Test::Randomizer<TestType> randomizer(-5, 5);

    uint ndim = GENERATE(2U, 3U);
    size3_t shape = Test::getRandomShape(ndim);
    size_t elements = getElements(shape);

    uint batches = Test::IntRandomizer<uint>(1, 3).get();

    Memory::PtrHost<TestType> h_mask(elements);
    Memory::PtrHost<TestType> h_data(elements * batches);

    CUDA::Memory::PtrDevice<TestType> d_mask(elements);
    CUDA::Memory::PtrDevice<TestType> d_data(elements * batches);
    Memory::PtrHost<TestType> h_cuda_mask(elements);
    Memory::PtrHost<TestType> h_cuda_data(elements * batches);

    CUDA::Stream stream(CUDA::STREAM_SERIAL);

    // Sphere parameters:
    Test::RealRandomizer<float> randomizer_float(-1.f, 1.f);
    Test::RealRandomizer<float> randomizer_radius(1, 30);
    float3_t shifts(randomizer_float.get() * 10, randomizer_float.get() * 10, randomizer_float.get() * 10);
    float3_t radius(randomizer_radius.get(), randomizer_radius.get(), randomizer_radius.get());
    float taper = Test::RealRandomizer<float>(0, 20).get();

    AND_THEN("INVERT = false") {
        Test::initDataRandom(h_data.get(), h_data.elements(), randomizer);
        CUDA::Memory::copy(h_data.get(), d_data.get(), h_data.size(), stream);

        // Test saving the mask.
        CUDA::Mask::rectangle(d_mask.get(), shape, shifts, radius, taper, stream);
        CUDA::Memory::copy(d_mask.get(), h_cuda_mask.get(), d_mask.size(), stream);
        Mask::rectangle(h_mask.get(), shape, shifts, radius, taper);
        CUDA::Stream::synchronize(stream);
        TestType diff = Test::getAverageDifference(h_mask.get(), h_cuda_mask.get(), elements);
        REQUIRE_THAT(diff, Test::isWithinAbs(float(0.), 1e-6));

        // Test on-the-fly, in-place.
        CUDA::Mask::rectangle(d_data.get(), d_data.get(), shape, shifts, radius, taper, batches, stream);
        CUDA::Memory::copy(d_data.get(), h_cuda_data.get(), d_data.size(), stream);
        Mask::rectangle(h_data.get(), h_data.get(), shape, shifts, radius, taper, batches);
        CUDA::Stream::synchronize(stream);
        diff = Test::getAverageDifference(h_data.get(), h_cuda_data.get(), elements * batches);
        REQUIRE_THAT(diff, Test::isWithinAbs(float(0.), 1e-6));
    }

    AND_THEN("INVERT = true") {
        Test::initDataRandom(h_data.get(), h_data.elements(), randomizer);
        CUDA::Memory::copy(h_data.get(), d_data.get(), h_data.size(), stream);

        // Test saving the mask.
        CUDA::Mask::rectangle<true>(d_mask.get(), shape, shifts, radius, taper, stream);
        CUDA::Memory::copy(d_mask.get(), h_cuda_mask.get(), d_mask.size(), stream);
        Mask::rectangle<true>(h_mask.get(), shape, shifts, radius, taper);
        CUDA::Stream::synchronize(stream);
        TestType diff = Test::getAverageDifference(h_mask.get(), h_cuda_mask.get(), elements);
        REQUIRE_THAT(diff, Test::isWithinAbs(float(0.), 1e-6));

        // Test on-the-fly, in-place.
        CUDA::Mask::rectangle<true>(d_data.get(), d_data.get(), shape, shifts, radius, taper, batches, stream);
        CUDA::Memory::copy(d_data.get(), h_cuda_data.get(), d_data.size(), stream);
        Mask::rectangle<true>(h_data.get(), h_data.get(), shape, shifts, radius, taper, batches);
        CUDA::Stream::synchronize(stream);
        diff = Test::getAverageDifference(h_data.get(), h_cuda_data.get(), elements * batches);
        REQUIRE_THAT(diff, Test::isWithinAbs(float(0.), 1e-6));
    }
}

TEMPLATE_TEST_CASE("CUDA::Mask - rectangle - padded", "[noa][cuda][masks]", float, double) {
    Test::Randomizer<TestType> randomizer(-5, 5);

    uint ndim = GENERATE(2U, 3U);
    size3_t shape = Test::getRandomShape(ndim);
    size_t elements = getElements(shape);

    uint batches = Test::IntRandomizer<uint>(1, 3).get();
    size3_t shape_batched(shape.x, shape.y * shape.z, batches);

    Memory::PtrHost<TestType> h_mask(elements);
    Memory::PtrHost<TestType> h_data(elements * batches);

    CUDA::Memory::PtrDevicePadded<TestType> d_mask(shape);
    CUDA::Memory::PtrDevicePadded<TestType> d_data(shape_batched);
    Memory::PtrHost<TestType> h_cuda_mask(elements);
    Memory::PtrHost<TestType> h_cuda_data(elements * batches);

    CUDA::Stream stream(CUDA::STREAM_SERIAL);

    // Sphere parameters:
    Test::RealRandomizer<float> randomizer_float(-1.f, 1.f);
    Test::RealRandomizer<float> randomizer_radius(1, 30);
    float3_t shifts(randomizer_float.get() * 10, randomizer_float.get() * 10, randomizer_float.get() * 10);
    float3_t radius(randomizer_radius.get(), randomizer_radius.get(), randomizer_radius.get());
    float taper = Test::RealRandomizer<float>(0, 20).get();

    AND_THEN("INVERT = false") {
        Test::initDataRandom(h_data.get(), h_data.elements(), randomizer);
        CUDA::Memory::copy(h_data.get(), shape.x, d_data.get(), d_data.pitch(), shape_batched, stream);

        // Test saving the mask.
        CUDA::Mask::rectangle(d_mask.get(), d_mask.pitch(), shape, shifts, radius, taper, stream);
        CUDA::Memory::copy(d_mask.get(), d_mask.pitch(), h_cuda_mask.get(), shape.x, shape, stream);
        Mask::rectangle(h_mask.get(), shape, shifts, radius, taper);
        CUDA::Stream::synchronize(stream);
        TestType diff = Test::getAverageDifference(h_mask.get(), h_cuda_mask.get(), elements);
        REQUIRE_THAT(diff, Test::isWithinAbs(float(0.), 1e-6));

        // Test on-the-fly, in-place.
        CUDA::Mask::rectangle(d_data.get(), d_data.pitch(), d_data.get(), d_data.pitch(), shape,
                           shifts, radius, taper, batches, stream);
        CUDA::Memory::copy(d_data.get(), d_data.pitch(), h_cuda_data.get(), shape.x, shape_batched, stream);
        Mask::rectangle(h_data.get(), h_data.get(), shape, shifts, radius, taper, batches);
        CUDA::Stream::synchronize(stream);
        diff = Test::getAverageDifference(h_data.get(), h_cuda_data.get(), elements * batches);
        REQUIRE_THAT(diff, Test::isWithinAbs(float(0.), 1e-6));
    }

    AND_THEN("INVERT = true") {
        Test::initDataRandom(h_data.get(), h_data.elements(), randomizer);
        CUDA::Memory::copy(h_data.get(), shape.x, d_data.get(), d_data.pitch(), shape_batched, stream);

        // Test saving the mask.
        CUDA::Mask::rectangle<true>(d_mask.get(), d_mask.pitch(), shape, shifts, radius, taper, stream);
        CUDA::Memory::copy(d_mask.get(), d_mask.pitch(), h_cuda_mask.get(), shape.x, shape, stream);
        Mask::rectangle<true>(h_mask.get(), shape, shifts, radius, taper);
        CUDA::Stream::synchronize(stream);
        TestType diff = Test::getAverageDifference(h_mask.get(), h_cuda_mask.get(), elements);
        REQUIRE_THAT(diff, Test::isWithinAbs(float(0.), 1e-6));

        // Test on-the-fly, in-place.
        CUDA::Mask::rectangle<true>(d_data.get(), d_data.pitch(), d_data.get(), d_data.pitch(), shape,
                                 shifts, radius, taper, batches, stream);
        CUDA::Memory::copy(d_data.get(), d_data.pitch(), h_cuda_data.get(), shape.x, shape_batched, stream);
        Mask::rectangle<true>(h_data.get(), h_data.get(), shape, shifts, radius, taper, batches);
        CUDA::Stream::synchronize(stream);
        diff = Test::getAverageDifference(h_data.get(), h_cuda_data.get(), elements * batches);
        REQUIRE_THAT(diff, Test::isWithinAbs(float(0.), 1e-6));
    }
}
