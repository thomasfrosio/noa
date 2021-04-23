#include <noa/gpu/cuda/masks/Cylinder.h>
#include <noa/cpu/masks/Cylinder.h>

#include <noa/cpu/PtrHost.h>
#include <noa/gpu/cuda/PtrDevice.h>
#include <noa/gpu/cuda/PtrDevicePadded.h>
#include <noa/gpu/cuda/Memory.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace Noa;

TEMPLATE_TEST_CASE("CUDA::Mask - cylinder - contiguous", "[noa][cuda][masks]", float, double) {
    Test::Randomizer<TestType> randomizer(-5, 5);

    uint ndim = GENERATE(2U, 3U);
    size3_t shape = Test::getRandomShape(ndim);
    size_t elements = getElements(shape);

    uint batches = Test::IntRandomizer<uint>(1, 3).get();

    PtrHost<TestType> h_mask(elements);
    PtrHost<TestType> h_data(elements * batches);

    CUDA::PtrDevice<TestType> d_mask(elements);
    CUDA::PtrDevice<TestType> d_data(elements * batches);
    PtrHost<TestType> h_cuda_mask(elements);
    PtrHost<TestType> h_cuda_data(elements * batches);

    CUDA::Stream stream(CUDA::Stream::SERIAL);

    // Sphere parameters:
    Test::RealRandomizer<float> randomizer_float(-1.f, 1.f);
    float3_t shifts(randomizer_float.get() * 10, randomizer_float.get() * 10, randomizer_float.get() * 10);
    float radius_xy = Test::RealRandomizer<float>(0, 20).get();
    float radius_z = Test::RealRandomizer<float>(0, 20).get();
    float taper = Test::RealRandomizer<float>(0, 20).get();

    AND_THEN("INVERT = false") {
        Test::initDataRandom(h_data.get(), h_data.elements(), randomizer);
        CUDA::Memory::copy(h_data.get(), d_data.get(), h_data.bytes(), stream);

        // Test saving the mask.
        CUDA::Mask::cylinder(d_mask.get(), shape, shifts, radius_xy, radius_z, taper, stream);
        CUDA::Memory::copy(d_mask.get(), h_cuda_mask.get(), d_mask.bytes(), stream);
        Mask::cylinder(h_mask.get(), shape, shifts, radius_xy, radius_z, taper);
        CUDA::Stream::synchronize(stream);
        TestType diff = Test::getAverageDifference(h_mask.get(), h_cuda_mask.get(), elements);
        REQUIRE_THAT(diff, Test::isWithinAbs(float(0.), 1e-6));

        // Test on-the-fly, in-place.
        CUDA::Mask::cylinder(d_data.get(), d_data.get(), shape, shifts, radius_xy, radius_z, taper, batches, stream);
        CUDA::Memory::copy(d_data.get(), h_cuda_data.get(), d_data.bytes(), stream);
        Mask::cylinder(h_data.get(), h_data.get(), shape, shifts, radius_xy, radius_z, taper, batches);
        CUDA::Stream::synchronize(stream);
        diff = Test::getAverageDifference(h_data.get(), h_cuda_data.get(), elements * batches);
        REQUIRE_THAT(diff, Test::isWithinAbs(float(0.), 1e-6));
    }

    AND_THEN("INVERT = true") {
        Test::initDataRandom(h_data.get(), h_data.elements(), randomizer);
        CUDA::Memory::copy(h_data.get(), d_data.get(), h_data.bytes(), stream);

        // Test saving the mask.
        CUDA::Mask::cylinder<true>(d_mask.get(), shape, shifts, radius_xy, radius_z, taper, stream);
        CUDA::Memory::copy(d_mask.get(), h_cuda_mask.get(), d_mask.bytes(), stream);
        Mask::cylinder<true>(h_mask.get(), shape, shifts, radius_xy, radius_z, taper);
        CUDA::Stream::synchronize(stream);
        TestType diff = Test::getAverageDifference(h_mask.get(), h_cuda_mask.get(), elements);
        REQUIRE_THAT(diff, Test::isWithinAbs(float(0.), 1e-6));

        // Test on-the-fly, in-place.
        CUDA::Mask::cylinder<true>(d_data.get(), d_data.get(), shape, shifts, radius_xy, radius_z,
                                   taper, batches, stream);
        CUDA::Memory::copy(d_data.get(), h_cuda_data.get(), d_data.bytes(), stream);
        Mask::cylinder<true>(h_data.get(), h_data.get(), shape, shifts, radius_xy, radius_z, taper, batches);
        CUDA::Stream::synchronize(stream);
        diff = Test::getAverageDifference(h_data.get(), h_cuda_data.get(), elements * batches);
        REQUIRE_THAT(diff, Test::isWithinAbs(float(0.), 1e-6));
    }
}

TEMPLATE_TEST_CASE("CUDA::Mask - cylinder - padded", "[noa][cuda][masks]", float, double) {
    Test::Randomizer<TestType> randomizer(-5, 5);

    uint ndim = GENERATE(2U, 3U);
    size3_t shape = Test::getRandomShape(ndim);
    size_t elements = getElements(shape);
    size_t pitch_bytes = shape.x * sizeof(TestType);

    uint batches = Test::IntRandomizer<uint>(1, 3).get();
    size3_t shape_batched(shape.x, shape.y * shape.z, batches);

    PtrHost<TestType> h_mask(elements);
    PtrHost<TestType> h_data(elements * batches);

    CUDA::PtrDevicePadded<TestType> d_mask(shape);
    CUDA::PtrDevicePadded<TestType> d_data(shape_batched);
    PtrHost<TestType> h_cuda_mask(elements);
    PtrHost<TestType> h_cuda_data(elements * batches);

    CUDA::Stream stream(CUDA::Stream::SERIAL);

    // Sphere parameters:
    Test::RealRandomizer<float> randomizer_float(-1.f, 1.f);
    float3_t shifts(randomizer_float.get() * 10, randomizer_float.get() * 10, randomizer_float.get() * 10);
    float radius_xy = Test::RealRandomizer<float>(0, 20).get();
    float radius_z = Test::RealRandomizer<float>(0, 20).get();
    float taper = Test::RealRandomizer<float>(0, 20).get();

    AND_THEN("INVERT = false") {
        Test::initDataRandom(h_data.get(), h_data.elements(), randomizer);
        CUDA::Memory::copy(h_data.get(), pitch_bytes, d_data.get(), d_data.pitch(), shape_batched, stream);

        // Test saving the mask.
        CUDA::Mask::cylinder(d_mask.get(), d_mask.pitchElements(), shape, shifts, radius_xy, radius_z, taper, stream);
        CUDA::Memory::copy(d_mask.get(), d_mask.pitch(), h_cuda_mask.get(), pitch_bytes, shape, stream);
        Mask::cylinder(h_mask.get(), shape, shifts, radius_xy, radius_z, taper);
        CUDA::Stream::synchronize(stream);
        TestType diff = Test::getAverageDifference(h_mask.get(), h_cuda_mask.get(), elements);
        REQUIRE_THAT(diff, Test::isWithinAbs(float(0.), 1e-6));

        // Test on-the-fly, in-place.
        CUDA::Mask::cylinder(d_data.get(), d_data.pitchElements(), d_data.get(), d_data.pitchElements(), shape,
                             shifts, radius_xy, radius_z, taper, batches, stream);
        CUDA::Memory::copy(d_data.get(), d_data.pitch(), h_cuda_data.get(), pitch_bytes, shape_batched, stream);
        Mask::cylinder(h_data.get(), h_data.get(), shape, shifts, radius_xy, radius_z, taper, batches);
        CUDA::Stream::synchronize(stream);
        diff = Test::getAverageDifference(h_data.get(), h_cuda_data.get(), elements * batches);
        REQUIRE_THAT(diff, Test::isWithinAbs(float(0.), 1e-6));
    }

    AND_THEN("INVERT = true") {
        Test::initDataRandom(h_data.get(), h_data.elements(), randomizer);
        CUDA::Memory::copy(h_data.get(), pitch_bytes, d_data.get(), d_data.pitch(), shape_batched, stream);

        // Test saving the mask.
        CUDA::Mask::cylinder<true>(d_mask.get(), d_mask.pitchElements(), shape, shifts, radius_xy, radius_z,
                                   taper, stream);
        CUDA::Memory::copy(d_mask.get(), d_mask.pitch(), h_cuda_mask.get(), pitch_bytes, shape, stream);
        Mask::cylinder<true>(h_mask.get(), shape, shifts, radius_xy, radius_z, taper);
        CUDA::Stream::synchronize(stream);
        TestType diff = Test::getAverageDifference(h_mask.get(), h_cuda_mask.get(), elements);
        REQUIRE_THAT(diff, Test::isWithinAbs(float(0.), 1e-6));

        // Test on-the-fly, in-place.
        CUDA::Mask::cylinder<true>(d_data.get(), d_data.pitchElements(), d_data.get(), d_data.pitchElements(), shape,
                                   shifts, radius_xy, radius_z, taper, batches, stream);
        CUDA::Memory::copy(d_data.get(), d_data.pitch(), h_cuda_data.get(), pitch_bytes, shape_batched, stream);
        Mask::cylinder<true>(h_data.get(), h_data.get(), shape, shifts, radius_xy, radius_z, taper, batches);
        CUDA::Stream::synchronize(stream);
        diff = Test::getAverageDifference(h_data.get(), h_cuda_data.get(), elements * batches);
        REQUIRE_THAT(diff, Test::isWithinAbs(float(0.), 1e-6));
    }
}
