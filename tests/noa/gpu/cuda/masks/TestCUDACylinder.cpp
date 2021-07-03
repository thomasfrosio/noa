#include <noa/gpu/cuda/mask/Cylinder.h>
#include <noa/cpu/mask/Cylinder.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("cuda::mask::cylinder(), contiguous", "[noa][cuda][masks]", float, double) {
    test::Randomizer<TestType> randomizer(-5, 5);

    uint ndim = GENERATE(2U, 3U);
    size3_t shape = test::getRandomShape(ndim);
    size_t elements = getElements(shape);

    uint batches = test::IntRandomizer<uint>(1, 3).get();

    memory::PtrHost<TestType> h_mask(elements);
    memory::PtrHost<TestType> h_data(elements * batches);

    cuda::memory::PtrDevice<TestType> d_mask(elements);
    cuda::memory::PtrDevice<TestType> d_data(elements * batches);
    memory::PtrHost<TestType> h_cuda_mask(elements);
    memory::PtrHost<TestType> h_cuda_data(elements * batches);

    cuda::Stream stream(cuda::Stream::SERIAL);

    // Sphere parameters:
    test::RealRandomizer<float> randomizer_float(-1.f, 1.f);
    float3_t shifts(randomizer_float.get() * 10, randomizer_float.get() * 10, randomizer_float.get() * 10);
    float radius_xy = test::RealRandomizer<float>(0, 20).get();
    float radius_z = test::RealRandomizer<float>(0, 20).get();
    float taper = test::RealRandomizer<float>(0, 20).get();

    AND_THEN("INVERT = false") {
        test::initDataRandom(h_data.get(), h_data.elements(), randomizer);
        cuda::memory::copy(h_data.get(), d_data.get(), h_data.size(), stream);

        // Test saving the mask.
        cuda::mask::cylinder(d_mask.get(), shape, shifts, radius_xy, radius_z, taper, stream);
        cuda::memory::copy(d_mask.get(), h_cuda_mask.get(), d_mask.size(), stream);
        mask::cylinder(h_mask.get(), shape, shifts, radius_xy, radius_z, taper);
        cuda::Stream::synchronize(stream);
        TestType diff = test::getAverageDifference(h_mask.get(), h_cuda_mask.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(float(0.), 1e-6));

        // Test on-the-fly, in-place.
        cuda::mask::cylinder(d_data.get(), d_data.get(), shape, shifts, radius_xy, radius_z, taper, batches, stream);
        cuda::memory::copy(d_data.get(), h_cuda_data.get(), d_data.size(), stream);
        mask::cylinder(h_data.get(), h_data.get(), shape, shifts, radius_xy, radius_z, taper, batches);
        cuda::Stream::synchronize(stream);
        diff = test::getAverageDifference(h_data.get(), h_cuda_data.get(), elements * batches);
        REQUIRE_THAT(diff, test::isWithinAbs(float(0.), 1e-6));
    }

    AND_THEN("INVERT = true") {
        test::initDataRandom(h_data.get(), h_data.elements(), randomizer);
        cuda::memory::copy(h_data.get(), d_data.get(), h_data.size(), stream);

        // Test saving the mask.
        cuda::mask::cylinder<true>(d_mask.get(), shape, shifts, radius_xy, radius_z, taper, stream);
        cuda::memory::copy(d_mask.get(), h_cuda_mask.get(), d_mask.size(), stream);
        mask::cylinder<true>(h_mask.get(), shape, shifts, radius_xy, radius_z, taper);
        cuda::Stream::synchronize(stream);
        TestType diff = test::getAverageDifference(h_mask.get(), h_cuda_mask.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(float(0.), 1e-6));

        // Test on-the-fly, in-place.
        cuda::mask::cylinder<true>(d_data.get(), d_data.get(), shape, shifts, radius_xy, radius_z,
                                   taper, batches, stream);
        cuda::memory::copy(d_data.get(), h_cuda_data.get(), d_data.size(), stream);
        mask::cylinder<true>(h_data.get(), h_data.get(), shape, shifts, radius_xy, radius_z, taper, batches);
        cuda::Stream::synchronize(stream);
        diff = test::getAverageDifference(h_data.get(), h_cuda_data.get(), elements * batches);
        REQUIRE_THAT(diff, test::isWithinAbs(float(0.), 1e-6));
    }
}

TEMPLATE_TEST_CASE("cuda::mask::cylinder(), padded", "[noa][cuda][masks]", float, double) {
    test::Randomizer<TestType> randomizer(-5, 5);

    uint ndim = GENERATE(2U, 3U);
    size3_t shape = test::getRandomShape(ndim);
    size_t elements = getElements(shape);

    uint batches = test::IntRandomizer<uint>(1, 3).get();
    size3_t shape_batched(shape.x, shape.y * shape.z, batches);

    memory::PtrHost<TestType> h_mask(elements);
    memory::PtrHost<TestType> h_data(elements * batches);

    cuda::memory::PtrDevicePadded<TestType> d_mask(shape);
    cuda::memory::PtrDevicePadded<TestType> d_data(shape_batched);
    memory::PtrHost<TestType> h_cuda_mask(elements);
    memory::PtrHost<TestType> h_cuda_data(elements * batches);

    cuda::Stream stream(cuda::Stream::SERIAL);

    // Sphere parameters:
    test::RealRandomizer<float> randomizer_float(-1.f, 1.f);
    float3_t shifts(randomizer_float.get() * 10, randomizer_float.get() * 10, randomizer_float.get() * 10);
    float radius_xy = test::RealRandomizer<float>(0, 20).get();
    float radius_z = test::RealRandomizer<float>(0, 20).get();
    float taper = test::RealRandomizer<float>(0, 20).get();

    AND_THEN("INVERT = false") {
        test::initDataRandom(h_data.get(), h_data.elements(), randomizer);
        cuda::memory::copy(h_data.get(), shape.x, d_data.get(), d_data.pitch(), shape_batched, stream);

        // Test saving the mask.
        cuda::mask::cylinder(d_mask.get(), d_mask.pitch(), shape, shifts, radius_xy, radius_z, taper, stream);
        cuda::memory::copy(d_mask.get(), d_mask.pitch(), h_cuda_mask.get(), shape.x, shape, stream);
        mask::cylinder(h_mask.get(), shape, shifts, radius_xy, radius_z, taper);
        cuda::Stream::synchronize(stream);
        TestType diff = test::getAverageDifference(h_mask.get(), h_cuda_mask.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(float(0.), 1e-6));

        // Test on-the-fly, in-place.
        cuda::mask::cylinder(d_data.get(), d_data.pitch(), d_data.get(), d_data.pitch(), shape,
                             shifts, radius_xy, radius_z, taper, batches, stream);
        cuda::memory::copy(d_data.get(), d_data.pitch(), h_cuda_data.get(), shape.x, shape_batched, stream);
        mask::cylinder(h_data.get(), h_data.get(), shape, shifts, radius_xy, radius_z, taper, batches);
        cuda::Stream::synchronize(stream);
        diff = test::getAverageDifference(h_data.get(), h_cuda_data.get(), elements * batches);
        REQUIRE_THAT(diff, test::isWithinAbs(float(0.), 1e-6));
    }

    AND_THEN("INVERT = true") {
        test::initDataRandom(h_data.get(), h_data.elements(), randomizer);
        cuda::memory::copy(h_data.get(), shape.x, d_data.get(), d_data.pitch(), shape_batched, stream);

        // Test saving the mask.
        cuda::mask::cylinder<true>(d_mask.get(), d_mask.pitch(), shape, shifts, radius_xy, radius_z,
                                   taper, stream);
        cuda::memory::copy(d_mask.get(), d_mask.pitch(), h_cuda_mask.get(), shape.x, shape, stream);
        mask::cylinder<true>(h_mask.get(), shape, shifts, radius_xy, radius_z, taper);
        cuda::Stream::synchronize(stream);
        TestType diff = test::getAverageDifference(h_mask.get(), h_cuda_mask.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(float(0.), 1e-6));

        // Test on-the-fly, in-place.
        cuda::mask::cylinder<true>(d_data.get(), d_data.pitch(), d_data.get(), d_data.pitch(), shape,
                                   shifts, radius_xy, radius_z, taper, batches, stream);
        cuda::memory::copy(d_data.get(), d_data.pitch(), h_cuda_data.get(), shape.x, shape_batched, stream);
        mask::cylinder<true>(h_data.get(), h_data.get(), shape, shifts, radius_xy, radius_z, taper, batches);
        cuda::Stream::synchronize(stream);
        diff = test::getAverageDifference(h_data.get(), h_cuda_data.get(), elements * batches);
        REQUIRE_THAT(diff, test::isWithinAbs(float(0.), 1e-6));
    }
}
