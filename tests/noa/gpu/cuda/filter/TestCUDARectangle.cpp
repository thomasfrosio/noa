#include <noa/gpu/cuda/filter/Rectangle.h>
#include <noa/cpu/filter/Rectangle.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("cuda::filter::rectangle(), contiguous", "[noa][cuda][filter]", float, double) {
    test::Randomizer<TestType> randomizer(-5, 5);

    uint ndim = GENERATE(2U, 3U);
    size3_t shape = test::getRandomShape(ndim);
    size_t elements = getElements(shape);

    uint batches = test::IntRandomizer<uint>(1, 3).get();

    cpu::memory::PtrHost<TestType> h_mask(elements);
    cpu::memory::PtrHost<TestType> h_data(elements * batches);

    cuda::memory::PtrDevice<TestType> d_mask(elements);
    cuda::memory::PtrDevice<TestType> d_data(elements * batches);
    cpu::memory::PtrHost<TestType> h_cuda_mask(elements);
    cpu::memory::PtrHost<TestType> h_cuda_data(elements * batches);

    cuda::Stream stream(cuda::Stream::SERIAL);

    // Sphere parameters:
    test::RealRandomizer<float> randomizer_float(-1.f, 1.f);
    test::RealRandomizer<float> randomizer_radius(1, 30);
    float3_t shifts(randomizer_float.get() * 10, randomizer_float.get() * 10, randomizer_float.get() * 10);
    float3_t radius(randomizer_radius.get(), randomizer_radius.get(), randomizer_radius.get());
    float taper = test::RealRandomizer<float>(0, 20).get();

    AND_THEN("INVERT = false") {
        test::initDataRandom(h_data.get(), h_data.elements(), randomizer);
        cuda::memory::copy(h_data.get(), d_data.get(), h_data.size(), stream);

        // Test saving the mask.
        cuda::filter::rectangle(d_mask.get(), shape, shifts, radius, taper, stream);
        cuda::memory::copy(d_mask.get(), h_cuda_mask.get(), d_mask.size(), stream);
        cpu::filter::rectangle(h_mask.get(), shape, shifts, radius, taper);
        cuda::Stream::synchronize(stream);
        TestType diff = test::getAverageDifference(h_mask.get(), h_cuda_mask.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(float(0.), 1e-6));

        // Test on-the-fly, in-place.
        cuda::filter::rectangle(d_data.get(), d_data.get(), shape, shifts, radius, taper, batches, stream);
        cuda::memory::copy(d_data.get(), h_cuda_data.get(), d_data.size(), stream);
        cpu::filter::rectangle(h_data.get(), h_data.get(), shape, shifts, radius, taper, batches);
        cuda::Stream::synchronize(stream);
        diff = test::getAverageDifference(h_data.get(), h_cuda_data.get(), elements * batches);
        REQUIRE_THAT(diff, test::isWithinAbs(float(0.), 1e-6));
    }

    AND_THEN("INVERT = true") {
        test::initDataRandom(h_data.get(), h_data.elements(), randomizer);
        cuda::memory::copy(h_data.get(), d_data.get(), h_data.size(), stream);

        // Test saving the mask.
        cuda::filter::rectangle<true>(d_mask.get(), shape, shifts, radius, taper, stream);
        cuda::memory::copy(d_mask.get(), h_cuda_mask.get(), d_mask.size(), stream);
        cpu::filter::rectangle<true>(h_mask.get(), shape, shifts, radius, taper);
        cuda::Stream::synchronize(stream);
        TestType diff = test::getAverageDifference(h_mask.get(), h_cuda_mask.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(float(0.), 1e-6));

        // Test on-the-fly, in-place.
        cuda::filter::rectangle<true>(d_data.get(), d_data.get(), shape, shifts, radius, taper, batches, stream);
        cuda::memory::copy(d_data.get(), h_cuda_data.get(), d_data.size(), stream);
        cpu::filter::rectangle<true>(h_data.get(), h_data.get(), shape, shifts, radius, taper, batches);
        cuda::Stream::synchronize(stream);
        diff = test::getAverageDifference(h_data.get(), h_cuda_data.get(), elements * batches);
        REQUIRE_THAT(diff, test::isWithinAbs(float(0.), 1e-6));
    }
}

TEMPLATE_TEST_CASE("cuda::filter::rectangle(), padded", "[noa][cuda][filter]", float, double) {
    test::Randomizer<TestType> randomizer(-5, 5);

    uint ndim = GENERATE(2U, 3U);
    size3_t shape = test::getRandomShape(ndim);
    size_t elements = getElements(shape);

    uint batches = test::IntRandomizer<uint>(1, 3).get();
    size3_t shape_batched(shape.x, shape.y * shape.z, batches);

    cpu::memory::PtrHost<TestType> h_mask(elements);
    cpu::memory::PtrHost<TestType> h_data(elements * batches);

    cuda::memory::PtrDevicePadded<TestType> d_mask(shape);
    cuda::memory::PtrDevicePadded<TestType> d_data(shape_batched);
    cpu::memory::PtrHost<TestType> h_cuda_mask(elements);
    cpu::memory::PtrHost<TestType> h_cuda_data(elements * batches);

    cuda::Stream stream(cuda::Stream::SERIAL);

    // Sphere parameters:
    test::RealRandomizer<float> randomizer_float(-1.f, 1.f);
    test::RealRandomizer<float> randomizer_radius(1, 30);
    float3_t shifts(randomizer_float.get() * 10, randomizer_float.get() * 10, randomizer_float.get() * 10);
    float3_t radius(randomizer_radius.get(), randomizer_radius.get(), randomizer_radius.get());
    float taper = test::RealRandomizer<float>(0, 20).get();

    AND_THEN("INVERT = false") {
        test::initDataRandom(h_data.get(), h_data.elements(), randomizer);
        cuda::memory::copy(h_data.get(), shape.x, d_data.get(), d_data.pitch(), shape_batched, stream);

        // Test saving the mask.
        cuda::filter::rectangle(d_mask.get(), d_mask.pitch(), shape, shifts, radius, taper, stream);
        cuda::memory::copy(d_mask.get(), d_mask.pitch(), h_cuda_mask.get(), shape.x, shape, stream);
        cpu::filter::rectangle(h_mask.get(), shape, shifts, radius, taper);
        cuda::Stream::synchronize(stream);
        TestType diff = test::getAverageDifference(h_mask.get(), h_cuda_mask.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(float(0.), 1e-6));

        // Test on-the-fly, in-place.
        cuda::filter::rectangle(d_data.get(), d_data.pitch(), d_data.get(), d_data.pitch(), shape,
                                shifts, radius, taper, batches, stream);
        cuda::memory::copy(d_data.get(), d_data.pitch(), h_cuda_data.get(), shape.x, shape_batched, stream);
        cpu::filter::rectangle(h_data.get(), h_data.get(), shape, shifts, radius, taper, batches);
        cuda::Stream::synchronize(stream);
        diff = test::getAverageDifference(h_data.get(), h_cuda_data.get(), elements * batches);
        REQUIRE_THAT(diff, test::isWithinAbs(float(0.), 1e-6));
    }

    AND_THEN("INVERT = true") {
        test::initDataRandom(h_data.get(), h_data.elements(), randomizer);
        cuda::memory::copy(h_data.get(), shape.x, d_data.get(), d_data.pitch(), shape_batched, stream);

        // Test saving the mask.
        cuda::filter::rectangle<true>(d_mask.get(), d_mask.pitch(), shape, shifts, radius, taper, stream);
        cuda::memory::copy(d_mask.get(), d_mask.pitch(), h_cuda_mask.get(), shape.x, shape, stream);
        cpu::filter::rectangle<true>(h_mask.get(), shape, shifts, radius, taper);
        cuda::Stream::synchronize(stream);
        TestType diff = test::getAverageDifference(h_mask.get(), h_cuda_mask.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(float(0.), 1e-6));

        // Test on-the-fly, in-place.
        cuda::filter::rectangle<true>(d_data.get(), d_data.pitch(), d_data.get(), d_data.pitch(), shape,
                                      shifts, radius, taper, batches, stream);
        cuda::memory::copy(d_data.get(), d_data.pitch(), h_cuda_data.get(), shape.x, shape_batched, stream);
        cpu::filter::rectangle<true>(h_data.get(), h_data.get(), shape, shifts, radius, taper, batches);
        cuda::Stream::synchronize(stream);
        diff = test::getAverageDifference(h_data.get(), h_cuda_data.get(), elements * batches);
        REQUIRE_THAT(diff, test::isWithinAbs(float(0.), 1e-6));
    }
}
