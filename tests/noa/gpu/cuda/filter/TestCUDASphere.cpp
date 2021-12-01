#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/filter/Shape.h>

#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>
#include <noa/gpu/cuda/filter/Shape.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("cuda::filter::sphere()", "[noa][cuda][filter]", float, double) {
    test::Randomizer<TestType> randomizer(-5, 5);

    uint ndim = GENERATE(2U, 3U);
    size3_t shape = test::getRandomShape(ndim);
    size_t elements = noa::elements(shape);

    size_t batches = test::Randomizer<size_t>(1, 3).get();
    size3_t shape_batched(shape.x, shape.y * shape.z, batches);

    cpu::memory::PtrHost<TestType> h_mask(elements * batches);
    cpu::memory::PtrHost<TestType> h_data(elements * batches);

    cuda::memory::PtrDevicePadded<TestType> d_mask(shape_batched);
    cuda::memory::PtrDevicePadded<TestType> d_data(shape_batched);
    cpu::memory::PtrHost<TestType> h_cuda_mask(elements * batches);
    cpu::memory::PtrHost<TestType> h_cuda_data(elements * batches);

    cuda::Stream stream(cuda::Stream::SERIAL);

    // Sphere parameters:
    test::Randomizer<float> randomizer_float(-10.f, 10.f);
    float3_t shifts(randomizer_float.get() * 10, randomizer_float.get() * 10, randomizer_float.get() * 10);
    float radius = test::Randomizer<float>(0, 20).get();
    float taper = test::Randomizer<float>(0, 20).get();
    float3_t center(shape / size_t{2});
    center += shifts;

    AND_THEN("INVERT = false") {
        test::randomize(h_data.get(), h_data.elements(), randomizer);
        cuda::memory::copy(h_data.get(), shape.x, d_data.get(), d_data.pitch(), shape_batched, stream);

        // Test saving the mask.
        if (ndim == 2)
            cuda::filter::sphere2D<false, TestType>(nullptr, 0, d_mask.get(), d_mask.pitch(),
                                                    {shape.x, shape.y}, batches,
                                                    {center.x, center.y}, radius, taper, stream);
        else
            cuda::filter::sphere3D<false, TestType>(nullptr, 0, d_mask.get(), d_mask.pitch(),
                                                    shape, batches,
                                                    center, radius, taper, stream);
        cuda::memory::copy(d_mask.get(), d_mask.pitch(), h_cuda_mask.get(), shape.x, d_mask.shape(), stream);
        if (ndim == 2)
            cpu::filter::sphere2D<false, TestType>(nullptr, h_mask.get(), {shape.x, shape.y}, batches,
                                                   {center.x, center.y}, radius, taper);
        else
            cpu::filter::sphere3D<false, TestType>(nullptr, h_mask.get(), shape, batches,
                                                   center, radius, taper);
        cuda::Stream::synchronize(stream);
        TestType diff = test::getAverageDifference(h_mask.get(), h_cuda_mask.get(), h_mask.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(float(0.), 1e-6));

        // Test on-the-fly, in-place.
        if (ndim == 2)
            cuda::filter::sphere2D<false>(d_data.get(), d_data.pitch(), d_data.get(), d_data.pitch(),
                                          {shape.x, shape.y}, batches,
                                          {center.x, center.y}, radius, taper, stream);
        else
            cuda::filter::sphere3D<false>(d_data.get(), d_data.pitch(), d_data.get(), d_data.pitch(),
                                          shape, batches,
                                          center, radius, taper, stream);
        cuda::memory::copy(d_data.get(), d_data.pitch(), h_cuda_data.get(), shape.x, shape_batched, stream);
        if (ndim == 2)
            cpu::filter::sphere2D<false>(h_data.get(), h_data.get(),
                                         {shape.x, shape.y}, batches,
                                         {center.x, center.y}, radius, taper);
        else
            cpu::filter::sphere3D<false>(h_data.get(), h_data.get(),
                                         shape, batches,
                                         center, radius, taper);
        cuda::Stream::synchronize(stream);
        diff = test::getAverageDifference(h_data.get(), h_cuda_data.get(), elements * batches);
        REQUIRE_THAT(diff, test::isWithinAbs(float(0.), 1e-6));
    }

    AND_THEN("INVERT = true") {
        test::randomize(h_data.get(), h_data.elements(), randomizer);
        cuda::memory::copy(h_data.get(), shape.x, d_data.get(), d_data.pitch(), shape_batched, stream);

        // Test saving the mask.
        if (ndim == 2)
            cuda::filter::sphere2D<true, TestType>(nullptr, 0, d_mask.get(), d_mask.pitch(),
                                                   {shape.x, shape.y}, batches,
                                                   {center.x, center.y}, radius, taper, stream);
        else
            cuda::filter::sphere3D<true, TestType>(nullptr, 0, d_mask.get(), d_mask.pitch(),
                                                   shape, batches,
                                                   center, radius, taper, stream);
        cuda::memory::copy(d_mask.get(), d_mask.pitch(), h_cuda_mask.get(), shape.x, d_mask.shape(), stream);
        if (ndim == 2)
            cpu::filter::sphere2D<true, TestType>(nullptr, h_mask.get(), {shape.x, shape.y}, batches,
                                                  {center.x, center.y}, radius, taper);
        else
            cpu::filter::sphere3D<true, TestType>(nullptr, h_mask.get(), shape, batches,
                                                  center, radius, taper);
        cuda::Stream::synchronize(stream);
        TestType diff = test::getAverageDifference(h_mask.get(), h_cuda_mask.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(float(0.), 1e-6));

        // Test on-the-fly, in-place.
        if (ndim == 2)
            cuda::filter::sphere2D<true>(d_data.get(), d_data.pitch(), d_data.get(), d_data.pitch(),
                                         {shape.x, shape.y}, batches,
                                         {center.x, center.y}, radius, taper, stream);
        else
            cuda::filter::sphere3D<true>(d_data.get(), d_data.pitch(), d_data.get(), d_data.pitch(),
                                         shape, batches,
                                         center, radius, taper, stream);
        cuda::memory::copy(d_data.get(), d_data.pitch(), h_cuda_data.get(), shape.x, shape_batched, stream);
        if (ndim == 2)
            cpu::filter::sphere2D<true>(h_data.get(), h_data.get(),
                                        {shape.x, shape.y}, batches,
                                        {center.x, center.y}, radius, taper);
        else
            cpu::filter::sphere3D<true>(h_data.get(), h_data.get(),
                                        shape, batches,
                                        center, radius, taper);
        cuda::Stream::synchronize(stream);
        diff = test::getAverageDifference(h_data.get(), h_cuda_data.get(), elements * batches);
        REQUIRE_THAT(diff, test::isWithinAbs(float(0.), 1e-6));
    }
}
