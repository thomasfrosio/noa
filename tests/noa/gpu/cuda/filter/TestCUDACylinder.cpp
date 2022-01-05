#include <noa/cpu/filter/Shape.h>
#include <noa/cpu/memory/PtrHost.h>

#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>
#include <noa/gpu/cuda/filter/Shape.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("cuda::filter::cylinder(), padded", "[noa][cuda][filter]", float, double) {
    test::Randomizer<TestType> randomizer(-5, 5);

    uint ndim = GENERATE(2U, 3U);
    size3_t shape = test::getRandomShape(ndim);
    size_t elements = noa::elements(shape);

    size_t batches = test::Randomizer<size_t>(1, 3).get();
    size3_t shape_batched(shape.x, shape.y * shape.z, batches);

    cpu::memory::PtrHost<TestType> h_mask(elements * batches);
    cpu::memory::PtrHost<TestType> h_data(elements * batches);

    cuda::memory::PtrDevicePadded<TestType> d_mask(shape * batches);
    cuda::memory::PtrDevicePadded<TestType> d_data(shape_batched);
    cpu::memory::PtrHost<TestType> h_cuda_mask(elements * batches);
    cpu::memory::PtrHost<TestType> h_cuda_data(elements * batches);

    cuda::Stream stream(cuda::Stream::SERIAL);
    cpu::Stream cpu_stream;

    // Sphere parameters:
    test::Randomizer<float> randomizer_float(-10.f, 10.f);
    float3_t shifts(randomizer_float.get() * 10, randomizer_float.get() * 10,
                    ndim == 3 ? randomizer_float.get() * 10 : 0);
    float radius_xy = test::Randomizer<float>(0, 20).get();
    float radius_z = test::Randomizer<float>(0, 20).get();
    float taper = test::Randomizer<float>(0, 20).get();
    float3_t center(shape / size_t{2});
    center += shifts;

    AND_THEN("INVERT = false") {
        test::randomize(h_data.get(), h_data.elements(), randomizer);
        cuda::memory::copy(h_data.get(), shape.x, d_data.get(), d_data.pitch(), shape_batched, stream);

        // Test saving the mask.
        cuda::filter::cylinder3D<false, TestType>(nullptr, 0, d_mask.get(), d_mask.pitch(), shape, batches,
                                                  center, radius_xy, radius_z, taper, stream);
        cuda::memory::copy(d_mask.get(), d_mask.pitch(), h_cuda_mask.get(), shape.x, shape_batched, stream);
        cpu::filter::cylinder<false, TestType>(nullptr, shape, h_mask.get(), shape, shape, batches,
                                               center, radius_xy, radius_z, taper, cpu_stream);
        cuda::Stream::synchronize(stream);
        REQUIRE(test::Matcher(test::MATCH_ABS, h_mask.get(), h_cuda_mask.get(), elements, 1e-5));

        // Test on-the-fly, in-place.
        cuda::filter::cylinder3D<false>(d_data.get(), d_data.pitch(), d_data.get(), d_data.pitch(), shape, batches,
                                        center, radius_xy, radius_z, taper, stream);
        cuda::memory::copy(d_data.get(), d_data.pitch(), h_cuda_data.get(), shape.x, shape_batched, stream);
        cpu::filter::cylinder<false>(h_data.get(), shape, h_data.get(), shape, shape, batches,
                                     center, radius_xy, radius_z, taper, cpu_stream);
        cuda::Stream::synchronize(stream);
        REQUIRE(test::Matcher(test::MATCH_ABS, h_data.get(), h_cuda_data.get(), elements * batches, 1e-5));
    }

    AND_THEN("INVERT = true") {
        test::randomize(h_data.get(), h_data.elements(), randomizer);
        cuda::memory::copy(h_data.get(), shape.x, d_data.get(), d_data.pitch(), shape_batched, stream);

        // Test saving the mask.
        cuda::filter::cylinder3D<true, TestType>(nullptr, 0, d_mask.get(), d_mask.pitch(), shape, batches,
                                                 center, radius_xy, radius_z,
                                                 taper, stream);
        cuda::memory::copy(d_mask.get(), d_mask.pitch(), h_cuda_mask.get(), shape.x, shape, stream);
        cpu::filter::cylinder<true, TestType>(nullptr, shape, h_mask.get(), shape, shape, batches,
                                              center, radius_xy, radius_z, taper, cpu_stream);
        cuda::Stream::synchronize(stream);
        REQUIRE(test::Matcher(test::MATCH_ABS, h_mask.get(), h_cuda_mask.get(), elements, 1e-5));

        // Test on-the-fly, in-place.
        cuda::filter::cylinder3D<true>(d_data.get(), d_data.pitch(), d_data.get(), d_data.pitch(), shape, batches,
                                       center, radius_xy, radius_z, taper, stream);
        cuda::memory::copy(d_data.get(), d_data.pitch(), h_cuda_data.get(), shape.x, shape_batched, stream);
        cpu::filter::cylinder<true>(h_data.get(), shape, h_data.get(), shape, shape, batches,
                                    center, radius_xy, radius_z, taper, cpu_stream);
        cuda::Stream::synchronize(stream);
        REQUIRE(test::Matcher(test::MATCH_ABS, h_data.get(), h_cuda_data.get(), elements * batches, 1e-5));
    }
}
