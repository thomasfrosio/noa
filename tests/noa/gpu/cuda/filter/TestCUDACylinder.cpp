#include <noa/cpu/filter/Shape.h>
#include <noa/cpu/memory/PtrHost.h>

#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>
#include <noa/gpu/cuda/filter/Shape.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("cuda::filter::cylinder(), padded", "[noa][cuda][filter]", half_t, float, double) {
    test::Randomizer<TestType> randomizer(-5, 5);

    uint ndim = GENERATE(1u, 2u, 3u);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t stride = shape.stride();
    const size_t elements = shape.elements();

    cpu::memory::PtrHost<TestType> h_mask(elements);
    cpu::memory::PtrHost<TestType> h_data(elements);

    cuda::memory::PtrDevicePadded<TestType> d_mask(shape);
    cuda::memory::PtrDevicePadded<TestType> d_data(shape);
    cpu::memory::PtrHost<TestType> h_cuda_mask(elements);
    cpu::memory::PtrHost<TestType> h_cuda_data(elements);

    cuda::Stream gpu_stream;
    cpu::Stream cpu_stream;

    // Sphere parameters:
    test::Randomizer<float> randomizer_float(-10.f, 10.f);
    float3_t shifts(ndim == 3 ? randomizer_float.get() * 10 : 0,
                    randomizer_float.get() * 10, randomizer_float.get() * 10);
    float radius = test::Randomizer<float>(0, 20).get();
    float length = test::Randomizer<float>(0, 20).get();
    float taper = test::Randomizer<float>(0, 20).get();
    float3_t center(size3_t{shape.get() + 1} / 2);
    center += shifts;

    double epsilon = std::is_same_v<TestType, half_t> ? 5e-4 : 5e-5;

    AND_THEN("INVERT = false") {
        test::randomize(h_data.get(), h_data.elements(), randomizer);
        cuda::memory::copy(h_data.get(), stride, d_data.get(), d_data.stride(), shape, gpu_stream);

        // Test saving the mask.
        cuda::filter::cylinder<false, TestType>(nullptr, {}, d_mask.get(), d_mask.stride(), shape,
                                                center, radius, length, taper, gpu_stream);
        cuda::memory::copy(d_mask.get(), d_mask.stride(), h_cuda_mask.get(), stride, d_mask.shape(), gpu_stream);
        cpu::filter::cylinder<false, TestType>(nullptr, {}, h_mask.get(), stride, shape,
                                               center, radius, length, taper, cpu_stream);
        gpu_stream.synchronize();
        cpu_stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, h_mask.get(), h_cuda_mask.get(), elements, epsilon));

        // Test on-the-fly, in-place.
        cuda::filter::cylinder<false>(d_data.get(), d_data.stride(), d_data.get(), d_data.stride(), shape,
                                      center, radius, length, taper, gpu_stream);
        cuda::memory::copy(d_data.get(), d_data.stride(), h_cuda_data.get(), stride, shape, gpu_stream);
        cpu::filter::cylinder<false>(h_data.get(), stride, h_data.get(), stride, shape,
                                     center, radius, length, taper, cpu_stream);
        gpu_stream.synchronize();
        cpu_stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, h_cuda_data.get(), h_data.get(), elements, epsilon));
    }

    AND_THEN("INVERT = true") {
        test::randomize(h_data.get(), h_data.elements(), randomizer);
        cuda::memory::copy(h_data.get(), stride, d_data.get(), d_data.stride(), shape, gpu_stream);

        // Test saving the mask.
        cuda::filter::cylinder<true, TestType>(nullptr, {}, d_mask.get(), d_mask.stride(), shape,
                                               center, radius, length, taper, gpu_stream);
        cuda::memory::copy(d_mask.get(), d_mask.stride(), h_cuda_mask.get(), stride, d_mask.shape(), gpu_stream);
        cpu::filter::cylinder<true, TestType>(nullptr, {}, h_mask.get(), stride, shape,
                                              center, radius, length, taper, cpu_stream);
        gpu_stream.synchronize();
        cpu_stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, h_mask.get(), h_cuda_mask.get(), elements, epsilon));

        // Test on-the-fly, in-place.
        cuda::filter::cylinder<true>(d_data.get(), d_data.stride(), d_data.get(), d_data.stride(), shape,
                                     center, radius, length, taper, gpu_stream);
        cuda::memory::copy(d_data.get(), d_data.stride(), h_cuda_data.get(), stride, shape, gpu_stream);
        cpu::filter::cylinder<true>(h_data.get(), stride, h_data.get(), stride, shape,
                                    center, radius, length, taper, cpu_stream);
        gpu_stream.synchronize();
        cpu_stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, h_cuda_data.get(), h_data.get(), elements, epsilon));
    }
}
