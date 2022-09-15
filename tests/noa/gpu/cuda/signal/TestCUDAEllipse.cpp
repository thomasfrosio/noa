#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/signal/Shape.h>

#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>
#include <noa/gpu/cuda/signal/Shape.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("cuda::signal::ellipse()", "[noa][cuda]", half_t, float, double) {
    test::Randomizer<TestType> rand(-5, 5);

    uint ndim = GENERATE(1u, 2u, 3u);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t stride = shape.strides();
    const size_t elements = shape.elements();
    INFO(shape);

    cpu::memory::PtrHost<TestType> h_mask(elements);
    cpu::memory::PtrHost<TestType> h_data(elements);

    cuda::memory::PtrDevicePadded<TestType> d_mask(shape);
    cuda::memory::PtrDevicePadded<TestType> d_data(shape);
    cpu::memory::PtrHost<TestType> h_cuda_mask(elements);
    cpu::memory::PtrHost<TestType> h_cuda_data(elements);

    cuda::Stream gpu_stream;
    cpu::Stream cpu_stream;

    // Sphere parameters:
    test::Randomizer<float> rand_float(-10.f, 10.f);
    test::Randomizer<float> rand_pfloat(0, 20);
    const float3_t shifts(ndim == 3 ? rand_float.get() * 10 : 0, rand_float.get() * 10, rand_float.get() * 10);
    const float3_t radius(ndim == 3 ? rand_pfloat.get() : 1, rand_pfloat.get(), rand_pfloat.get());
    const float taper = test::Randomizer<float>(0, 20).get();
    float3_t center(size3_t(shape.get(1)) / 2);
    center += shifts;

    double epsilon = std::is_same_v<TestType, half_t> ? 5e-4 : 5e-5;

    AND_THEN("INVERT = false") {
        test::randomize(h_data.get(), h_data.elements(), rand);
        cuda::memory::copy(h_data.share(), stride, d_data.share(), d_data.strides(), shape, gpu_stream);

        // Test saving the mask.
        cuda::signal::ellipse<false, TestType>(nullptr, {}, d_mask.share(), d_mask.strides(), shape, center, radius, taper, gpu_stream);
        cuda::memory::copy(d_mask.share(), d_mask.strides(),
                           h_cuda_mask.share(), stride,
                           d_mask.shape(), gpu_stream);
        cpu::signal::ellipse<false, TestType>(nullptr, {}, h_mask.share(), stride, shape, center, radius, taper, cpu_stream);
        gpu_stream.synchronize();
        cpu_stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, h_mask.get(), h_cuda_mask.get(), elements, epsilon));

        // Test on-the-fly, in-place.
        cuda::signal::ellipse<false, TestType>(d_data.share(), d_data.strides(), d_data.share(), d_data.strides(), shape, center, radius, taper, gpu_stream);
        cuda::memory::copy(d_data.share(), d_data.strides(), h_cuda_data.share(), stride, shape, gpu_stream);
        cpu::signal::ellipse<false, TestType>(h_data.share(), stride, h_data.share(), stride, shape,
                                             center, radius, taper, cpu_stream);
        gpu_stream.synchronize();
        cpu_stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, h_cuda_data.get(), h_data.get(), elements, epsilon));
    }

    AND_THEN("INVERT = true") {
        test::randomize(h_data.get(), h_data.elements(), rand);
        cuda::memory::copy(h_data.share(), stride, d_data.share(), d_data.strides(), shape, gpu_stream);

        // Test saving the mask.
        cuda::signal::ellipse<true, TestType>(nullptr, {}, d_mask.share(), d_mask.strides(), shape, center, radius, taper, gpu_stream);
        cuda::memory::copy(d_mask.share(), d_mask.strides(),
                           h_cuda_mask.share(), stride,
                           d_mask.shape(), gpu_stream);
        cpu::signal::ellipse<true, TestType>(nullptr, {}, h_mask.share(), stride, shape, center, radius, taper, cpu_stream);
        gpu_stream.synchronize();
        cpu_stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, h_mask.get(), h_cuda_mask.get(), elements, epsilon));

        // Test on-the-fly, in-place.
        cuda::signal::ellipse<true, TestType>(d_data.share(), d_data.strides(), d_data.share(), d_data.strides(), shape,
                                             center, radius, taper, gpu_stream);
        cuda::memory::copy(d_data.share(), d_data.strides(), h_cuda_data.share(), stride, shape, gpu_stream);
        cpu::signal::ellipse<true, TestType>(h_data.share(), stride, h_data.share(), stride, shape,
                                            center, radius, taper, cpu_stream);
        gpu_stream.synchronize();
        cpu_stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, h_cuda_data.get(), h_data.get(), elements, epsilon));
    }
}
