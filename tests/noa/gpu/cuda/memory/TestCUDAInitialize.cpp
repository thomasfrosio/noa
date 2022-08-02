#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/memory/Arange.h>
#include <noa/cpu/memory/Linspace.h>
#include <noa/cpu/memory/Set.h>

#include <noa/gpu/cuda/memory/Copy.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/PtrManaged.h>

#include <noa/gpu/cuda/memory/Arange.h>
#include <noa/gpu/cuda/memory/Linspace.h>
#include <noa/gpu/cuda/memory/Set.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEMPLATE_TEST_CASE("cuda::memory::set()", "[noa][cuda][memory]",
                   int32_t, uint32_t, int64_t, uint64_t, float, double, cfloat_t, cdouble_t) {
    const uint ndim = GENERATE(1u, 2u, 3u);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t stride = shape.strides();
    const size_t elements = shape.elements();

    const TestType value = test::Randomizer<TestType>(0, 400).get();

    cpu::Stream cpu_stream;
    cuda::Stream gpu_stream;

    cpu::memory::PtrHost<TestType> h_data(elements);
    cuda::memory::PtrManaged<TestType> d_data(elements, gpu_stream);

    cpu::memory::set(h_data.share(), stride, shape, value, cpu_stream);
    cuda::memory::set(d_data.share(), stride, shape, value, gpu_stream);
    cpu_stream.synchronize();
    gpu_stream.synchronize();

    REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_data.get(), d_data.get(), elements, 1e-6));
}

TEMPLATE_TEST_CASE("cuda::memory::set(), padded", "[noa][cuda][memory]",
                   int32_t, uint32_t, int64_t, uint64_t, float, double, cfloat_t, cdouble_t) {
    const uint ndim = GENERATE(1u, 2u, 3u);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t stride = shape.strides();
    const size_t elements = shape.elements();

    const TestType value = test::Randomizer<TestType>(0, 400).get();

    cpu::Stream cpu_stream;
    cuda::Stream gpu_stream;

    cpu::memory::PtrHost<TestType> h_data(elements);
    cpu::memory::PtrHost<TestType> h_cuda_data(elements);
    cuda::memory::PtrDevicePadded<TestType> d_data(shape);

    cpu::memory::set(h_data.share(), stride, shape, value, cpu_stream);
    cuda::memory::set(d_data.share(), d_data.strides(), shape, value, gpu_stream);
    cuda::memory::copy(d_data.share(), d_data.strides(), h_cuda_data.share(), stride, shape, gpu_stream);
    cpu_stream.synchronize();
    gpu_stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_data.get(), h_cuda_data.get(), elements, 1e-6));
}

TEMPLATE_TEST_CASE("cuda::memory::arange()", "[noa][cuda][memory]",
                   int32_t, uint32_t, int64_t, uint64_t, float, double, cfloat_t, cdouble_t) {
    const uint ndim = GENERATE(1u, 2u, 3u);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t stride = shape.strides();
    const size_t elements = shape.elements();

    cpu::Stream cpu_stream;
    cuda::Stream gpu_stream;

    cpu::memory::PtrHost<TestType> h_data(elements);
    cuda::memory::PtrManaged<TestType> d_data(elements, gpu_stream);

    cpu::memory::arange(h_data.share(), stride, shape, cpu_stream);
    cuda::memory::arange(d_data.share(), stride, shape, gpu_stream);
    cpu_stream.synchronize();
    gpu_stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_data.get(), d_data.get(), elements, 1e-6));
}

TEMPLATE_TEST_CASE("cuda::memory::arange(), start, step", "[noa][cuda][memory]",
                   int32_t, uint32_t, int64_t, uint64_t, float, double, cfloat_t, cdouble_t) {
    const uint ndim = GENERATE(1u, 2u, 3u);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t stride = shape.strides();
    const size_t elements = shape.elements();

    const TestType start = 1;
    const TestType step = 2;

    cpu::Stream cpu_stream;
    cuda::Stream gpu_stream;

    cpu::memory::PtrHost<TestType> h_data(elements);
    cuda::memory::PtrManaged<TestType> d_data(elements, gpu_stream);

    cpu::memory::arange(h_data.share(), stride, shape, start, step, cpu_stream);
    cuda::memory::arange(d_data.share(), stride, shape, start, step, gpu_stream);
    cpu_stream.synchronize();
    gpu_stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_data.get(), d_data.get(), elements, 1e-6));
}

TEMPLATE_TEST_CASE("cuda::memory::arange(), padded", "[noa][cuda][memory]",
                   int32_t, uint32_t, int64_t, uint64_t, float, double, cfloat_t, cdouble_t) {
    const uint ndim = GENERATE(1u, 2u, 3u);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t stride = shape.strides();
    const size_t elements = shape.elements();

    cpu::Stream cpu_stream;
    cuda::Stream gpu_stream;

    cpu::memory::PtrHost<TestType> h_data(elements);
    cpu::memory::PtrHost<TestType> h_cuda_data(elements);
    cuda::memory::PtrDevicePadded<TestType> d_data(shape);

    cpu::memory::arange(h_data.share(), stride, shape, cpu_stream);
    cuda::memory::arange(d_data.share(), d_data.strides(), shape, gpu_stream);
    cuda::memory::copy(d_data.share(), d_data.strides(), h_cuda_data.share(), stride, shape, gpu_stream);
    cpu_stream.synchronize();
    gpu_stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_data.get(), h_cuda_data.get(), elements, 1e-6));
}

TEMPLATE_TEST_CASE("cuda::memory::arange(), padded, start, step", "[noa][cuda][memory]",
                   int32_t, uint32_t, int64_t, uint64_t, float, double, cfloat_t, cdouble_t) {
    const uint ndim = GENERATE(1u, 2u, 3u);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t stride = shape.strides();
    const size_t elements = shape.elements();

    const TestType start = 1;
    const TestType step = 2;

    cpu::Stream cpu_stream;
    cuda::Stream gpu_stream;

    cpu::memory::PtrHost<TestType> h_data(elements);
    cpu::memory::PtrHost<TestType> h_cuda_data(elements);
    cuda::memory::PtrDevicePadded<TestType> d_data(shape);

    cpu::memory::arange(h_data.share(), stride, shape, start, step, cpu_stream);
    cuda::memory::arange(d_data.share(), d_data.strides(), shape, start, step, gpu_stream);
    cuda::memory::copy(d_data.share(), d_data.strides(), h_cuda_data.share(), stride, shape, gpu_stream);
    cpu_stream.synchronize();
    gpu_stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_data.get(), h_cuda_data.get(), elements, 1e-6));
}

TEMPLATE_TEST_CASE("cuda::memory::linspace()", "[noa][cuda][memory]",
                   float, double, cfloat_t, cdouble_t) {
    const uint ndim = GENERATE(1u, 2u, 3u);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t stride = shape.strides();
    const size_t elements = shape.elements();

    const TestType start = -10;
    const TestType stop = 20;
    const bool endpoint = GENERATE(true, false);

    cpu::Stream cpu_stream;
    cuda::Stream gpu_stream;

    cpu::memory::PtrHost<TestType> h_data(elements);
    cuda::memory::PtrManaged<TestType> d_data(elements, gpu_stream);

    cpu::memory::linspace(h_data.share(), stride, shape, start, stop, endpoint, cpu_stream);
    cuda::memory::linspace(d_data.share(), stride, shape, start, stop, endpoint, gpu_stream);
    cpu_stream.synchronize();
    gpu_stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_data.get(), d_data.get(), elements, 1e-6));
}

TEMPLATE_TEST_CASE("cuda::memory::linspace(), padded", "[noa][cuda][memory]",
                   float, double, cfloat_t, cdouble_t) {
    const uint ndim = GENERATE(1u, 2u, 3u);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t stride = shape.strides();
    const size_t elements = shape.elements();

    const TestType start = -10;
    const TestType stop = 20;
    const bool endpoint = GENERATE(true, false);

    cpu::Stream cpu_stream;
    cuda::Stream gpu_stream;

    cpu::memory::PtrHost<TestType> h_data(elements);
    cpu::memory::PtrHost<TestType> h_cuda_data(elements);
    cuda::memory::PtrDevicePadded<TestType> d_data(shape);

    cpu::memory::linspace(h_data.share(), stride, shape, start, stop, endpoint, cpu_stream);
    cuda::memory::linspace(d_data.share(), d_data.strides(), shape, start, stop, endpoint, gpu_stream);
    cuda::memory::copy(d_data.share(), d_data.strides(), h_cuda_data.share(), stride, shape, gpu_stream);
    cpu_stream.synchronize();
    gpu_stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_data.get(), h_cuda_data.get(), elements, 1e-6));
}
