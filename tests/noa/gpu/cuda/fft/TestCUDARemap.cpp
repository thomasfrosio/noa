#include <noa/cpu/fft/Remap.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>
#include <noa/gpu/cuda/fft/Remap.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("cuda::fft::h2f(), f2h()", "[noa][cuda][fft]",
                   half_t, float, double, chalf_t, cfloat_t, cdouble_t) {
    test::Randomizer<TestType> randomizer_data(1., 128.);

    const uint ndim = GENERATE(1U, 2U, 3U);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t shape_fft = shape.fft();
    const size4_t stride = shape.strides();
    const size4_t stride_fft = shape_fft.strides();
    const size_t elements = shape.elements();
    const size_t elements_fft = shape_fft.elements();

    cuda::Stream gpu_stream(cuda::Stream::SERIAL);
    cpu::Stream cpu_stream;
    INFO(shape);

    AND_THEN("h2f") {
        cpu::memory::PtrHost<TestType> h_half(elements_fft);
        cpu::memory::PtrHost<TestType> h_full(elements);
        test::randomize(h_half.get(), h_half.elements(), randomizer_data);
        test::memset(h_full.get(), h_full.elements(), 0);
        cpu::fft::remap(fft::H2F, h_half.get(), stride_fft, h_full.get(), stride, shape, cpu_stream);

        AND_THEN("contiguous") {
            cuda::memory::PtrDevice<TestType> d_half(elements_fft);
            cuda::memory::PtrDevice<TestType> d_full(elements);
            cpu::memory::PtrHost<TestType> h_full_cuda(elements);

            cuda::memory::copy(h_half.get(), d_half.get(), h_half.size(), gpu_stream);
            cuda::fft::remap(fft::H2F, d_half.get(), stride_fft, d_full.get(), stride, shape, gpu_stream);
            cuda::memory::copy(d_full.get(), h_full_cuda.get(), d_full.size(), gpu_stream);
            gpu_stream.synchronize();

            REQUIRE(test::Matcher(test::MATCH_ABS, h_full.get(), h_full_cuda.get(), elements, 1e-14));
        }

        AND_THEN("padded") {
            cuda::memory::PtrDevicePadded<TestType> d_half(shape_fft);
            cuda::memory::PtrDevicePadded<TestType> d_full(shape);
            cpu::memory::PtrHost<TestType> h_full_cuda(elements);

            cuda::memory::copy(h_half.get(), stride_fft, d_half.get(), d_half.strides(), shape_fft, gpu_stream);
            cuda::fft::remap(fft::H2F, d_half.get(), d_half.strides(), d_full.get(), d_full.strides(), shape, gpu_stream);
            cuda::memory::copy(d_full.get(), d_full.strides(), h_full_cuda.get(), stride, shape, gpu_stream);
            gpu_stream.synchronize();

            REQUIRE(test::Matcher(test::MATCH_ABS, h_full.get(), h_full_cuda.get(), elements, 1e-14));
        }
    }

    AND_THEN("f2h") {
        cpu::memory::PtrHost<TestType> h_full(elements);
        cpu::memory::PtrHost<TestType> h_half(elements_fft);
        test::randomize(h_full.get(), h_full.elements(), randomizer_data);
        test::memset(h_half.get(), h_half.elements(), 0);
        cpu::fft::remap(fft::F2H, h_full.get(), stride, h_half.get(), stride_fft, shape, cpu_stream);

        AND_THEN("contiguous") {
            cuda::memory::PtrDevice<TestType> d_full(elements);
            cuda::memory::PtrDevice<TestType> d_half(elements_fft);
            cpu::memory::PtrHost<TestType> h_half_cuda(elements_fft);
            cuda::memory::copy(h_full.get(), d_full.get(), h_full.size(), gpu_stream);
            cuda::fft::remap(fft::F2H, d_full.get(), stride, d_half.get(), stride_fft, shape, gpu_stream);
            cuda::memory::copy(d_half.get(), h_half_cuda.get(), d_half.size(), gpu_stream);
            gpu_stream.synchronize();

            REQUIRE(test::Matcher(test::MATCH_ABS, h_half.get(), h_half_cuda.get(), elements_fft, 1e-14));
        }

        AND_THEN("padded") {
            cuda::memory::PtrDevicePadded<TestType> d_full(shape);
            cuda::memory::PtrDevicePadded<TestType> d_half(shape_fft);
            cpu::memory::PtrHost<TestType> h_half_cuda(elements_fft);

            cuda::memory::copy(h_full.get(), stride, d_full.get(), d_full.strides(), shape, gpu_stream);
            cuda::fft::remap(fft::F2H, d_full.get(), d_full.strides(), d_half.get(), d_half.strides(), shape, gpu_stream);
            cuda::memory::copy(d_half.get(), d_half.strides(), h_half_cuda.get(), stride_fft, shape_fft, gpu_stream);
            gpu_stream.synchronize();

            REQUIRE(test::Matcher(test::MATCH_ABS, h_half.get(), h_half_cuda.get(), elements_fft, 1e-14));
        }
    }
}

TEMPLATE_TEST_CASE("cuda::fft::hc2f(), f2hc()", "[noa][cuda][fft]",
                   half_t, float, double, chalf_t, cfloat_t, cdouble_t) {
    test::Randomizer<TestType> randomizer_data(1., 128.);

    const uint ndim = GENERATE(1U, 2U, 3U);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t shape_fft = shape.fft();
    const size4_t stride = shape.strides();
    const size4_t stride_fft = shape_fft.strides();
    const size_t elements = shape.elements();
    const size_t elements_fft = shape_fft.elements();

    cuda::Stream gpu_stream(cuda::Stream::SERIAL);
    cpu::Stream cpu_stream;
    INFO(shape);

    AND_THEN("hc2f") {
        cpu::memory::PtrHost<TestType> h_half(elements_fft);
        cpu::memory::PtrHost<TestType> h_full(elements);
        test::randomize(h_half.get(), h_half.elements(), randomizer_data);
        test::memset(h_full.get(), h_full.elements(), 0);
        cpu::fft::remap(fft::HC2F, h_half.get(), stride_fft, h_full.get(), stride, shape, cpu_stream);

        AND_THEN("contiguous") {
            cuda::memory::PtrDevice<TestType> d_half(elements_fft);
            cuda::memory::PtrDevice<TestType> d_full(elements);
            cpu::memory::PtrHost<TestType> h_full_cuda(elements);

            cuda::memory::copy(h_half.get(), d_half.get(), h_half.size(), gpu_stream);
            cuda::fft::remap(fft::HC2F, d_half.get(), stride_fft, d_full.get(), stride, shape, gpu_stream);
            cuda::memory::copy(d_full.get(), h_full_cuda.get(), d_full.size(), gpu_stream);
            gpu_stream.synchronize();

            REQUIRE(test::Matcher(test::MATCH_ABS, h_full.get(), h_full_cuda.get(), elements, 1e-14));
        }

        AND_THEN("padded") {
            cuda::memory::PtrDevicePadded<TestType> d_half(shape_fft);
            cuda::memory::PtrDevicePadded<TestType> d_full(shape);
            cpu::memory::PtrHost<TestType> h_full_cuda(elements);

            cuda::memory::copy(h_half.get(), stride_fft, d_half.get(), d_half.strides(), shape_fft, gpu_stream);
            cuda::fft::remap(fft::HC2F, d_half.get(), d_half.strides(), d_full.get(), d_full.strides(), shape, gpu_stream);
            cuda::memory::copy(d_full.get(), d_full.strides(), h_full_cuda.get(), stride, shape, gpu_stream);
            gpu_stream.synchronize();

            REQUIRE(test::Matcher(test::MATCH_ABS, h_full.get(), h_full_cuda.get(), elements, 1e-14));
        }
    }

    AND_THEN("f2hc") {
        cpu::memory::PtrHost<TestType> h_full(elements);
        cpu::memory::PtrHost<TestType> h_half(elements_fft);
        test::randomize(h_full.get(), h_full.elements(), randomizer_data);
        test::memset(h_half.get(), h_half.elements(), 0);
        cpu::fft::remap(fft::F2HC, h_full.get(), stride, h_half.get(), stride_fft, shape, cpu_stream);

        AND_THEN("contiguous") {
            cuda::memory::PtrDevice<TestType> d_full(elements);
            cuda::memory::PtrDevice<TestType> d_half(elements_fft);
            cpu::memory::PtrHost<TestType> h_half_cuda(elements_fft);
            cuda::memory::copy(h_full.get(), d_full.get(), h_full.size(), gpu_stream);
            cuda::fft::remap(fft::F2HC, d_full.get(), stride, d_half.get(), stride_fft, shape, gpu_stream);
            cuda::memory::copy(d_half.get(), h_half_cuda.get(), d_half.size(), gpu_stream);
            gpu_stream.synchronize();

            REQUIRE(test::Matcher(test::MATCH_ABS, h_half.get(), h_half_cuda.get(), elements_fft, 1e-14));
        }

        AND_THEN("padded") {
            cuda::memory::PtrDevicePadded<TestType> d_full(shape);
            cuda::memory::PtrDevicePadded<TestType> d_half(shape_fft);
            cpu::memory::PtrHost<TestType> h_half_cuda(elements_fft);

            cuda::memory::copy(h_full.get(), stride, d_full.get(), d_full.strides(), shape, gpu_stream);
            cuda::fft::remap(fft::F2HC, d_full.get(), d_full.strides(), d_half.get(), d_half.strides(), shape, gpu_stream);
            cuda::memory::copy(d_half.get(), d_half.strides(), h_half_cuda.get(), stride_fft, shape_fft, gpu_stream);
            gpu_stream.synchronize();

            REQUIRE(test::Matcher(test::MATCH_ABS, h_half.get(), h_half_cuda.get(), elements_fft, 1e-14));
        }
    }
}

TEMPLATE_TEST_CASE("cuda::fft::f2fc(), fc2f()", "[noa][cuda][fft]",
                   half_t, float, double, chalf_t, cfloat_t, cdouble_t) {
    test::Randomizer<TestType> randomizer_data(1., 128.);

    const uint ndim = GENERATE(1U, 2U, 3U);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t stride = shape.strides();
    const size_t elements = shape.elements();

    cuda::Stream gpu_stream(cuda::Stream::SERIAL);
    cpu::Stream cpu_stream;
    INFO(shape);

    AND_THEN("f2fc") {
        cpu::memory::PtrHost<TestType> h_full(elements);
        cpu::memory::PtrHost<TestType> h_full_centered(elements);
        test::randomize(h_full.get(), h_full.elements(), randomizer_data);
        test::memset(h_full_centered.get(), h_full_centered.elements(), 0);
        cpu::fft::remap(fft::F2FC, h_full.get(), stride, h_full_centered.get(), stride, shape, cpu_stream);

        AND_THEN("contiguous") {
            cuda::memory::PtrDevice<TestType> d_full(elements);
            cuda::memory::PtrDevice<TestType> d_full_centered(elements);
            cpu::memory::PtrHost<TestType> h_full_centered_cuda(elements);

            cuda::memory::copy(h_full.get(), d_full.get(), h_full.size(), gpu_stream);
            cuda::fft::remap(fft::F2FC, d_full.get(), stride, d_full_centered.get(), stride, shape, gpu_stream);
            cuda::memory::copy(d_full_centered.get(), h_full_centered_cuda.get(), h_full.size(), gpu_stream);
            gpu_stream.synchronize();

            REQUIRE(test::Matcher(test::MATCH_ABS, h_full_centered.get(), h_full_centered_cuda.get(), elements, 1e-14));
        }

        AND_THEN("padded") {
            cuda::memory::PtrDevicePadded<TestType> d_full(shape);
            cuda::memory::PtrDevicePadded<TestType> d_full_centered(shape);
            cpu::memory::PtrHost<TestType> h_full_centered_cuda(elements);

            cuda::memory::copy(h_full.get(), stride, d_full.get(), d_full.strides(), shape, gpu_stream);
            cuda::fft::remap(fft::F2FC, d_full.get(), d_full.strides(),
                             d_full_centered.get(), d_full_centered.strides(),
                             shape, gpu_stream);
            cuda::memory::copy(d_full_centered.get(), d_full_centered.strides(),
                               h_full_centered_cuda.get(), stride, shape, gpu_stream);
            gpu_stream.synchronize();

            REQUIRE(test::Matcher(test::MATCH_ABS, h_full_centered.get(), h_full_centered_cuda.get(), elements, 1e-14));
        }
    }

    AND_THEN("f2fc") {
        cpu::memory::PtrHost<TestType> h_full_centered(elements);
        cpu::memory::PtrHost<TestType> h_full(elements);
        test::randomize(h_full_centered.get(), h_full_centered.elements(), randomizer_data);
        test::memset(h_full.get(), h_full.elements(), 0);
        cpu::fft::remap(fft::FC2F, h_full_centered.get(), stride, h_full.get(), stride, shape, cpu_stream);

        AND_THEN("contiguous") {
            cuda::memory::PtrDevice<TestType> d_full_centered(elements);
            cuda::memory::PtrDevice<TestType> d_full(elements);
            cpu::memory::PtrHost<TestType> h_full_cuda(elements);

            cuda::memory::copy(h_full_centered.get(), d_full_centered.get(), d_full_centered.elements(), gpu_stream);
            cuda::fft::remap(fft::FC2F, d_full_centered.get(), stride, d_full.get(), stride, shape, gpu_stream);
            cuda::memory::copy(d_full.get(), h_full_cuda.get(), h_full.elements(), gpu_stream);
            gpu_stream.synchronize();

            REQUIRE(test::Matcher(test::MATCH_ABS, h_full.get(), h_full_cuda.get(), elements, 1e-14));
        }

        AND_THEN("padded") {
            cuda::memory::PtrDevicePadded<TestType> d_full_centered(shape);
            cuda::memory::PtrDevicePadded<TestType> d_full(shape);
            cpu::memory::PtrHost<TestType> h_full_cuda(elements);

            cuda::memory::copy(h_full_centered.get(), stride,
                               d_full_centered.get(), d_full_centered.strides(), shape, gpu_stream);
            cuda::fft::remap(fft::FC2F, d_full_centered.get(), d_full_centered.strides(),
                             d_full.get(), d_full.strides(),
                             shape, gpu_stream);
            cuda::memory::copy(d_full.get(), d_full.strides(), h_full_cuda.get(), stride, shape, gpu_stream);
            gpu_stream.synchronize();

            REQUIRE(test::Matcher(test::MATCH_ABS, h_full.get(), h_full_cuda.get(), elements, 1e-14));
        }
    }
}

TEMPLATE_TEST_CASE("cuda::fft::h2hc(), hc2h()", "[noa][cuda][fft]",
                   half_t, float, double, chalf_t, cfloat_t, cdouble_t) {
    test::Randomizer<TestType> randomizer_data(1., 128.);

    const uint ndim = GENERATE(1U, 2U, 3U);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t shape_fft = shape.fft();
    const size4_t stride_fft = shape_fft.strides();
    const size_t elements_fft = shape_fft.elements();

    cuda::Stream gpu_stream(cuda::Stream::SERIAL);
    cpu::Stream cpu_stream;
    INFO(shape);

    AND_THEN("h2hc") {
        cpu::memory::PtrHost<TestType> h_half(elements_fft);
        cpu::memory::PtrHost<TestType> h_half_centered(elements_fft);
        test::randomize(h_half.get(), h_half.elements(), randomizer_data);
        test::memset(h_half_centered.get(), h_half_centered.elements(), 0);
        cpu::fft::remap(fft::H2HC, h_half.get(), stride_fft, h_half_centered.get(), stride_fft, shape, cpu_stream);

        AND_THEN("contiguous") {
            cuda::memory::PtrDevice<TestType> d_half(elements_fft);
            cuda::memory::PtrDevice<TestType> d_half_centered(elements_fft);
            cpu::memory::PtrHost<TestType> h_half_centered_cuda(elements_fft);

            cuda::memory::copy(h_half.get(), d_half.get(), h_half.size(), gpu_stream);
            cuda::fft::remap(fft::H2HC, d_half.get(), stride_fft,
                             d_half_centered.get(), stride_fft,
                             shape, gpu_stream);
            cuda::memory::copy(d_half_centered.get(), h_half_centered_cuda.get(), h_half.size(), gpu_stream);
            gpu_stream.synchronize();

            test::Matcher match(test::MATCH_ABS, h_half_centered.get(), h_half_centered_cuda.get(), elements_fft, 1e-14);
            REQUIRE(match);
        }

        AND_THEN("padded") {
            cuda::memory::PtrDevicePadded<TestType> d_half(shape_fft);
            cuda::memory::PtrDevicePadded<TestType> d_half_centered(shape_fft);
            cpu::memory::PtrHost<TestType> h_half_centered_cuda(elements_fft);

            cuda::memory::copy(h_half.get(), stride_fft, d_half.get(), d_half.strides(), shape_fft, gpu_stream);
            cuda::fft::remap(fft::H2HC, d_half.get(), d_half.strides(),
                             d_half_centered.get(), d_half_centered.strides(),
                             shape, gpu_stream);
            cuda::memory::copy(d_half_centered.get(), d_half_centered.strides(),
                               h_half_centered_cuda.get(), stride_fft,
                               shape_fft, gpu_stream);
            gpu_stream.synchronize();

            test::Matcher match(test::MATCH_ABS, h_half_centered.get(), h_half_centered_cuda.get(), elements_fft, 1e-14);
            REQUIRE(match);
        }
    }

    AND_THEN("hc2h") {
        cpu::memory::PtrHost<TestType> h_half_centered(elements_fft);
        cpu::memory::PtrHost<TestType> h_half(elements_fft);
        test::randomize(h_half_centered.get(), h_half_centered.elements(), randomizer_data);
        test::memset(h_half.get(), h_half.elements(), 0);
        cpu::fft::remap(fft::HC2H, h_half_centered.get(), stride_fft, h_half.get(), stride_fft, shape, cpu_stream);

        AND_THEN("contiguous") {
            cuda::memory::PtrDevice<TestType> d_half_centered(elements_fft);
            cuda::memory::PtrDevice<TestType> d_half(elements_fft);
            cpu::memory::PtrHost<TestType> h_half_cuda(elements_fft);

            cuda::memory::copy(h_half_centered.get(), d_half_centered.get(), h_half.size(), gpu_stream);
            cuda::fft::remap(fft::HC2H, d_half_centered.get(), stride_fft,
                             d_half.get(), stride_fft,
                             shape, gpu_stream);
            cuda::memory::copy(d_half.get(), h_half_cuda.get(), h_half.size(), gpu_stream);
            gpu_stream.synchronize();

            REQUIRE(test::Matcher(test::MATCH_ABS, h_half.get(), h_half_cuda.get(), elements_fft, 1e-14));
        }

        AND_THEN("padded") {
            cuda::memory::PtrDevicePadded<TestType> d_half_centered(shape_fft);
            cuda::memory::PtrDevicePadded<TestType> d_half(shape_fft);
            cpu::memory::PtrHost<TestType> h_half_cuda(elements_fft);

            cuda::memory::copy(h_half_centered.get(), stride_fft,
                               d_half_centered.get(), d_half_centered.strides(),
                               shape_fft, gpu_stream);
            cuda::fft::remap(fft::HC2H, d_half_centered.get(), d_half_centered.strides(),
                             d_half.get(), d_half.strides(),
                             shape, gpu_stream);
            cuda::memory::copy(d_half.get(), d_half.strides(),
                               h_half_cuda.get(), stride_fft,
                               shape_fft, gpu_stream);
            gpu_stream.synchronize();

            REQUIRE(test::Matcher(test::MATCH_ABS, h_half.get(), h_half_cuda.get(), elements_fft, 1e-14));
        }
    }
}

TEMPLATE_TEST_CASE("cuda::fft::h2hc(), in-place", "[noa][cuda][fft]",
                   half_t, float, double, chalf_t, cfloat_t, cdouble_t) {
    test::Randomizer<TestType> randomizer_data(1., 128.);

    const uint ndim = GENERATE(1U, 2U, 3U);
    const size4_t shape = test::getRandomShapeBatched(ndim, true); // even only
    const size4_t shape_fft = shape.fft();
    const size4_t stride_fft = shape_fft.strides();
    const size_t elements_fft = shape_fft.elements();

    cuda::Stream gpu_stream(cuda::Stream::SERIAL);
    cpu::Stream cpu_stream;
    INFO(shape);

    cpu::memory::PtrHost<TestType> h_half(elements_fft);
    cpu::memory::PtrHost<TestType> h_half_centered(h_half.size());
    test::randomize(h_half.get(), h_half.size(), randomizer_data);
    cpu::fft::remap(fft::H2HC, h_half.get(), stride_fft, h_half_centered.get(), stride_fft, shape, cpu_stream);

    cuda::memory::PtrDevicePadded<TestType> d_half(shape_fft);
    cpu::memory::PtrHost<TestType> h_half_centered_cuda(h_half.size());

    cuda::memory::copy(h_half.get(), stride_fft, d_half.get(), d_half.strides(), shape_fft, gpu_stream);
    cuda::fft::remap(fft::H2HC, d_half.get(), d_half.strides(), d_half.get(), d_half.strides(), shape, gpu_stream);
    cuda::memory::copy(d_half.get(), d_half.strides(), h_half_centered_cuda.get(), stride_fft, shape_fft, gpu_stream);
    gpu_stream.synchronize();

    REQUIRE(test::Matcher(test::MATCH_ABS, h_half_centered.get(), h_half_centered_cuda.get(), elements_fft, 1e-14));
}

TEMPLATE_TEST_CASE("cuda::fft::fc2h()", "[noa][cuda][fft]",
                   half_t, float, double, chalf_t, cfloat_t, cdouble_t) {
    test::Randomizer<TestType> randomizer_data(1., 128.);

    const uint ndim = GENERATE(1U, 2U, 3U);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t shape_fft = shape.fft();
    const size4_t stride = shape.strides();
    const size4_t stride_fft = shape_fft.strides();
    const size_t elements = shape.elements();
    const size_t elements_fft = shape_fft.elements();

    cuda::Stream gpu_stream(cuda::Stream::SERIAL);
    cpu::Stream cpu_stream;
    INFO(shape);

    AND_THEN("fc2h") {
        cpu::memory::PtrHost<TestType> h_full_centered(elements);
        cpu::memory::PtrHost<TestType> h_half(elements_fft);
        test::randomize(h_full_centered.get(), h_full_centered.elements(), randomizer_data);
        test::memset(h_half.get(), h_half.elements(), 0);
        cpu::fft::remap(fft::FC2H, h_full_centered.get(), stride, h_half.get(), stride_fft, shape, cpu_stream);

        AND_THEN("contiguous") {
            cuda::memory::PtrDevice<TestType> d_full_centered(elements);
            cuda::memory::PtrDevice<TestType> d_half(elements_fft);
            cpu::memory::PtrHost<TestType> h_half_cuda(elements_fft);

            cuda::memory::copy(h_full_centered.get(), d_full_centered.get(), h_full_centered.size(), gpu_stream);
            cuda::fft::remap(fft::FC2H, d_full_centered.get(), stride,
                             d_half.get(), stride_fft,
                             shape, gpu_stream);
            cuda::memory::copy(d_half.get(), h_half_cuda.get(), h_half.size(), gpu_stream);
            gpu_stream.synchronize();

            REQUIRE(test::Matcher(test::MATCH_ABS, h_half.get(), h_half_cuda.get(), elements_fft, 1e-14));
        }

        AND_THEN("padded") {
            cuda::memory::PtrDevicePadded<TestType> d_full_centered(shape);
            cuda::memory::PtrDevicePadded<TestType> d_half(shape_fft);
            cpu::memory::PtrHost<TestType> h_half_cuda(elements_fft);

            cuda::memory::copy(h_full_centered.get(), stride,
                               d_full_centered.get(), d_full_centered.strides(),
                               shape, gpu_stream);
            cuda::fft::remap(fft::FC2H, d_full_centered.get(), d_full_centered.strides(),
                             d_half.get(), d_half.strides(),
                             shape, gpu_stream);
            cuda::memory::copy(d_half.get(), d_half.strides(),
                               h_half_cuda.get(), stride_fft,
                               shape_fft, gpu_stream);
            gpu_stream.synchronize();

            REQUIRE(test::Matcher(test::MATCH_ABS, h_half.get(), h_half_cuda.get(), elements_fft, 1e-14));
        }
    }
}
