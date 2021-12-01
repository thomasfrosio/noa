#include <noa/cpu/fft/Resize.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>
#include <noa/gpu/cuda/fft/Resize.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("cuda::fft::pad(), crop()", "[noa][cuda][fft]", float, cfloat_t, double, cdouble_t) {
    test::Randomizer<size_t> randomizer(0, 100);
    test::Randomizer<TestType> randomizer_real(-10., 10.);
    uint ndim = GENERATE(1U, 2U, 3U);

    size3_t shape = test::getRandomShape(ndim);
    size3_t shape_fft = shapeFFT(shape);
    size_t elements_fft = noa::elements(shape_fft);

    size3_t shape_padded(shape);
    if (ndim > 2) shape_padded.z += randomizer.get();
    if (ndim > 1) shape_padded.y += randomizer.get();
    shape_padded.x += randomizer.get();
    size3_t shape_fft_padded = shapeFFT(shape_padded);
    size_t elements_fft_padded = noa::elements(shape_fft_padded);

    cuda::Stream stream(cuda::Stream::SERIAL);

    AND_THEN("no cropping") {
        cpu::memory::PtrHost<TestType> h_in(elements_fft);
        cpu::memory::PtrHost<TestType> h_out(elements_fft);
        cuda::memory::PtrDevicePadded<TestType> d_in(shape_fft);
        cuda::memory::PtrDevice<TestType> d_out(elements_fft);
        cpu::memory::PtrHost<TestType> h_out_cuda(elements_fft);

        test::randomize(h_in.get(), h_in.elements(), randomizer_real);
        cuda::memory::copy(h_in.get(), shape_fft.x, d_in.get(), d_in.pitch(), d_in.shape(), stream);
        cuda::fft::crop(d_in.get(), d_in.pitch(), shape,
                        d_out.get(), shape_fft.x, shape, 1U, stream); // this should simply trigger a copy.
        cuda::memory::copy(d_out.get(), h_out_cuda.get(), d_out.size(), stream);
        cpu::fft::crop(h_in.get(), shape, h_out.get(), shape, 1); // this should simply trigger a copy.
        cuda::Stream::synchronize(stream);

        TestType diff = test::getAverageDifference(h_out.get(), h_out_cuda.get(), h_out.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
    }

    AND_THEN("no padding") {
        cpu::memory::PtrHost<TestType> h_in(elements_fft);
        cpu::memory::PtrHost<TestType> h_out(elements_fft);
        cuda::memory::PtrDevicePadded<TestType> d_in(shape_fft);
        cuda::memory::PtrDevice<TestType> d_out(elements_fft);
        cpu::memory::PtrHost<TestType> h_out_cuda(elements_fft);

        test::randomize(h_in.get(), h_in.elements(), randomizer_real);
        cuda::memory::copy(h_in.get(), shape_fft.x, d_in.get(), d_in.pitch(), d_in.shape(), stream);
        cuda::fft::pad(d_in.get(), d_in.pitch(), shape,
                       d_out.get(), shape_fft.x, shape, 1U, stream); // this should simply trigger a copy.
        cuda::memory::copy(d_out.get(), h_out_cuda.get(), d_out.size(), stream);
        cpu::fft::pad(h_in.get(), shape, h_out.get(), shape, 1); // this should simply trigger a copy.
        cuda::Stream::synchronize(stream);

        TestType diff = test::getAverageDifference(h_out.get(), h_out_cuda.get(), h_out.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
    }

    AND_THEN("crop") {
        cpu::memory::PtrHost<TestType> h_in(elements_fft_padded);
        cpu::memory::PtrHost<TestType> h_out(elements_fft);
        cuda::memory::PtrDevice<TestType> d_in(elements_fft_padded);
        cuda::memory::PtrDevice<TestType> d_out(elements_fft);
        cpu::memory::PtrHost<TestType> h_out_cuda(elements_fft);

        test::randomize(h_in.get(), h_in.elements(), randomizer_real);
        cuda::memory::copy(h_in.get(), d_in.get(), h_in.size(), stream);
        cuda::fft::crop(d_in.get(), shape_padded.x / 2 + 1, shape_padded,
                        d_out.get(), shape.x / 2 + 1, shape, 1, stream);
        cuda::memory::copy(d_out.get(), h_out_cuda.get(), d_out.size(), stream);
        cpu::fft::crop(h_in.get(), shape_padded, h_out.get(), shape, 1);
        cuda::Stream::synchronize(stream);

        TestType diff = test::getAverageDifference(h_out.get(), h_out_cuda.get(), h_out.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
    }

    AND_THEN("pad") {
        cpu::memory::PtrHost<TestType> h_in(elements_fft);
        cpu::memory::PtrHost<TestType> h_out(elements_fft_padded);
        cuda::memory::PtrDevice<TestType> d_in(elements_fft);
        cuda::memory::PtrDevice<TestType> d_out(elements_fft_padded);
        cpu::memory::PtrHost<TestType> h_out_cuda(elements_fft_padded);

        test::randomize(h_in.get(), h_in.elements(), randomizer_real);
        cuda::memory::copy(h_in.get(), d_in.get(), h_in.size(), stream);
        cuda::fft::pad(d_in.get(), shape.x / 2 + 1, shape,
                       d_out.get(), shape_padded.x / 2 + 1, shape_padded, 1U, stream);
        cuda::memory::copy(d_out.get(), h_out_cuda.get(), d_out.size(), stream);
        cpu::fft::pad(h_in.get(), shape, h_out.get(), shape_padded, 1);
        cuda::Stream::synchronize(stream);

        TestType diff = test::getAverageDifference(h_out.get(), h_out_cuda.get(), h_out.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
    }

    AND_THEN("crop padded") {
        cpu::memory::PtrHost<TestType> h_in(elements_fft_padded);
        cpu::memory::PtrHost<TestType> h_out(elements_fft);
        cuda::memory::PtrDevicePadded<TestType> d_in(shape_fft_padded);
        cuda::memory::PtrDevicePadded<TestType> d_out(shape_fft);
        cpu::memory::PtrHost<TestType> h_out_cuda(elements_fft);

        test::randomize(h_in.get(), h_in.elements(), randomizer_real);
        cuda::memory::copy(h_in.get(), shape_fft_padded.x, d_in.get(), d_in.pitch(), d_in.shape(), stream);
        cuda::fft::crop(d_in.get(), d_in.pitch(), shape_padded, d_out.get(), d_out.pitch(), shape, 1U, stream);
        cuda::memory::copy(d_out.get(), d_out.pitch(), h_out_cuda.get(), shape_fft.x, shape_fft, stream);
        cpu::fft::crop(h_in.get(), shape_padded, h_out.get(), shape, 1);
        cuda::Stream::synchronize(stream);

        TestType diff = test::getAverageDifference(h_out.get(), h_out_cuda.get(), h_out.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
    }
}

TEMPLATE_TEST_CASE("cuda::fft::padFull(), cropFull()", "[noa][cuda][fft]", float, cfloat_t, double, cdouble_t) {
    test::Randomizer<size_t> randomizer(0, 50);
    test::Randomizer<TestType> randomizer_real(-10., 10.);
    uint ndim = GENERATE(1U, 2U, 3U);

    size3_t shape = test::getRandomShape(ndim);
    size_t elements = noa::elements(shape);

    size3_t shape_padded(shape);
    if (ndim > 2) shape_padded.z += randomizer.get();
    if (ndim > 1) shape_padded.y += randomizer.get();
    shape_padded.x += randomizer.get();
    size_t elements_padded = noa::elements(shape_padded);

    cuda::Stream stream(cuda::Stream::SERIAL);

    AND_THEN("no cropping") {
        cpu::memory::PtrHost<TestType> h_in(elements);
        cpu::memory::PtrHost<TestType> h_out(elements);
        cuda::memory::PtrDevicePadded<TestType> d_in(shape);
        cuda::memory::PtrDevice<TestType> d_out(elements);
        cpu::memory::PtrHost<TestType> h_out_cuda(elements);

        test::randomize(h_in.get(), h_in.elements(), randomizer_real);
        cuda::memory::copy(h_in.get(), shape.x, d_in.get(), d_in.pitch(), d_in.shape(), stream);
        cuda::fft::cropFull(d_in.get(), d_in.pitch(), shape, d_out.get(), shape.x, shape, 1U,
                            stream); // this should simply trigger a copy.
        cuda::memory::copy(d_out.get(), h_out_cuda.get(), d_out.size(), stream);
        cpu::fft::cropFull(h_in.get(), shape, h_out.get(), shape, 1); // triggers a copy.
        cuda::Stream::synchronize(stream);

        TestType diff = test::getAverageDifference(h_out.get(), h_out_cuda.get(), h_out.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
    }

    AND_THEN("no padding") {
        cpu::memory::PtrHost<TestType> h_in(elements);
        cpu::memory::PtrHost<TestType> h_out(elements);
        cuda::memory::PtrDevicePadded<TestType> d_in(shape);
        cuda::memory::PtrDevice<TestType> d_out(elements);
        cpu::memory::PtrHost<TestType> h_out_cuda(elements);

        test::randomize(h_in.get(), h_in.elements(), randomizer_real);
        cuda::memory::copy(h_in.get(), shape.x, d_in.get(), d_in.pitch(), d_in.shape(), stream);
        cuda::fft::padFull(d_in.get(), d_in.pitch(), shape, d_out.get(), shape.x, shape, 1U,
                           stream); // this should simply trigger a copy.
        cuda::memory::copy(d_out.get(), h_out_cuda.get(), d_out.size(), stream);
        cpu::fft::padFull(h_in.get(), shape, h_out.get(), shape, 1); // this should simply trigger a copy.
        cuda::Stream::synchronize(stream);

        TestType diff = test::getAverageDifference(h_out.get(), h_out_cuda.get(), h_out.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
    }

    AND_THEN("cropFull") {
        cpu::memory::PtrHost<TestType> h_in(elements_padded);
        cpu::memory::PtrHost<TestType> h_out(elements);
        cuda::memory::PtrDevice<TestType> d_in(elements_padded);
        cuda::memory::PtrDevice<TestType> d_out(elements);
        cpu::memory::PtrHost<TestType> h_out_cuda(elements);

        test::randomize(h_in.get(), h_in.elements(), randomizer_real);
        cuda::memory::copy(h_in.get(), d_in.get(), h_in.size(), stream);
        cuda::fft::cropFull(d_in.get(), shape_padded.x, shape_padded, d_out.get(), shape.x, shape, 1, stream);
        cuda::memory::copy(d_out.get(), h_out_cuda.get(), d_out.size(), stream);
        cpu::fft::cropFull(h_in.get(), shape_padded, h_out.get(), shape, 1);
        cuda::Stream::synchronize(stream);

        TestType diff = test::getAverageDifference(h_out.get(), h_out_cuda.get(), h_out.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
    }

    AND_THEN("padFull") {
        cpu::memory::PtrHost<TestType> h_in(elements);
        cpu::memory::PtrHost<TestType> h_out(elements_padded);
        cuda::memory::PtrDevice<TestType> d_in(elements);
        cuda::memory::PtrDevice<TestType> d_out(elements_padded);
        cpu::memory::PtrHost<TestType> h_out_cuda(elements_padded);

        test::randomize(h_in.get(), h_in.elements(), randomizer_real);
        cuda::memory::copy(h_in.get(), d_in.get(), h_in.size(), stream);
        cuda::fft::padFull(d_in.get(), shape.x, shape, d_out.get(), shape_padded.x, shape_padded, 1U, stream);
        cuda::memory::copy(d_out.get(), h_out_cuda.get(), d_out.size(), stream);
        cpu::fft::padFull(h_in.get(), shape, h_out.get(), shape_padded, 1);
        cuda::Stream::synchronize(stream);

        TestType diff = test::getAverageDifference(h_out.get(), h_out_cuda.get(), h_out.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
    }

    AND_THEN("cropFull padded") {
        cpu::memory::PtrHost<TestType> h_in(elements_padded);
        cpu::memory::PtrHost<TestType> h_out(elements);
        cuda::memory::PtrDevicePadded<TestType> d_in(shape_padded);
        cuda::memory::PtrDevicePadded<TestType> d_out(shape);
        cpu::memory::PtrHost<TestType> h_out_cuda(elements);

        test::randomize(h_in.get(), h_in.elements(), randomizer_real);
        cuda::memory::copy(h_in.get(), shape_padded.x, d_in.get(), d_in.pitch(), d_in.shape(), stream);
        cuda::fft::cropFull(d_in.get(), d_in.pitch(), shape_padded, d_out.get(), d_out.pitch(), shape, 1U, stream);
        cuda::memory::copy(d_out.get(), d_out.pitch(), h_out_cuda.get(), shape.x, shape, stream);
        cpu::fft::cropFull(h_in.get(), shape_padded, h_out.get(), shape, 1);
        cuda::Stream::synchronize(stream);

        TestType diff = test::getAverageDifference(h_out.get(), h_out_cuda.get(), h_out.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
    }

    AND_THEN("padFull padded") {
        cpu::memory::PtrHost<TestType> h_in(elements);
        cpu::memory::PtrHost<TestType> h_out(elements_padded);
        cuda::memory::PtrDevicePadded<TestType> d_in(shape);
        cuda::memory::PtrDevicePadded<TestType> d_out(shape_padded);
        cpu::memory::PtrHost<TestType> h_out_cuda(elements_padded);

        test::randomize(h_in.get(), h_in.elements(), randomizer_real);
        cuda::memory::copy(h_in.get(), shape.x, d_in.get(), d_in.pitch(), d_in.shape(), stream);
        cuda::fft::padFull(d_in.get(), d_in.pitch(), shape, d_out.get(), d_out.pitch(), shape_padded, 1U, stream);
        cuda::memory::copy(d_out.get(), d_out.pitch(), h_out_cuda.get(), shape_padded.x, shape_padded, stream);
        cpu::fft::padFull(h_in.get(), shape, h_out.get(), shape_padded, 1);
        cuda::Stream::synchronize(stream);

        TestType diff = test::getAverageDifference(h_out.get(), h_out_cuda.get(), h_out.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
    }
}
