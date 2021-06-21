#include <noa/gpu/cuda/fourier/Resize.h>

#include <noa/cpu/fourier/Resize.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("cuda::fourier: pad / crop", "[noa][cuda][fourier]", float, cfloat_t, double, cdouble_t) {
    test::IntRandomizer<size_t> randomizer(0, 15);
    test::RealRandomizer<TestType> randomizer_real(-1., 1.);
    uint ndim = GENERATE(1U, 2U, 3U);

    size3_t shape = test::getRandomShape(ndim);
    size3_t shape_fft = getShapeFFT(shape);
    size_t elements_fft = getElements(shape_fft);

    size3_t shape_padded(shape);
    if (ndim > 2) shape_padded.z += randomizer.get();
    if (ndim > 1) shape_padded.y += randomizer.get();
    shape_padded.x += randomizer.get();
    size3_t shape_fft_padded = getShapeFFT(shape_padded);
    size_t elements_fft_padded = getElements(shape_fft_padded);

    cuda::Stream stream(cuda::Stream::SERIAL);

    AND_THEN("no cropping") {
        memory::PtrHost<TestType> h_in(elements_fft);
        memory::PtrHost<TestType> h_out(elements_fft);
        cuda::memory::PtrDevicePadded<TestType> d_in(shape_fft);
        cuda::memory::PtrDevice<TestType> d_out(elements_fft);
        memory::PtrHost<TestType> h_out_cuda(elements_fft);

        test::initDataRandom(h_in.get(), h_in.elements(), randomizer_real);
        cuda::memory::copy(h_in.get(), shape_fft.x, d_in.get(), d_in.pitch(), d_in.shape(), stream);
        cuda::fourier::crop(d_in.get(), d_in.pitch(), shape,
                            d_out.get(), shape_fft.x, shape, 1U, stream); // this should simply trigger a copy.
        cuda::memory::copy(d_out.get(), h_out_cuda.get(), d_out.size(), stream);
        fourier::crop(h_in.get(), shape, h_out.get(), shape); // this should simply trigger a copy.
        cuda::Stream::synchronize(stream);

        TestType diff = test::getAverageDifference(h_out.get(), h_out_cuda.get(), h_out.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
    }

    AND_THEN("no padding") {
        memory::PtrHost<TestType> h_in(elements_fft);
        memory::PtrHost<TestType> h_out(elements_fft);
        cuda::memory::PtrDevicePadded<TestType> d_in(shape_fft);
        cuda::memory::PtrDevice<TestType> d_out(elements_fft);
        memory::PtrHost<TestType> h_out_cuda(elements_fft);

        test::initDataRandom(h_in.get(), h_in.elements(), randomizer_real);
        cuda::memory::copy(h_in.get(), shape_fft.x, d_in.get(), d_in.pitch(), d_in.shape(), stream);
        cuda::fourier::pad(d_in.get(), d_in.pitch(), shape,
                           d_out.get(), shape_fft.x, shape, 1U, stream); // this should simply trigger a copy.
        cuda::memory::copy(d_out.get(), h_out_cuda.get(), d_out.size(), stream);
        fourier::pad(h_in.get(), shape, h_out.get(), shape); // this should simply trigger a copy.
        cuda::Stream::synchronize(stream);

        TestType diff = test::getAverageDifference(h_out.get(), h_out_cuda.get(), h_out.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
    }

    AND_THEN("crop") {
        memory::PtrHost<TestType> h_in(elements_fft_padded);
        memory::PtrHost<TestType> h_out(elements_fft);
        cuda::memory::PtrDevice<TestType> d_in(elements_fft_padded);
        cuda::memory::PtrDevice<TestType> d_out(elements_fft);
        memory::PtrHost<TestType> h_out_cuda(elements_fft);

        test::initDataRandom(h_in.get(), h_in.elements(), randomizer_real);
        cuda::memory::copy(h_in.get(), d_in.get(), h_in.size(), stream);
        cuda::fourier::crop(d_in.get(), shape_padded, d_out.get(), shape, 1U, stream);
        cuda::memory::copy(d_out.get(), h_out_cuda.get(), d_out.size(), stream);
        fourier::crop(h_in.get(), shape_padded, h_out.get(), shape);
        cuda::Stream::synchronize(stream);

        TestType diff = test::getAverageDifference(h_out.get(), h_out_cuda.get(), h_out.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
    }

    AND_THEN("pad") {
        memory::PtrHost<TestType> h_in(elements_fft);
        memory::PtrHost<TestType> h_out(elements_fft_padded);
        cuda::memory::PtrDevice<TestType> d_in(elements_fft);
        cuda::memory::PtrDevice<TestType> d_out(elements_fft_padded);
        memory::PtrHost<TestType> h_out_cuda(elements_fft_padded);

        test::initDataRandom(h_in.get(), h_in.elements(), randomizer_real);
        cuda::memory::copy(h_in.get(), d_in.get(), h_in.size(), stream);
        cuda::fourier::pad(d_in.get(), shape, d_out.get(), shape_padded, 1U, stream);
        cuda::memory::copy(d_out.get(), h_out_cuda.get(), d_out.size(), stream);
        fourier::pad(h_in.get(), shape, h_out.get(), shape_padded);
        cuda::Stream::synchronize(stream);

        TestType diff = test::getAverageDifference(h_out.get(), h_out_cuda.get(), h_out.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
    }

    AND_THEN("crop padded") {
        memory::PtrHost<TestType> h_in(elements_fft_padded);
        memory::PtrHost<TestType> h_out(elements_fft);
        cuda::memory::PtrDevicePadded<TestType> d_in(shape_fft_padded);
        cuda::memory::PtrDevicePadded<TestType> d_out(shape_fft);
        memory::PtrHost<TestType> h_out_cuda(elements_fft);

        test::initDataRandom(h_in.get(), h_in.elements(), randomizer_real);
        cuda::memory::copy(h_in.get(), shape_fft_padded.x, d_in.get(), d_in.pitch(), d_in.shape(), stream);
        cuda::fourier::crop(d_in.get(), d_in.pitch(), shape_padded, d_out.get(), d_out.pitch(), shape, 1U, stream);
        cuda::memory::copy(d_out.get(), d_out.pitch(), h_out_cuda.get(), shape_fft.x, shape_fft, stream);
        fourier::crop(h_in.get(), shape_padded, h_out.get(), shape);
        cuda::Stream::synchronize(stream);

        TestType diff = test::getAverageDifference(h_out.get(), h_out_cuda.get(), h_out.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
    }
}

TEMPLATE_TEST_CASE("cuda::fourier: padFull / cropFull", "[noa][cuda][fourier]", float, cfloat_t, double, cdouble_t) {
    test::IntRandomizer<size_t> randomizer(0, 15);
    test::RealRandomizer<TestType> randomizer_real(-1., 1.);
    uint ndim = GENERATE(1U, 2U, 3U);

    size3_t shape = test::getRandomShape(ndim);
    size_t elements = getElements(shape);

    size3_t shape_padded(shape);
    if (ndim > 2) shape_padded.z += randomizer.get();
    if (ndim > 1) shape_padded.y += randomizer.get();
    shape_padded.x += randomizer.get();
    size_t elements_padded = getElements(shape_padded);

    cuda::Stream stream(cuda::Stream::SERIAL);

    AND_THEN("no cropping") {
        memory::PtrHost<TestType> h_in(elements);
        memory::PtrHost<TestType> h_out(elements);
        cuda::memory::PtrDevicePadded<TestType> d_in(shape);
        cuda::memory::PtrDevice<TestType> d_out(elements);
        memory::PtrHost<TestType> h_out_cuda(elements);

        test::initDataRandom(h_in.get(), h_in.elements(), randomizer_real);
        cuda::memory::copy(h_in.get(), shape.x, d_in.get(), d_in.pitch(), d_in.shape(), stream);
        cuda::fourier::cropFull(d_in.get(), d_in.pitch(), shape, d_out.get(), shape.x, shape, 1U,
                                stream); // this should simply trigger a copy.
        cuda::memory::copy(d_out.get(), h_out_cuda.get(), d_out.size(), stream);
        fourier::cropFull(h_in.get(), shape, h_out.get(), shape); // this should simply trigger a copy.
        cuda::Stream::synchronize(stream);

        TestType diff = test::getAverageDifference(h_out.get(), h_out_cuda.get(), h_out.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
    }

    AND_THEN("no padding") {
        memory::PtrHost<TestType> h_in(elements);
        memory::PtrHost<TestType> h_out(elements);
        cuda::memory::PtrDevicePadded<TestType> d_in(shape);
        cuda::memory::PtrDevice<TestType> d_out(elements);
        memory::PtrHost<TestType> h_out_cuda(elements);

        test::initDataRandom(h_in.get(), h_in.elements(), randomizer_real);
        cuda::memory::copy(h_in.get(), shape.x, d_in.get(), d_in.pitch(), d_in.shape(), stream);
        cuda::fourier::padFull(d_in.get(), d_in.pitch(), shape, d_out.get(), shape.x, shape, 1U,
                               stream); // this should simply trigger a copy.
        cuda::memory::copy(d_out.get(), h_out_cuda.get(), d_out.size(), stream);
        fourier::padFull(h_in.get(), shape, h_out.get(), shape); // this should simply trigger a copy.
        cuda::Stream::synchronize(stream);

        TestType diff = test::getAverageDifference(h_out.get(), h_out_cuda.get(), h_out.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
    }

    AND_THEN("cropFull") {
        memory::PtrHost<TestType> h_in(elements_padded);
        memory::PtrHost<TestType> h_out(elements);
        cuda::memory::PtrDevice<TestType> d_in(elements_padded);
        cuda::memory::PtrDevice<TestType> d_out(elements);
        memory::PtrHost<TestType> h_out_cuda(elements);

        test::initDataRandom(h_in.get(), h_in.elements(), randomizer_real);
        cuda::memory::copy(h_in.get(), d_in.get(), h_in.size(), stream);
        cuda::fourier::cropFull(d_in.get(), shape_padded, d_out.get(), shape, 1U, stream);
        cuda::memory::copy(d_out.get(), h_out_cuda.get(), d_out.size(), stream);
        fourier::cropFull(h_in.get(), shape_padded, h_out.get(), shape);
        cuda::Stream::synchronize(stream);

        TestType diff = test::getAverageDifference(h_out.get(), h_out_cuda.get(), h_out.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
    }

    AND_THEN("padFull") {
        memory::PtrHost<TestType> h_in(elements);
        memory::PtrHost<TestType> h_out(elements_padded);
        cuda::memory::PtrDevice<TestType> d_in(elements);
        cuda::memory::PtrDevice<TestType> d_out(elements_padded);
        memory::PtrHost<TestType> h_out_cuda(elements_padded);

        test::initDataRandom(h_in.get(), h_in.elements(), randomizer_real);
        cuda::memory::copy(h_in.get(), d_in.get(), h_in.size(), stream);
        cuda::fourier::padFull(d_in.get(), shape, d_out.get(), shape_padded, 1U, stream);
        cuda::memory::copy(d_out.get(), h_out_cuda.get(), d_out.size(), stream);
        fourier::padFull(h_in.get(), shape, h_out.get(), shape_padded);
        cuda::Stream::synchronize(stream);

        TestType diff = test::getAverageDifference(h_out.get(), h_out_cuda.get(), h_out.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
    }

    AND_THEN("cropFull padded") {
        memory::PtrHost<TestType> h_in(elements_padded);
        memory::PtrHost<TestType> h_out(elements);
        cuda::memory::PtrDevicePadded<TestType> d_in(shape_padded);
        cuda::memory::PtrDevicePadded<TestType> d_out(shape);
        memory::PtrHost<TestType> h_out_cuda(elements);

        test::initDataRandom(h_in.get(), h_in.elements(), randomizer_real);
        cuda::memory::copy(h_in.get(), shape_padded.x, d_in.get(), d_in.pitch(), d_in.shape(), stream);
        cuda::fourier::cropFull(d_in.get(), d_in.pitch(), shape_padded, d_out.get(), d_out.pitch(), shape, 1U, stream);
        cuda::memory::copy(d_out.get(), d_out.pitch(), h_out_cuda.get(), shape.x, shape, stream);
        fourier::cropFull(h_in.get(), shape_padded, h_out.get(), shape);
        cuda::Stream::synchronize(stream);

        TestType diff = test::getAverageDifference(h_out.get(), h_out_cuda.get(), h_out.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
    }

    AND_THEN("padFull padded") {
        memory::PtrHost<TestType> h_in(elements);
        memory::PtrHost<TestType> h_out(elements_padded);
        cuda::memory::PtrDevicePadded<TestType> d_in(shape);
        cuda::memory::PtrDevicePadded<TestType> d_out(shape_padded);
        memory::PtrHost<TestType> h_out_cuda(elements_padded);

        test::initDataRandom(h_in.get(), h_in.elements(), randomizer_real);
        cuda::memory::copy(h_in.get(), shape.x, d_in.get(), d_in.pitch(), d_in.shape(), stream);
        cuda::fourier::padFull(d_in.get(), d_in.pitch(), shape, d_out.get(), d_out.pitch(), shape_padded, 1U, stream);
        cuda::memory::copy(d_out.get(), d_out.pitch(), h_out_cuda.get(), shape_padded.x, shape_padded, stream);
        fourier::padFull(h_in.get(), shape, h_out.get(), shape_padded);
        cuda::Stream::synchronize(stream);

        TestType diff = test::getAverageDifference(h_out.get(), h_out_cuda.get(), h_out.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
    }
}
