#include <noa/gpu/cuda/fourier/Resize.h>

#include <noa/cpu/fourier/Resize.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace Noa;

TEMPLATE_TEST_CASE("CUDA::Fourier: pad / crop", "[noa][cuda][fourier]", float, cfloat_t) {
    Test::IntRandomizer<size_t> randomizer(0, 15);
    Test::RealRandomizer<TestType> randomizer_real(-1., 1.);
    uint ndim = GENERATE(1U, 2U, 3U);

    size3_t shape = Test::getRandomShape(ndim);
    size3_t shape_fft = getShapeFFT(shape);
    size_t elements_fft = getElements(shape_fft);

    size3_t shape_padded(shape);
    if (ndim > 2) shape_padded.z += randomizer.get();
    if (ndim > 1) shape_padded.y += randomizer.get();
    shape_padded.x += randomizer.get();
    size3_t shape_fft_padded = getShapeFFT(shape_padded);
    size_t elements_fft_padded = getElements(shape_fft_padded);

    CUDA::Stream stream(CUDA::Stream::SERIAL);

    AND_THEN("no cropping") {
        Memory::PtrHost<TestType> h_in(elements_fft);
        Memory::PtrHost<TestType> h_out(elements_fft);
        CUDA::Memory::PtrDevicePadded<TestType> d_in(shape_fft);
        CUDA::Memory::PtrDevice<TestType> d_out(elements_fft);
        Memory::PtrHost<TestType> h_out_cuda(elements_fft);

        Test::initDataRandom(h_in.get(), h_in.elements(), randomizer_real);
        CUDA::Memory::copy(h_in.get(), shape_fft.x * sizeof(TestType), d_in.get(), d_in.pitch(), d_in.shape(), stream);
        CUDA::Fourier::crop(d_in.get(), shape, d_in.pitchElements(),
                            d_out.get(), shape, shape_fft.x, 1U, stream); // this should simply trigger a copy.
        CUDA::Memory::copy(d_out.get(), h_out_cuda.get(), d_out.bytes(), stream);
        Fourier::crop(h_in.get(), shape, h_out.get(), shape); // this should simply trigger a copy.
        CUDA::Stream::synchronize(stream);

        TestType diff = Test::getAverageDifference(h_out.get(), h_out_cuda.get(), h_out.elements());
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-14));
    }

    AND_THEN("no padding") {
        Memory::PtrHost<TestType> h_in(elements_fft);
        Memory::PtrHost<TestType> h_out(elements_fft);
        CUDA::Memory::PtrDevicePadded<TestType> d_in(shape_fft);
        CUDA::Memory::PtrDevice<TestType> d_out(elements_fft);
        Memory::PtrHost<TestType> h_out_cuda(elements_fft);

        Test::initDataRandom(h_in.get(), h_in.elements(), randomizer_real);
        CUDA::Memory::copy(h_in.get(), shape_fft.x * sizeof(TestType), d_in.get(), d_in.pitch(), d_in.shape(), stream);
        CUDA::Fourier::pad(d_in.get(), shape, d_in.pitchElements(),
                           d_out.get(), shape, shape_fft.x, 1U, stream); // this should simply trigger a copy.
        CUDA::Memory::copy(d_out.get(), h_out_cuda.get(), d_out.bytes(), stream);
        Fourier::pad(h_in.get(), shape, h_out.get(), shape); // this should simply trigger a copy.
        CUDA::Stream::synchronize(stream);

        TestType diff = Test::getAverageDifference(h_out.get(), h_out_cuda.get(), h_out.elements());
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-14));
    }

    AND_THEN("crop") {
        Memory::PtrHost<TestType> h_in(elements_fft_padded);
        Memory::PtrHost<TestType> h_out(elements_fft);
        CUDA::Memory::PtrDevice<TestType> d_in(elements_fft_padded);
        CUDA::Memory::PtrDevice<TestType> d_out(elements_fft);
        Memory::PtrHost<TestType> h_out_cuda(elements_fft);

        Test::initDataRandom(h_in.get(), h_in.elements(), randomizer_real);
        CUDA::Memory::copy(h_in.get(), d_in.get(), h_in.bytes(), stream);
        CUDA::Fourier::crop(d_in.get(), shape_padded, d_out.get(), shape, 1U, stream);
        CUDA::Memory::copy(d_out.get(), h_out_cuda.get(), d_out.bytes(), stream);
        Fourier::crop(h_in.get(), shape_padded, h_out.get(), shape);
        CUDA::Stream::synchronize(stream);

        TestType diff = Test::getAverageDifference(h_out.get(), h_out_cuda.get(), h_out.elements());
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-14));
    }

    AND_THEN("pad") {
        Memory::PtrHost<TestType> h_in(elements_fft);
        Memory::PtrHost<TestType> h_out(elements_fft_padded);
        CUDA::Memory::PtrDevice<TestType> d_in(elements_fft);
        CUDA::Memory::PtrDevice<TestType> d_out(elements_fft_padded);
        Memory::PtrHost<TestType> h_out_cuda(elements_fft_padded);

        Test::initDataRandom(h_in.get(), h_in.elements(), randomizer_real);
        CUDA::Memory::copy(h_in.get(), d_in.get(), h_in.bytes(), stream);
        CUDA::Fourier::pad(d_in.get(), shape, d_out.get(), shape_padded, 1U, stream);
        CUDA::Memory::copy(d_out.get(), h_out_cuda.get(), d_out.bytes(), stream);
        Fourier::pad(h_in.get(), shape, h_out.get(), shape_padded);
        CUDA::Stream::synchronize(stream);

        TestType diff = Test::getAverageDifference(h_out.get(), h_out_cuda.get(), h_out.elements());
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-14));
    }

    AND_THEN("crop padded") {
        Memory::PtrHost<TestType> h_in(elements_fft_padded);
        Memory::PtrHost<TestType> h_out(elements_fft);
        CUDA::Memory::PtrDevicePadded<TestType> d_in(shape_fft_padded);
        CUDA::Memory::PtrDevicePadded<TestType> d_out(shape_fft);
        Memory::PtrHost<TestType> h_out_cuda(elements_fft);

        Test::initDataRandom(h_in.get(), h_in.elements(), randomizer_real);
        CUDA::Memory::copy(h_in.get(), shape_fft_padded.x * sizeof(TestType),
                           d_in.get(), d_in.pitch(), d_in.shape(), stream);
        CUDA::Fourier::crop(d_in.get(), shape_padded, d_in.pitchElements(),
                            d_out.get(), shape, d_out.pitchElements(),
                            1U, stream);
        CUDA::Memory::copy(d_out.get(), d_out.pitch(), h_out_cuda.get(), shape_fft.x * sizeof(TestType),
                           shape_fft, stream);
        Fourier::crop(h_in.get(), shape_padded, h_out.get(), shape);
        CUDA::Stream::synchronize(stream);

        TestType diff = Test::getAverageDifference(h_out.get(), h_out_cuda.get(), h_out.elements());
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-14));
    }
}

TEMPLATE_TEST_CASE("CUDA::Fourier: padFull / cropFull", "[noa][cuda][fourier]", float, cfloat_t) {
    Test::IntRandomizer<size_t> randomizer(0, 15);
    Test::RealRandomizer<TestType> randomizer_real(-1., 1.);
    uint ndim = GENERATE(1U, 2U, 3U);

    size3_t shape = Test::getRandomShape(ndim);
    size_t elements = getElements(shape);

    size3_t shape_padded(shape);
    if (ndim > 2) shape_padded.z += randomizer.get();
    if (ndim > 1) shape_padded.y += randomizer.get();
    shape_padded.x += randomizer.get();
    size_t elements_padded = getElements(shape_padded);

    CUDA::Stream stream(CUDA::Stream::SERIAL);

    AND_THEN("no cropping") {
        Memory::PtrHost<TestType> h_in(elements);
        Memory::PtrHost<TestType> h_out(elements);
        CUDA::Memory::PtrDevicePadded<TestType> d_in(shape);
        CUDA::Memory::PtrDevice<TestType> d_out(elements);
        Memory::PtrHost<TestType> h_out_cuda(elements);

        Test::initDataRandom(h_in.get(), h_in.elements(), randomizer_real);
        CUDA::Memory::copy(h_in.get(), shape.x * sizeof(TestType), d_in.get(), d_in.pitch(), d_in.shape(), stream);
        CUDA::Fourier::cropFull(d_in.get(), shape, d_in.pitchElements(),
                                d_out.get(), shape, shape.x, 1U, stream); // this should simply trigger a copy.
        CUDA::Memory::copy(d_out.get(), h_out_cuda.get(), d_out.bytes(), stream);
        Fourier::cropFull(h_in.get(), shape, h_out.get(), shape); // this should simply trigger a copy.
        CUDA::Stream::synchronize(stream);

        TestType diff = Test::getAverageDifference(h_out.get(), h_out_cuda.get(), h_out.elements());
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-14));
    }

    AND_THEN("no padding") {
        Memory::PtrHost<TestType> h_in(elements);
        Memory::PtrHost<TestType> h_out(elements);
        CUDA::Memory::PtrDevicePadded<TestType> d_in(shape);
        CUDA::Memory::PtrDevice<TestType> d_out(elements);
        Memory::PtrHost<TestType> h_out_cuda(elements);

        Test::initDataRandom(h_in.get(), h_in.elements(), randomizer_real);
        CUDA::Memory::copy(h_in.get(), shape.x * sizeof(TestType), d_in.get(), d_in.pitch(), d_in.shape(), stream);
        CUDA::Fourier::padFull(d_in.get(), shape, d_in.pitchElements(),
                               d_out.get(), shape, shape.x, 1U, stream); // this should simply trigger a copy.
        CUDA::Memory::copy(d_out.get(), h_out_cuda.get(), d_out.bytes(), stream);
        Fourier::padFull(h_in.get(), shape, h_out.get(), shape); // this should simply trigger a copy.
        CUDA::Stream::synchronize(stream);

        TestType diff = Test::getAverageDifference(h_out.get(), h_out_cuda.get(), h_out.elements());
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-14));
    }

    AND_THEN("cropFull") {
        Memory::PtrHost<TestType> h_in(elements_padded);
        Memory::PtrHost<TestType> h_out(elements);
        CUDA::Memory::PtrDevice<TestType> d_in(elements_padded);
        CUDA::Memory::PtrDevice<TestType> d_out(elements);
        Memory::PtrHost<TestType> h_out_cuda(elements);

        Test::initDataRandom(h_in.get(), h_in.elements(), randomizer_real);
        CUDA::Memory::copy(h_in.get(), d_in.get(), h_in.bytes(), stream);
        CUDA::Fourier::cropFull(d_in.get(), shape_padded, d_out.get(), shape, 1U, stream);
        CUDA::Memory::copy(d_out.get(), h_out_cuda.get(), d_out.bytes(), stream);
        Fourier::cropFull(h_in.get(), shape_padded, h_out.get(), shape);
        CUDA::Stream::synchronize(stream);

        TestType diff = Test::getAverageDifference(h_out.get(), h_out_cuda.get(), h_out.elements());
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-14));
    }

    AND_THEN("padFull") {
        Memory::PtrHost<TestType> h_in(elements);
        Memory::PtrHost<TestType> h_out(elements_padded);
        CUDA::Memory::PtrDevice<TestType> d_in(elements);
        CUDA::Memory::PtrDevice<TestType> d_out(elements_padded);
        Memory::PtrHost<TestType> h_out_cuda(elements_padded);

        Test::initDataRandom(h_in.get(), h_in.elements(), randomizer_real);
        CUDA::Memory::copy(h_in.get(), d_in.get(), h_in.bytes(), stream);
        CUDA::Fourier::padFull(d_in.get(), shape, d_out.get(), shape_padded, 1U, stream);
        CUDA::Memory::copy(d_out.get(), h_out_cuda.get(), d_out.bytes(), stream);
        Fourier::padFull(h_in.get(), shape, h_out.get(), shape_padded);
        CUDA::Stream::synchronize(stream);

        TestType diff = Test::getAverageDifference(h_out.get(), h_out_cuda.get(), h_out.elements());
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-14));
    }

    AND_THEN("cropFull padded") {
        Memory::PtrHost<TestType> h_in(elements_padded);
        Memory::PtrHost<TestType> h_out(elements);
        CUDA::Memory::PtrDevicePadded<TestType> d_in(shape_padded);
        CUDA::Memory::PtrDevicePadded<TestType> d_out(shape);
        Memory::PtrHost<TestType> h_out_cuda(elements);

        Test::initDataRandom(h_in.get(), h_in.elements(), randomizer_real);
        CUDA::Memory::copy(h_in.get(), shape_padded.x * sizeof(TestType),
                           d_in.get(), d_in.pitch(), d_in.shape(), stream);
        CUDA::Fourier::cropFull(d_in.get(), shape_padded, d_in.pitchElements(),
                                d_out.get(), shape, d_out.pitchElements(),
                                1U, stream);
        CUDA::Memory::copy(d_out.get(), d_out.pitch(), h_out_cuda.get(), shape.x * sizeof(TestType), shape, stream);
        Fourier::cropFull(h_in.get(), shape_padded, h_out.get(), shape);
        CUDA::Stream::synchronize(stream);

        TestType diff = Test::getAverageDifference(h_out.get(), h_out_cuda.get(), h_out.elements());
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-14));
    }

    AND_THEN("padFull padded") {
        Memory::PtrHost<TestType> h_in(elements);
        Memory::PtrHost<TestType> h_out(elements_padded);
        CUDA::Memory::PtrDevicePadded<TestType> d_in(shape);
        CUDA::Memory::PtrDevicePadded<TestType> d_out(shape_padded);
        Memory::PtrHost<TestType> h_out_cuda(elements_padded);

        Test::initDataRandom(h_in.get(), h_in.elements(), randomizer_real);
        CUDA::Memory::copy(h_in.get(), shape.x * sizeof(TestType),
                           d_in.get(), d_in.pitch(), d_in.shape(), stream);
        CUDA::Fourier::padFull(d_in.get(), shape, d_in.pitchElements(),
                               d_out.get(), shape_padded, d_out.pitchElements(),
                               1U, stream);
        CUDA::Memory::copy(d_out.get(), d_out.pitch(), h_out_cuda.get(), shape_padded.x * sizeof(TestType),
                           shape_padded, stream);
        Fourier::padFull(h_in.get(), shape, h_out.get(), shape_padded);
        CUDA::Stream::synchronize(stream);

        TestType diff = Test::getAverageDifference(h_out.get(), h_out_cuda.get(), h_out.elements());
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-14));
    }
}
