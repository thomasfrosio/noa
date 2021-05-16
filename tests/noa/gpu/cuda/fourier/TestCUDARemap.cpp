#include <noa/gpu/cuda/fourier/Remap.h>

#include <noa/cpu/fourier/Remap.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace Noa;

TEMPLATE_TEST_CASE("CUDA::Fourier: H2F <-> F2H", "[noa][cuda][fourier]", float, cfloat_t) {
    Test::RealRandomizer<TestType> randomizer_data(1., 128.);

    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape = Test::getRandomShape(ndim);
    size3_t shape_fft = getShapeFFT(shape);
    size_t elements = getElements(shape);
    size_t elements_fft = getElements(shape_fft);

    CUDA::Stream stream(CUDA::Stream::SERIAL);
    INFO(shape);

    AND_THEN("H2F") {
        Memory::PtrHost<TestType> h_half(elements_fft);
        Memory::PtrHost<TestType> h_full(elements);
        Test::initDataRandom(h_half.get(), h_half.elements(), randomizer_data);
        Test::initDataZero(h_full.get(), h_full.elements());
        Fourier::H2F(h_half.get(), h_full.get(), shape);

        AND_THEN("contiguous") {
            CUDA::Memory::PtrDevice<TestType> d_half(elements_fft);
            CUDA::Memory::PtrDevice<TestType> d_full(elements);
            Memory::PtrHost<TestType> h_full_cuda(elements);

            CUDA::Memory::copy(h_half.get(), d_half.get(), h_half.size(), stream);
            CUDA::Fourier::H2F(d_half.get(), d_full.get(), shape, 1, stream);
            CUDA::Memory::copy(d_full.get(), h_full_cuda.get(), d_full.size(), stream);
            CUDA::Stream::synchronize(stream);

            TestType diff = Test::getAverageDifference(h_full.get(), h_full_cuda.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-14));
        }

        AND_THEN("padded") {
            CUDA::Memory::PtrDevicePadded<TestType> d_half(shape_fft);
            CUDA::Memory::PtrDevicePadded<TestType> d_full(shape);
            Memory::PtrHost<TestType> h_full_cuda(elements);

            CUDA::Memory::copy(h_half.get(), shape_fft.x, d_half.get(), d_half.pitch(), shape_fft, stream);
            CUDA::Fourier::H2F(d_half.get(), d_half.pitch(), d_full.get(), d_full.pitch(), shape, 1, stream);
            CUDA::Memory::copy(d_full.get(), d_full.pitch(), h_full_cuda.get(), shape.x, shape, stream);
            CUDA::Stream::synchronize(stream);

            TestType diff = Test::getAverageDifference(h_full.get(), h_full_cuda.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-14));
        }
    }

    AND_THEN("F2H") {
        Memory::PtrHost<TestType> h_full(elements);
        Memory::PtrHost<TestType> h_half(elements_fft);
        Test::initDataRandom(h_full.get(), h_full.elements(), randomizer_data);
        Test::initDataZero(h_half.get(), h_half.elements());
        Fourier::F2H(h_full.get(), h_half.get(), shape);

        AND_THEN("contiguous") {
            CUDA::Memory::PtrDevice<TestType> d_full(elements);
            CUDA::Memory::PtrDevice<TestType> d_half(elements_fft);
            Memory::PtrHost<TestType> h_half_cuda(elements_fft);

            CUDA::Memory::copy(h_full.get(), d_full.get(), h_full.size(), stream);
            CUDA::Fourier::F2H(d_full.get(), d_half.get(), shape, 1, stream);
            CUDA::Memory::copy(d_half.get(), h_half_cuda.get(), d_half.size(), stream);
            CUDA::Stream::synchronize(stream);

            TestType diff = Test::getAverageDifference(h_half.get(), h_half_cuda.get(), elements_fft);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-14));
        }

        AND_THEN("padded") {
            CUDA::Memory::PtrDevicePadded<TestType> d_full(shape);
            CUDA::Memory::PtrDevicePadded<TestType> d_half(shape_fft);
            Memory::PtrHost<TestType> h_half_cuda(elements_fft);

            CUDA::Memory::copy(h_full.get(), shape.x, d_full.get(), d_full.pitch(), shape, stream);
            CUDA::Fourier::F2H(d_full.get(), d_full.pitch(), d_half.get(), d_half.pitch(), shape, 1, stream);
            CUDA::Memory::copy(d_half.get(), d_half.pitch(), h_half_cuda.get(), shape_fft.x, shape_fft, stream);
            CUDA::Stream::synchronize(stream);

            TestType diff = Test::getAverageDifference(h_half.get(), h_half_cuda.get(), elements_fft);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-14));
        }
    }
}

TEMPLATE_TEST_CASE("CUDA::Fourier: F2FC <-> FC2F", "[noa][cuda][fourier]", float, cfloat_t) {
    Test::RealRandomizer<TestType> randomizer_data(1., 128.);

    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape = Test::getRandomShape(ndim);
    size_t elements = getElements(shape);

    CUDA::Stream stream(CUDA::Stream::SERIAL);
    INFO(shape);

    AND_THEN("F2FC") {
        Memory::PtrHost<TestType> h_full(elements);
        Memory::PtrHost<TestType> h_full_centered(elements);
        Test::initDataRandom(h_full.get(), h_full.elements(), randomizer_data);
        Test::initDataZero(h_full_centered.get(), h_full_centered.elements());
        Fourier::F2FC(h_full.get(), h_full_centered.get(), shape);

        AND_THEN("contiguous") {
            CUDA::Memory::PtrDevice<TestType> d_full(elements);
            CUDA::Memory::PtrDevice<TestType> d_full_centered(elements);
            Memory::PtrHost<TestType> h_full_centered_cuda(elements);

            CUDA::Memory::copy(h_full.get(), d_full.get(), h_full.size(), stream);
            CUDA::Fourier::F2FC(d_full.get(), d_full_centered.get(), shape, 1, stream);
            CUDA::Memory::copy(d_full_centered.get(), h_full_centered_cuda.get(), h_full.size(), stream);
            CUDA::Stream::synchronize(stream);

            TestType diff = Test::getAverageDifference(h_full_centered.get(), h_full_centered_cuda.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-14));
        }

        AND_THEN("padded") {
            CUDA::Memory::PtrDevicePadded<TestType> d_full(shape);
            CUDA::Memory::PtrDevicePadded<TestType> d_full_centered(shape);
            Memory::PtrHost<TestType> h_full_centered_cuda(elements);

            CUDA::Memory::copy(h_full.get(), shape.x,
                               d_full.get(), d_full.pitch(), shape, stream);
            CUDA::Fourier::F2FC(d_full.get(), d_full.pitch(),
                                d_full_centered.get(), d_full_centered.pitch(),
                                shape, 1, stream);
            CUDA::Memory::copy(d_full_centered.get(), d_full_centered.pitch(),
                               h_full_centered_cuda.get(), shape.x, shape, stream);
            CUDA::Stream::synchronize(stream);

            TestType diff = Test::getAverageDifference(h_full_centered.get(), h_full_centered_cuda.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-14));
        }
    }

    AND_THEN("F2FC") {
        Memory::PtrHost<TestType> h_full_centered(elements);
        Memory::PtrHost<TestType> h_full(elements);
        Test::initDataRandom(h_full_centered.get(), h_full_centered.elements(), randomizer_data);
        Test::initDataZero(h_full.get(), h_full.elements());
        Fourier::FC2F(h_full_centered.get(), h_full.get(), shape);

        AND_THEN("contiguous") {
            CUDA::Memory::PtrDevice<TestType> d_full_centered(elements);
            CUDA::Memory::PtrDevice<TestType> d_full(elements);
            Memory::PtrHost<TestType> h_full_cuda(elements);

            CUDA::Memory::copy(h_full_centered.get(), d_full_centered.get(), d_full_centered.elements(), stream);
            CUDA::Fourier::FC2F(d_full_centered.get(), d_full.get(), shape, 1, stream);
            CUDA::Memory::copy(d_full.get(), h_full_cuda.get(), h_full.elements(), stream);
            CUDA::Stream::synchronize(stream);

            TestType diff = Test::getAverageDifference(h_full.get(), h_full_cuda.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-14));
        }

        AND_THEN("padded") {
            CUDA::Memory::PtrDevicePadded<TestType> d_full_centered(shape);
            CUDA::Memory::PtrDevicePadded<TestType> d_full(shape);
            Memory::PtrHost<TestType> h_full_cuda(elements);

            CUDA::Memory::copy(h_full_centered.get(), shape.x,
                               d_full_centered.get(), d_full_centered.pitch(), shape, stream);
            CUDA::Fourier::FC2F(d_full_centered.get(), d_full_centered.pitch(),
                                d_full.get(), d_full.pitch(),
                                shape, 1, stream);
            CUDA::Memory::copy(d_full.get(), d_full.pitch(), h_full_cuda.get(), shape.x, shape, stream);
            CUDA::Stream::synchronize(stream);

            TestType diff = Test::getAverageDifference(h_full.get(), h_full_cuda.get(), elements);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-14));
        }
    }
}

TEMPLATE_TEST_CASE("CUDA::Fourier: H2HC <-> HC2H", "[noa][cuda][fourier]", float, cfloat_t) {
    Test::RealRandomizer<TestType> randomizer_data(1., 128.);

    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape = Test::getRandomShape(ndim);
    size3_t shape_fft = getShapeFFT(shape);
    size_t elements_fft = getElements(shape_fft);

    CUDA::Stream stream(CUDA::Stream::SERIAL);
    INFO(shape);

    AND_THEN("H2HC") {
        Memory::PtrHost<TestType> h_half(elements_fft);
        Memory::PtrHost<TestType> h_half_centered(elements_fft);
        Test::initDataRandom(h_half.get(), h_half.elements(), randomizer_data);
        Test::initDataZero(h_half_centered.get(), h_half_centered.elements());
        Fourier::H2HC(h_half.get(), h_half_centered.get(), shape);

        AND_THEN("contiguous") {
            CUDA::Memory::PtrDevice<TestType> d_half(elements_fft);
            CUDA::Memory::PtrDevice<TestType> d_half_centered(elements_fft);
            Memory::PtrHost<TestType> h_half_centered_cuda(elements_fft);

            CUDA::Memory::copy(h_half.get(), d_half.get(), h_half.size(), stream);
            CUDA::Fourier::H2HC(d_half.get(), d_half_centered.get(), shape, 1, stream);
            CUDA::Memory::copy(d_half_centered.get(), h_half_centered_cuda.get(), h_half.size(), stream);
            CUDA::Stream::synchronize(stream);

            TestType diff = Test::getAverageDifference(h_half_centered.get(), h_half_centered_cuda.get(), elements_fft);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-14));
        }

        AND_THEN("padded") {
            CUDA::Memory::PtrDevicePadded<TestType> d_half(shape_fft);
            CUDA::Memory::PtrDevicePadded<TestType> d_half_centered(shape_fft);
            Memory::PtrHost<TestType> h_half_centered_cuda(elements_fft);

            CUDA::Memory::copy(h_half.get(), shape_fft.x,
                               d_half.get(), d_half.pitch(),
                               shape_fft, stream);
            CUDA::Fourier::H2HC(d_half.get(), d_half.pitch(),
                               d_half_centered.get(), d_half_centered.pitch(),
                               shape, 1, stream);
            CUDA::Memory::copy(d_half_centered.get(), d_half_centered.pitch(),
                               h_half_centered_cuda.get(), shape_fft.x,
                               shape_fft, stream);
            CUDA::Stream::synchronize(stream);

            TestType diff = Test::getAverageDifference(h_half_centered.get(), h_half_centered_cuda.get(), elements_fft);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-14));
        }
    }

    AND_THEN("HC2H") {
        Memory::PtrHost<TestType> h_half_centered(elements_fft);
        Memory::PtrHost<TestType> h_half(elements_fft);
        Test::initDataRandom(h_half_centered.get(), h_half_centered.elements(), randomizer_data);
        Test::initDataZero(h_half.get(), h_half.elements());
        Fourier::HC2H(h_half_centered.get(), h_half.get(), shape);

        AND_THEN("contiguous") {
            CUDA::Memory::PtrDevice<TestType> d_half_centered(elements_fft);
            CUDA::Memory::PtrDevice<TestType> d_half(elements_fft);
            Memory::PtrHost<TestType> h_half_cuda(elements_fft);

            CUDA::Memory::copy(h_half_centered.get(), d_half_centered.get(), h_half.size(), stream);
            CUDA::Fourier::HC2H(d_half_centered.get(), d_half.get(), shape, 1, stream);
            CUDA::Memory::copy(d_half.get(), h_half_cuda.get(), h_half.size(), stream);
            CUDA::Stream::synchronize(stream);

            TestType diff = Test::getAverageDifference(h_half.get(), h_half_cuda.get(), elements_fft);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-14));
        }

        AND_THEN("padded") {
            CUDA::Memory::PtrDevicePadded<TestType> d_half_centered(shape_fft);
            CUDA::Memory::PtrDevicePadded<TestType> d_half(shape_fft);
            Memory::PtrHost<TestType> h_half_cuda(elements_fft);

            CUDA::Memory::copy(h_half_centered.get(), shape_fft.x,
                               d_half_centered.get(), d_half_centered.pitch(),
                               shape_fft, stream);
            CUDA::Fourier::HC2H(d_half_centered.get(), d_half_centered.pitch(),
                                d_half.get(), d_half.pitch(),
                                shape, 1, stream);
            CUDA::Memory::copy(d_half.get(), d_half.pitch(),
                               h_half_cuda.get(), shape_fft.x,
                               shape_fft, stream);
            CUDA::Stream::synchronize(stream);

            TestType diff = Test::getAverageDifference(h_half.get(), h_half_cuda.get(), elements_fft);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-14));
        }
    }
}

TEMPLATE_TEST_CASE("CUDA::Fourier: FC2H", "[noa][cuda][fourier]", float, cfloat_t) {
    Test::RealRandomizer<TestType> randomizer_data(1., 128.);

    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape = Test::getRandomShape(ndim);
    size3_t shape_fft = getShapeFFT(shape);
    size_t elements = getElements(shape);
    size_t elements_fft = getElements(shape_fft);

    CUDA::Stream stream(CUDA::Stream::SERIAL);
    INFO(shape);

    AND_THEN("FC2H") {
        Memory::PtrHost<TestType> h_full_centered(elements);
        Memory::PtrHost<TestType> h_half(elements_fft);
        Test::initDataRandom(h_full_centered.get(), h_full_centered.elements(), randomizer_data);
        Test::initDataZero(h_half.get(), h_half.elements());
        Fourier::FC2H(h_full_centered.get(), h_half.get(), shape);

        AND_THEN("contiguous") {
            CUDA::Memory::PtrDevice<TestType> d_full_centered(elements);
            CUDA::Memory::PtrDevice<TestType> d_half(elements_fft);
            Memory::PtrHost<TestType> h_half_cuda(elements_fft);

            CUDA::Memory::copy(h_full_centered.get(), d_full_centered.get(), h_full_centered.size(), stream);
            CUDA::Fourier::FC2H(d_full_centered.get(), d_half.get(), shape, 1, stream);
            CUDA::Memory::copy(d_half.get(), h_half_cuda.get(), h_half.size(), stream);
            CUDA::Stream::synchronize(stream);

            TestType diff = Test::getAverageDifference(h_half.get(), h_half_cuda.get(), elements_fft);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-14));
        }

        AND_THEN("padded") {
            CUDA::Memory::PtrDevicePadded<TestType> d_full_centered(shape);
            CUDA::Memory::PtrDevicePadded<TestType> d_half(shape_fft);
            Memory::PtrHost<TestType> h_half_cuda(elements_fft);

            CUDA::Memory::copy(h_full_centered.get(), shape.x,
                               d_full_centered.get(), d_full_centered.pitch(),
                               shape, stream);
            CUDA::Fourier::FC2H(d_full_centered.get(), d_full_centered.pitch(),
                                d_half.get(), d_half.pitch(),
                                shape, 1, stream);
            CUDA::Memory::copy(d_half.get(), d_half.pitch(),
                               h_half_cuda.get(), shape_fft.x,
                               shape_fft, stream);
            CUDA::Stream::synchronize(stream);

            TestType diff = Test::getAverageDifference(h_half.get(), h_half_cuda.get(), elements_fft);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-14));
        }
    }
}
