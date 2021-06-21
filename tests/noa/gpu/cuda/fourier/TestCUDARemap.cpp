#include <noa/gpu/cuda/fourier/Remap.h>

#include <noa/cpu/fourier/Remap.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("cuda::fourier: h2f <-> f2h", "[noa][cuda][fourier]", float, cfloat_t) {
    test::RealRandomizer<TestType> randomizer_data(1., 128.);

    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape = test::getRandomShape(ndim);
    size3_t shape_fft = getShapeFFT(shape);
    size_t elements = getElements(shape);
    size_t elements_fft = getElements(shape_fft);

    cuda::Stream stream(cuda::STREAM_SERIAL);
    INFO(shape);

    AND_THEN("h2f") {
        memory::PtrHost<TestType> h_half(elements_fft);
        memory::PtrHost<TestType> h_full(elements);
        test::initDataRandom(h_half.get(), h_half.elements(), randomizer_data);
        test::initDataZero(h_full.get(), h_full.elements());
        fourier::h2f(h_half.get(), h_full.get(), shape);

        AND_THEN("contiguous") {
            cuda::memory::PtrDevice<TestType> d_half(elements_fft);
            cuda::memory::PtrDevice<TestType> d_full(elements);
            memory::PtrHost<TestType> h_full_cuda(elements);

            cuda::memory::copy(h_half.get(), d_half.get(), h_half.size(), stream);
            cuda::fourier::h2f(d_half.get(), d_full.get(), shape, 1, stream);
            cuda::memory::copy(d_full.get(), h_full_cuda.get(), d_full.size(), stream);
            cuda::Stream::synchronize(stream);

            TestType diff = test::getAverageDifference(h_full.get(), h_full_cuda.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
        }

        AND_THEN("padded") {
            cuda::memory::PtrDevicePadded<TestType> d_half(shape_fft);
            cuda::memory::PtrDevicePadded<TestType> d_full(shape);
            memory::PtrHost<TestType> h_full_cuda(elements);

            cuda::memory::copy(h_half.get(), shape_fft.x, d_half.get(), d_half.pitch(), shape_fft, stream);
            cuda::fourier::h2f(d_half.get(), d_half.pitch(), d_full.get(), d_full.pitch(), shape, 1, stream);
            cuda::memory::copy(d_full.get(), d_full.pitch(), h_full_cuda.get(), shape.x, shape, stream);
            cuda::Stream::synchronize(stream);

            TestType diff = test::getAverageDifference(h_full.get(), h_full_cuda.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
        }
    }

    AND_THEN("f2h") {
        memory::PtrHost<TestType> h_full(elements);
        memory::PtrHost<TestType> h_half(elements_fft);
        test::initDataRandom(h_full.get(), h_full.elements(), randomizer_data);
        test::initDataZero(h_half.get(), h_half.elements());
        fourier::f2h(h_full.get(), h_half.get(), shape);

        AND_THEN("contiguous") {
            cuda::memory::PtrDevice<TestType> d_full(elements);
            cuda::memory::PtrDevice<TestType> d_half(elements_fft);
            memory::PtrHost<TestType> h_half_cuda(elements_fft);

            cuda::memory::copy(h_full.get(), d_full.get(), h_full.size(), stream);
            cuda::fourier::f2h(d_full.get(), d_half.get(), shape, 1, stream);
            cuda::memory::copy(d_half.get(), h_half_cuda.get(), d_half.size(), stream);
            cuda::Stream::synchronize(stream);

            TestType diff = test::getAverageDifference(h_half.get(), h_half_cuda.get(), elements_fft);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
        }

        AND_THEN("padded") {
            cuda::memory::PtrDevicePadded<TestType> d_full(shape);
            cuda::memory::PtrDevicePadded<TestType> d_half(shape_fft);
            memory::PtrHost<TestType> h_half_cuda(elements_fft);

            cuda::memory::copy(h_full.get(), shape.x, d_full.get(), d_full.pitch(), shape, stream);
            cuda::fourier::f2h(d_full.get(), d_full.pitch(), d_half.get(), d_half.pitch(), shape, 1, stream);
            cuda::memory::copy(d_half.get(), d_half.pitch(), h_half_cuda.get(), shape_fft.x, shape_fft, stream);
            cuda::Stream::synchronize(stream);

            TestType diff = test::getAverageDifference(h_half.get(), h_half_cuda.get(), elements_fft);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
        }
    }
}

TEMPLATE_TEST_CASE("cuda::fourier: f2fc <-> fc2f", "[noa][cuda][fourier]", float, cfloat_t) {
    test::RealRandomizer<TestType> randomizer_data(1., 128.);

    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape = test::getRandomShape(ndim);
    size_t elements = getElements(shape);

    cuda::Stream stream(cuda::STREAM_SERIAL);
    INFO(shape);

    AND_THEN("f2fc") {
        memory::PtrHost<TestType> h_full(elements);
        memory::PtrHost<TestType> h_full_centered(elements);
        test::initDataRandom(h_full.get(), h_full.elements(), randomizer_data);
        test::initDataZero(h_full_centered.get(), h_full_centered.elements());
        fourier::f2fc(h_full.get(), h_full_centered.get(), shape);

        AND_THEN("contiguous") {
            cuda::memory::PtrDevice<TestType> d_full(elements);
            cuda::memory::PtrDevice<TestType> d_full_centered(elements);
            memory::PtrHost<TestType> h_full_centered_cuda(elements);

            cuda::memory::copy(h_full.get(), d_full.get(), h_full.size(), stream);
            cuda::fourier::f2fc(d_full.get(), d_full_centered.get(), shape, 1, stream);
            cuda::memory::copy(d_full_centered.get(), h_full_centered_cuda.get(), h_full.size(), stream);
            cuda::Stream::synchronize(stream);

            TestType diff = test::getAverageDifference(h_full_centered.get(), h_full_centered_cuda.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
        }

        AND_THEN("padded") {
            cuda::memory::PtrDevicePadded<TestType> d_full(shape);
            cuda::memory::PtrDevicePadded<TestType> d_full_centered(shape);
            memory::PtrHost<TestType> h_full_centered_cuda(elements);

            cuda::memory::copy(h_full.get(), shape.x,
                               d_full.get(), d_full.pitch(), shape, stream);
            cuda::fourier::f2fc(d_full.get(), d_full.pitch(),
                                d_full_centered.get(), d_full_centered.pitch(),
                                shape, 1, stream);
            cuda::memory::copy(d_full_centered.get(), d_full_centered.pitch(),
                               h_full_centered_cuda.get(), shape.x, shape, stream);
            cuda::Stream::synchronize(stream);

            TestType diff = test::getAverageDifference(h_full_centered.get(), h_full_centered_cuda.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
        }
    }

    AND_THEN("f2fc") {
        memory::PtrHost<TestType> h_full_centered(elements);
        memory::PtrHost<TestType> h_full(elements);
        test::initDataRandom(h_full_centered.get(), h_full_centered.elements(), randomizer_data);
        test::initDataZero(h_full.get(), h_full.elements());
        fourier::fc2f(h_full_centered.get(), h_full.get(), shape);

        AND_THEN("contiguous") {
            cuda::memory::PtrDevice<TestType> d_full_centered(elements);
            cuda::memory::PtrDevice<TestType> d_full(elements);
            memory::PtrHost<TestType> h_full_cuda(elements);

            cuda::memory::copy(h_full_centered.get(), d_full_centered.get(), d_full_centered.elements(), stream);
            cuda::fourier::fc2f(d_full_centered.get(), d_full.get(), shape, 1, stream);
            cuda::memory::copy(d_full.get(), h_full_cuda.get(), h_full.elements(), stream);
            cuda::Stream::synchronize(stream);

            TestType diff = test::getAverageDifference(h_full.get(), h_full_cuda.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
        }

        AND_THEN("padded") {
            cuda::memory::PtrDevicePadded<TestType> d_full_centered(shape);
            cuda::memory::PtrDevicePadded<TestType> d_full(shape);
            memory::PtrHost<TestType> h_full_cuda(elements);

            cuda::memory::copy(h_full_centered.get(), shape.x,
                               d_full_centered.get(), d_full_centered.pitch(), shape, stream);
            cuda::fourier::fc2f(d_full_centered.get(), d_full_centered.pitch(),
                                d_full.get(), d_full.pitch(),
                                shape, 1, stream);
            cuda::memory::copy(d_full.get(), d_full.pitch(), h_full_cuda.get(), shape.x, shape, stream);
            cuda::Stream::synchronize(stream);

            TestType diff = test::getAverageDifference(h_full.get(), h_full_cuda.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
        }
    }
}

TEMPLATE_TEST_CASE("cuda::fourier: h2hc <-> hc2h", "[noa][cuda][fourier]", float, cfloat_t) {
    test::RealRandomizer<TestType> randomizer_data(1., 128.);

    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape = test::getRandomShape(ndim);
    size3_t shape_fft = getShapeFFT(shape);
    size_t elements_fft = getElements(shape_fft);

    cuda::Stream stream(cuda::STREAM_SERIAL);
    INFO(shape);

    AND_THEN("h2hc") {
        memory::PtrHost<TestType> h_half(elements_fft);
        memory::PtrHost<TestType> h_half_centered(elements_fft);
        test::initDataRandom(h_half.get(), h_half.elements(), randomizer_data);
        test::initDataZero(h_half_centered.get(), h_half_centered.elements());
        fourier::h2hc(h_half.get(), h_half_centered.get(), shape);

        AND_THEN("contiguous") {
            cuda::memory::PtrDevice<TestType> d_half(elements_fft);
            cuda::memory::PtrDevice<TestType> d_half_centered(elements_fft);
            memory::PtrHost<TestType> h_half_centered_cuda(elements_fft);

            cuda::memory::copy(h_half.get(), d_half.get(), h_half.size(), stream);
            cuda::fourier::h2hc(d_half.get(), d_half_centered.get(), shape, 1, stream);
            cuda::memory::copy(d_half_centered.get(), h_half_centered_cuda.get(), h_half.size(), stream);
            cuda::Stream::synchronize(stream);

            TestType diff = test::getAverageDifference(h_half_centered.get(), h_half_centered_cuda.get(), elements_fft);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
        }

        AND_THEN("padded") {
            cuda::memory::PtrDevicePadded<TestType> d_half(shape_fft);
            cuda::memory::PtrDevicePadded<TestType> d_half_centered(shape_fft);
            memory::PtrHost<TestType> h_half_centered_cuda(elements_fft);

            cuda::memory::copy(h_half.get(), shape_fft.x,
                               d_half.get(), d_half.pitch(),
                               shape_fft, stream);
            cuda::fourier::h2hc(d_half.get(), d_half.pitch(),
                               d_half_centered.get(), d_half_centered.pitch(),
                               shape, 1, stream);
            cuda::memory::copy(d_half_centered.get(), d_half_centered.pitch(),
                               h_half_centered_cuda.get(), shape_fft.x,
                               shape_fft, stream);
            cuda::Stream::synchronize(stream);

            TestType diff = test::getAverageDifference(h_half_centered.get(), h_half_centered_cuda.get(), elements_fft);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
        }
    }

    AND_THEN("hc2h") {
        memory::PtrHost<TestType> h_half_centered(elements_fft);
        memory::PtrHost<TestType> h_half(elements_fft);
        test::initDataRandom(h_half_centered.get(), h_half_centered.elements(), randomizer_data);
        test::initDataZero(h_half.get(), h_half.elements());
        fourier::hc2h(h_half_centered.get(), h_half.get(), shape);

        AND_THEN("contiguous") {
            cuda::memory::PtrDevice<TestType> d_half_centered(elements_fft);
            cuda::memory::PtrDevice<TestType> d_half(elements_fft);
            memory::PtrHost<TestType> h_half_cuda(elements_fft);

            cuda::memory::copy(h_half_centered.get(), d_half_centered.get(), h_half.size(), stream);
            cuda::fourier::hc2h(d_half_centered.get(), d_half.get(), shape, 1, stream);
            cuda::memory::copy(d_half.get(), h_half_cuda.get(), h_half.size(), stream);
            cuda::Stream::synchronize(stream);

            TestType diff = test::getAverageDifference(h_half.get(), h_half_cuda.get(), elements_fft);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
        }

        AND_THEN("padded") {
            cuda::memory::PtrDevicePadded<TestType> d_half_centered(shape_fft);
            cuda::memory::PtrDevicePadded<TestType> d_half(shape_fft);
            memory::PtrHost<TestType> h_half_cuda(elements_fft);

            cuda::memory::copy(h_half_centered.get(), shape_fft.x,
                               d_half_centered.get(), d_half_centered.pitch(),
                               shape_fft, stream);
            cuda::fourier::hc2h(d_half_centered.get(), d_half_centered.pitch(),
                                d_half.get(), d_half.pitch(),
                                shape, 1, stream);
            cuda::memory::copy(d_half.get(), d_half.pitch(),
                               h_half_cuda.get(), shape_fft.x,
                               shape_fft, stream);
            cuda::Stream::synchronize(stream);

            TestType diff = test::getAverageDifference(h_half.get(), h_half_cuda.get(), elements_fft);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
        }
    }
}

TEMPLATE_TEST_CASE("cuda::fourier: fc2h", "[noa][cuda][fourier]", float, cfloat_t) {
    test::RealRandomizer<TestType> randomizer_data(1., 128.);

    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape = test::getRandomShape(ndim);
    size3_t shape_fft = getShapeFFT(shape);
    size_t elements = getElements(shape);
    size_t elements_fft = getElements(shape_fft);

    cuda::Stream stream(cuda::STREAM_SERIAL);
    INFO(shape);

    AND_THEN("fc2h") {
        memory::PtrHost<TestType> h_full_centered(elements);
        memory::PtrHost<TestType> h_half(elements_fft);
        test::initDataRandom(h_full_centered.get(), h_full_centered.elements(), randomizer_data);
        test::initDataZero(h_half.get(), h_half.elements());
        fourier::fc2h(h_full_centered.get(), h_half.get(), shape);

        AND_THEN("contiguous") {
            cuda::memory::PtrDevice<TestType> d_full_centered(elements);
            cuda::memory::PtrDevice<TestType> d_half(elements_fft);
            memory::PtrHost<TestType> h_half_cuda(elements_fft);

            cuda::memory::copy(h_full_centered.get(), d_full_centered.get(), h_full_centered.size(), stream);
            cuda::fourier::fc2h(d_full_centered.get(), d_half.get(), shape, 1, stream);
            cuda::memory::copy(d_half.get(), h_half_cuda.get(), h_half.size(), stream);
            cuda::Stream::synchronize(stream);

            TestType diff = test::getAverageDifference(h_half.get(), h_half_cuda.get(), elements_fft);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
        }

        AND_THEN("padded") {
            cuda::memory::PtrDevicePadded<TestType> d_full_centered(shape);
            cuda::memory::PtrDevicePadded<TestType> d_half(shape_fft);
            memory::PtrHost<TestType> h_half_cuda(elements_fft);

            cuda::memory::copy(h_full_centered.get(), shape.x,
                               d_full_centered.get(), d_full_centered.pitch(),
                               shape, stream);
            cuda::fourier::fc2h(d_full_centered.get(), d_full_centered.pitch(),
                                d_half.get(), d_half.pitch(),
                                shape, 1, stream);
            cuda::memory::copy(d_half.get(), d_half.pitch(),
                               h_half_cuda.get(), shape_fft.x,
                               shape_fft, stream);
            cuda::Stream::synchronize(stream);

            TestType diff = test::getAverageDifference(h_half.get(), h_half_cuda.get(), elements_fft);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
        }
    }
}
